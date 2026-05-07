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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif

#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
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

using Complex = std::complex<double>;

namespace {

struct TensorSpec {
  aclDataType dtype = ACL_FLOAT;
  aclFormat format = ACL_FORMAT_ND;
  std::vector<int64_t> shape;
  std::vector<Complex> values;
};

struct ScalarSpec {
  aclDataType dtype = ACL_FLOAT;
  Complex value = Complex(0.0, 0.0);
};

size_t GetShapeSize(const std::vector<int64_t>& shape) {
  size_t shape_size = 1;
  for (int64_t dim : shape) {
    shape_size *= static_cast<size_t>(dim);
  }
  return shape_size;
}

size_t GetTypeSize(aclDataType dtype) {
  switch (dtype) {
    case ACL_BOOL:
      return sizeof(bool);
    case ACL_INT8:
      return sizeof(int8_t);
    case ACL_UINT8:
      return sizeof(uint8_t);
    case ACL_INT16:
      return sizeof(int16_t);
    case ACL_UINT16:
      return sizeof(uint16_t);
    case ACL_INT32:
      return sizeof(int32_t);
    case ACL_INT64:
      return sizeof(int64_t);
    case ACL_FLOAT16:
      return sizeof(aclFloat16);
    case ACL_BF16:
      return sizeof(uint16_t);
    case ACL_FLOAT:
      return sizeof(float);
    case ACL_DOUBLE:
      return sizeof(double);
    case ACL_COMPLEX64:
      return sizeof(std::complex<float>);
    case ACL_COMPLEX128:
      return sizeof(std::complex<double>);
    default:
      return 0;
  }
}

bool IsFloatingDtype(aclDataType dtype) {
  return dtype == ACL_FLOAT16 || dtype == ACL_BF16 || dtype == ACL_FLOAT || dtype == ACL_DOUBLE;
}

bool IsComplexDtype(aclDataType dtype) {
  return dtype == ACL_COMPLEX64 || dtype == ACL_COMPLEX128;
}

bool IsIntegralDtype(aclDataType dtype) {
  return dtype == ACL_BOOL || dtype == ACL_INT8 || dtype == ACL_UINT8 || dtype == ACL_INT16 ||
         dtype == ACL_UINT16 || dtype == ACL_INT32 || dtype == ACL_INT64;
}

uint16_t FloatToBf16Bits(float value) {
  union {
    float f;
    uint32_t u;
  } data = {value};

  uint32_t bits = data.u;
  if (std::isnan(value)) {
    return 0x7fc0;
  }
  uint32_t lsb = (bits >> 16U) & 1U;
  bits += 0x7fffU + lsb;
  return static_cast<uint16_t>(bits >> 16U);
}

float Bf16BitsToFloat(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } data = {static_cast<uint32_t>(bits) << 16U};
  return data.f;
}

void EncodeValue(aclDataType dtype, const Complex& value, uint8_t* dst) {
  switch (dtype) {
    case ACL_BOOL: {
      bool casted = (value.real() != 0.0) || (value.imag() != 0.0);
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_INT8: {
      int8_t casted = static_cast<int8_t>(value.real());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_UINT8: {
      uint8_t casted = static_cast<uint8_t>(value.real());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_INT16: {
      int16_t casted = static_cast<int16_t>(value.real());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_UINT16: {
      uint16_t casted = static_cast<uint16_t>(value.real());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_INT32: {
      int32_t casted = static_cast<int32_t>(value.real());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_INT64: {
      int64_t casted = static_cast<int64_t>(value.real());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_FLOAT16: {
      aclFloat16 casted = aclFloatToFloat16(static_cast<float>(value.real()));
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_BF16: {
      uint16_t casted = FloatToBf16Bits(static_cast<float>(value.real()));
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_FLOAT: {
      float casted = static_cast<float>(value.real());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_DOUBLE: {
      double casted = value.real();
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_COMPLEX64: {
      std::complex<float> casted(static_cast<float>(value.real()), static_cast<float>(value.imag()));
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    case ACL_COMPLEX128: {
      std::complex<double> casted(value.real(), value.imag());
      std::memcpy(dst, &casted, sizeof(casted));
      return;
    }
    default:
      return;
  }
}

Complex DecodeValue(aclDataType dtype, const uint8_t* src) {
  switch (dtype) {
    case ACL_BOOL: {
      bool casted = false;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(casted ? 1.0 : 0.0, 0.0);
    }
    case ACL_INT8: {
      int8_t casted = 0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted), 0.0);
    }
    case ACL_UINT8: {
      uint8_t casted = 0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted), 0.0);
    }
    case ACL_INT16: {
      int16_t casted = 0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted), 0.0);
    }
    case ACL_UINT16: {
      uint16_t casted = 0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted), 0.0);
    }
    case ACL_INT32: {
      int32_t casted = 0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted), 0.0);
    }
    case ACL_INT64: {
      int64_t casted = 0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted), 0.0);
    }
    case ACL_FLOAT16: {
      aclFloat16 casted = 0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(aclFloat16ToFloat(casted)), 0.0);
    }
    case ACL_BF16: {
      uint16_t bits = 0;
      std::memcpy(&bits, src, sizeof(bits));
      return Complex(static_cast<double>(Bf16BitsToFloat(bits)), 0.0);
    }
    case ACL_FLOAT: {
      float casted = 0.0F;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted), 0.0);
    }
    case ACL_DOUBLE: {
      double casted = 0.0;
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(casted, 0.0);
    }
    case ACL_COMPLEX64: {
      std::complex<float> casted(0.0F, 0.0F);
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(static_cast<double>(casted.real()), static_cast<double>(casted.imag()));
    }
    case ACL_COMPLEX128: {
      std::complex<double> casted(0.0, 0.0);
      std::memcpy(&casted, src, sizeof(casted));
      return Complex(casted.real(), casted.imag());
    }
    default:
      return Complex(0.0, 0.0);
  }
}

std::vector<uint8_t> EncodeVector(aclDataType dtype, const std::vector<Complex>& values) {
  std::vector<uint8_t> bytes(values.size() * GetTypeSize(dtype), 0);
  size_t type_size = GetTypeSize(dtype);
  for (size_t i = 0; i < values.size(); ++i) {
    EncodeValue(dtype, values[i], bytes.data() + i * type_size);
  }
  return bytes;
}

std::vector<Complex> DecodeVector(aclDataType dtype, const std::vector<uint8_t>& bytes) {
  size_t type_size = GetTypeSize(dtype);
  if (type_size == 0) {
    return {};
  }
  size_t element_count = bytes.size() / type_size;
  std::vector<Complex> values(element_count);
  for (size_t i = 0; i < element_count; ++i) {
    values[i] = DecodeValue(dtype, bytes.data() + i * type_size);
  }
  return values;
}

Complex QuantizeValue(const Complex& value, aclDataType dtype) {
  std::vector<uint8_t> bytes(GetTypeSize(dtype), 0);
  EncodeValue(dtype, value, bytes.data());
  return DecodeValue(dtype, bytes.data());
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

size_t GetBroadcastIndex(const std::vector<int64_t>& out_shape,
                         const std::vector<int64_t>& in_shape,
                         size_t out_linear_index) {
  if (in_shape.empty()) {
    return 0;
  }

  std::vector<int64_t> out_coords(out_shape.size(), 0);
  size_t cursor = out_linear_index;
  for (int64_t i = static_cast<int64_t>(out_shape.size()) - 1; i >= 0; --i) {
    size_t dim_index = static_cast<size_t>(i);
    int64_t dim = out_shape[dim_index];
    if (dim != 0) {
      out_coords[dim_index] = static_cast<int64_t>(cursor % static_cast<size_t>(dim));
      cursor /= static_cast<size_t>(dim);
    }
  }

  std::vector<int64_t> in_strides = MakeContiguousStrides(in_shape);
  size_t in_linear_index = 0;
  size_t out_dims = out_shape.size();
  size_t in_dims = in_shape.size();
  for (size_t i = 0; i < in_dims; ++i) {
    size_t out_dim = out_dims - in_dims + i;
    int64_t coord = in_shape[i] == 1 ? 0 : out_coords[out_dim];
    in_linear_index += static_cast<size_t>(coord) * static_cast<size_t>(in_strides[i]);
  }
  return in_linear_index;
}

double GetAtol(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT16:
      return 1e-3;
    case ACL_BF16:
      return 1e-2;
    case ACL_FLOAT:
    case ACL_COMPLEX64:
      return 1e-5;
    case ACL_DOUBLE:
    case ACL_COMPLEX128:
      return 1e-10;
    default:
      return 0.0;
  }
}

double GetRtol(aclDataType dtype) {
  return GetAtol(dtype);
}

bool CompareDouble(double actual, double expected, double atol, double rtol) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  if (std::isinf(expected)) {
    return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
  }
  return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

std::string ComplexToString(const Complex& value) {
  char buffer[128] = {0};
  std::snprintf(buffer, sizeof(buffer), "(%.10f, %.10f)", value.real(), value.imag());
  return std::string(buffer);
}

bool CompareValue(const Complex& actual, const Complex& expected, aclDataType dtype, std::string* detail) {
  if (IsIntegralDtype(dtype)) {
    bool passed = actual.real() == expected.real() && actual.imag() == expected.imag();
    if (!passed && detail != nullptr) {
      *detail = "expected " + ComplexToString(expected) + ", got " + ComplexToString(actual);
    }
    return passed;
  }

  double atol = GetAtol(dtype);
  double rtol = GetRtol(dtype);
  bool real_ok = CompareDouble(actual.real(), expected.real(), atol, rtol);
  bool imag_ok = CompareDouble(actual.imag(), expected.imag(), atol, rtol);
  if (!(real_ok && imag_ok) && detail != nullptr) {
    *detail = "expected " + ComplexToString(expected) + ", got " + ComplexToString(actual);
  }
  return real_ok && imag_ok;
}

class RuntimeContext {
 public:
  int Init() {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(device_id_);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream_);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
  }

  void Finalize() {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
    aclrtResetDevice(device_id_);
    aclFinalize();
  }

  aclrtStream Stream() const { return stream_; }

 private:
  int32_t device_id_ = 0;
  aclrtStream stream_ = nullptr;
};

class TensorHandle {
 public:
  TensorHandle() = default;

  int Init(const TensorSpec& spec) {
    spec_ = spec;
    size_t expected_count = GetShapeSize(spec.shape);
    CHECK_RET(spec.values.size() == expected_count,
              LOG_PRINT("tensor value size mismatch, expected=%zu actual=%zu\n", expected_count, spec.values.size());
              return ACL_ERROR_FAILURE);

    host_bytes_ = EncodeVector(spec.dtype, spec.values);
    if (!host_bytes_.empty()) {
      auto ret = aclrtMalloc(&device_addr_, host_bytes_.size(), ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(device_addr_, host_bytes_.size(), host_bytes_.data(), host_bytes_.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides = MakeContiguousStrides(spec.shape);
    tensor_ = aclCreateTensor(spec.shape.empty() ? nullptr : spec.shape.data(), spec.shape.size(), spec.dtype,
                              strides.empty() ? nullptr : strides.data(), 0, spec.format,
                              spec.shape.empty() ? nullptr : spec.shape.data(), spec.shape.size(), device_addr_);
    CHECK_RET(tensor_ != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
  }

  std::vector<Complex> DecodeInputValues() const { return DecodeVector(spec_.dtype, host_bytes_); }

  std::vector<Complex> CopyBack() const {
    std::vector<uint8_t> output(host_bytes_.size(), 0);
    if (!output.empty()) {
      auto ret = aclrtMemcpy(output.data(), output.size(), device_addr_, output.size(), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy device to host failed. ERROR: %d\n", ret); return {});
    }
    return DecodeVector(spec_.dtype, output);
  }

  aclTensor* Tensor() const { return tensor_; }
  aclDataType Dtype() const { return spec_.dtype; }
  const std::vector<int64_t>& Shape() const { return spec_.shape; }

  void Destroy() {
    if (tensor_ != nullptr) {
      aclDestroyTensor(tensor_);
      tensor_ = nullptr;
    }
    if (device_addr_ != nullptr) {
      aclrtFree(device_addr_);
      device_addr_ = nullptr;
    }
    host_bytes_.clear();
  }

  ~TensorHandle() { Destroy(); }

 private:
  TensorSpec spec_;
  std::vector<uint8_t> host_bytes_;
  void* device_addr_ = nullptr;
  aclTensor* tensor_ = nullptr;
};

class ScalarHandle {
 public:
  int Init(const ScalarSpec& spec) {
    spec_ = spec;
    bytes_.resize(GetTypeSize(spec.dtype), 0);
    CHECK_RET(!bytes_.empty(), LOG_PRINT("unsupported scalar dtype: %d\n", spec.dtype); return ACL_ERROR_FAILURE);
    EncodeValue(spec.dtype, spec.value, bytes_.data());
    scalar_ = aclCreateScalar(bytes_.data(), spec.dtype);
    CHECK_RET(scalar_ != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
  }

  Complex DecodeValue() const { return ::DecodeValue(spec_.dtype, bytes_.data()); }
  aclScalar* Scalar() const { return scalar_; }

  void Destroy() {
    if (scalar_ != nullptr) {
      aclDestroyScalar(scalar_);
      scalar_ = nullptr;
    }
    bytes_.clear();
  }

  ~ScalarHandle() { Destroy(); }

 private:
  ScalarSpec spec_;
  std::vector<uint8_t> bytes_;
  aclScalar* scalar_ = nullptr;
};

class WorkspaceHandle {
 public:
  int Init(uint64_t size) {
    size_ = size;
    if (size_ == 0) {
      return ACL_SUCCESS;
    }
    auto ret = aclrtMalloc(&addr_, size_, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
  }

  void* Addr() const { return addr_; }
  uint64_t Size() const { return size_; }

  void Destroy() {
    if (addr_ != nullptr) {
      aclrtFree(addr_);
      addr_ = nullptr;
    }
    size_ = 0;
  }

  ~WorkspaceHandle() { Destroy(); }

 private:
  void* addr_ = nullptr;
  uint64_t size_ = 0;
};

void DestroyExecutor(aclOpExecutor** executor) {
  if (executor != nullptr && *executor != nullptr) {
    aclDestroyAclOpExecutor(*executor);
    *executor = nullptr;
  }
}

TensorSpec MakeZeroTensor(aclDataType dtype, const std::vector<int64_t>& shape, aclFormat format = ACL_FORMAT_ND) {
  return TensorSpec{dtype, format, shape, std::vector<Complex>(GetShapeSize(shape), Complex(0.0, 0.0))};
}

std::vector<Complex> ComputeMulExpected(const TensorHandle& self,
                                        const TensorHandle& other,
                                        aclDataType out_dtype,
                                        const std::vector<int64_t>& out_shape) {
  std::vector<Complex> self_values = self.DecodeInputValues();
  std::vector<Complex> other_values = other.DecodeInputValues();
  std::vector<Complex> expected(GetShapeSize(out_shape));
  for (size_t i = 0; i < expected.size(); ++i) {
    size_t self_index = GetBroadcastIndex(out_shape, self.Shape(), i);
    size_t other_index = GetBroadcastIndex(out_shape, other.Shape(), i);
    expected[i] = QuantizeValue(self_values[self_index] * other_values[other_index], out_dtype);
  }
  return expected;
}

std::vector<Complex> ComputeMulsExpected(const TensorHandle& self,
                                         const ScalarHandle& scalar,
                                         aclDataType out_dtype,
                                         const std::vector<int64_t>& out_shape) {
  std::vector<Complex> self_values = self.DecodeInputValues();
  Complex scalar_value = scalar.DecodeValue();
  std::vector<Complex> expected(GetShapeSize(out_shape));
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = QuantizeValue(self_values[i] * scalar_value, out_dtype);
  }
  return expected;
}

bool ValidateOutput(const std::string& name,
                    const std::vector<Complex>& actual,
                    const std::vector<Complex>& expected,
                    aclDataType dtype,
                    std::string* detail) {
  if (actual.size() != expected.size()) {
    if (detail != nullptr) {
      *detail = "output size mismatch";
    }
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!CompareValue(actual[i], expected[i], dtype, detail)) {
      if (detail != nullptr) {
        *detail = name + " idx=" + std::to_string(i) + " " + *detail;
      }
      return false;
    }
  }
  return true;
}

bool ReportCase(const std::string& name, bool passed, const std::string& detail = "") {
  if (passed) {
    LOG_PRINT("[PASS] %s\n", name.c_str());
  } else {
    LOG_PRINT("[FAIL] %s: %s\n", name.c_str(), detail.c_str());
  }
  return passed;
}

bool ExpectStatus(const std::string& name, aclnnStatus actual, aclnnStatus expected) {
  if (actual == expected) {
    return ReportCase(name, true);
  }
  return ReportCase(name, false,
                    "expected status=" + std::to_string(expected) + ", got=" + std::to_string(actual));
}

bool RunMulCase(RuntimeContext* runtime,
                const std::string& name,
                const TensorSpec& self_spec,
                const TensorSpec& other_spec,
                const TensorSpec& out_spec) {
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(other.Init(other_spec) == ACL_SUCCESS, return ReportCase(name, false, "other init failed"));
  CHECK_RET(out.Init(out_spec) == ACL_SUCCESS, return ReportCase(name, false, "out init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(self.Tensor(), other.Tensor(), out.Tensor(), &workspace_size, &executor);
  if (ret != ACLNN_SUCCESS) {
    DestroyExecutor(&executor);
    return ReportCase(name, false, "aclnnMulGetWorkspaceSize failed: " + std::to_string(ret));
  }

  WorkspaceHandle workspace;
  CHECK_RET(workspace.Init(workspace_size) == ACL_SUCCESS,
            DestroyExecutor(&executor); return ReportCase(name, false, "workspace init failed"));

  ret = aclnnMul(workspace.Addr(), workspace.Size(), executor, runtime->Stream());
  if (ret != ACLNN_SUCCESS) {
    return ReportCase(name, false, "aclnnMul failed: " + std::to_string(ret));
  }
  ret = aclrtSynchronizeStream(runtime->Stream());
  if (ret != ACL_SUCCESS) {
    return ReportCase(name, false, "aclrtSynchronizeStream failed: " + std::to_string(ret));
  }

  std::vector<Complex> actual = out.CopyBack();
  std::vector<Complex> expected = ComputeMulExpected(self, other, out.Dtype(), out.Shape());
  std::string detail;
  bool passed = ValidateOutput(name, actual, expected, out.Dtype(), &detail);
  return ReportCase(name, passed, detail);
}

bool RunMulExecuteExpectStatus(RuntimeContext* runtime,
                               const std::string& name,
                               const TensorSpec& self_spec,
                               const TensorSpec& other_spec,
                               const TensorSpec& out_spec,
                               aclnnStatus expected_status) {
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(other.Init(other_spec) == ACL_SUCCESS, return ReportCase(name, false, "other init failed"));
  CHECK_RET(out.Init(out_spec) == ACL_SUCCESS, return ReportCase(name, false, "out init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(self.Tensor(), other.Tensor(), out.Tensor(), &workspace_size, &executor);
  if (ret != ACLNN_SUCCESS) {
    DestroyExecutor(&executor);
    return ReportCase(name, false, "aclnnMulGetWorkspaceSize failed: " + std::to_string(ret));
  }

  WorkspaceHandle workspace;
  CHECK_RET(workspace.Init(workspace_size) == ACL_SUCCESS,
            DestroyExecutor(&executor); return ReportCase(name, false, "workspace init failed"));

  ret = aclnnMul(workspace.Addr(), workspace.Size(), executor, runtime->Stream());
  return ExpectStatus(name, ret, expected_status);
}

bool RunMulsCase(RuntimeContext* runtime,
                 const std::string& name,
                 const TensorSpec& self_spec,
                 const ScalarSpec& scalar_spec,
                 const TensorSpec& out_spec) {
  TensorHandle self;
  TensorHandle out;
  ScalarHandle scalar;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(scalar.Init(scalar_spec) == ACL_SUCCESS, return ReportCase(name, false, "scalar init failed"));
  CHECK_RET(out.Init(out_spec) == ACL_SUCCESS, return ReportCase(name, false, "out init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulsGetWorkspaceSize(self.Tensor(), scalar.Scalar(), out.Tensor(), &workspace_size, &executor);
  if (ret != ACLNN_SUCCESS) {
    DestroyExecutor(&executor);
    return ReportCase(name, false, "aclnnMulsGetWorkspaceSize failed: " + std::to_string(ret));
  }

  WorkspaceHandle workspace;
  CHECK_RET(workspace.Init(workspace_size) == ACL_SUCCESS,
            DestroyExecutor(&executor); return ReportCase(name, false, "workspace init failed"));

  ret = aclnnMuls(workspace.Addr(), workspace.Size(), executor, runtime->Stream());
  if (ret != ACLNN_SUCCESS) {
    return ReportCase(name, false, "aclnnMuls failed: " + std::to_string(ret));
  }
  ret = aclrtSynchronizeStream(runtime->Stream());
  if (ret != ACL_SUCCESS) {
    return ReportCase(name, false, "aclrtSynchronizeStream failed: " + std::to_string(ret));
  }

  std::vector<Complex> actual = out.CopyBack();
  std::vector<Complex> expected = ComputeMulsExpected(self, scalar, out.Dtype(), out.Shape());
  std::string detail;
  bool passed = ValidateOutput(name, actual, expected, out.Dtype(), &detail);
  return ReportCase(name, passed, detail);
}

bool RunInplaceMulExecuteExpectStatus(RuntimeContext* runtime,
                                      const std::string& name,
                                      const TensorSpec& self_spec,
                                      const TensorSpec& other_spec,
                                      aclnnStatus expected_status) {
  TensorHandle self;
  TensorHandle other;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(other.Init(other_spec) == ACL_SUCCESS, return ReportCase(name, false, "other init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulGetWorkspaceSize(self.Tensor(), other.Tensor(), &workspace_size, &executor);
  if (ret != ACLNN_SUCCESS) {
    DestroyExecutor(&executor);
    return ReportCase(name, false, "aclnnInplaceMulGetWorkspaceSize failed: " + std::to_string(ret));
  }

  WorkspaceHandle workspace;
  CHECK_RET(workspace.Init(workspace_size) == ACL_SUCCESS,
            DestroyExecutor(&executor); return ReportCase(name, false, "workspace init failed"));

  ret = aclnnInplaceMul(workspace.Addr(), workspace.Size(), executor, runtime->Stream());
  return ExpectStatus(name, ret, expected_status);
}

bool RunInplaceMulCase(RuntimeContext* runtime,
                       const std::string& name,
                       const TensorSpec& self_spec,
                       const TensorSpec& other_spec) {
  TensorHandle self;
  TensorHandle other;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(other.Init(other_spec) == ACL_SUCCESS, return ReportCase(name, false, "other init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulGetWorkspaceSize(self.Tensor(), other.Tensor(), &workspace_size, &executor);
  if (ret != ACLNN_SUCCESS) {
    DestroyExecutor(&executor);
    return ReportCase(name, false, "aclnnInplaceMulGetWorkspaceSize failed: " + std::to_string(ret));
  }

  WorkspaceHandle workspace;
  CHECK_RET(workspace.Init(workspace_size) == ACL_SUCCESS,
            DestroyExecutor(&executor); return ReportCase(name, false, "workspace init failed"));

  ret = aclnnInplaceMul(workspace.Addr(), workspace.Size(), executor, runtime->Stream());
  if (ret != ACLNN_SUCCESS) {
    return ReportCase(name, false, "aclnnInplaceMul failed: " + std::to_string(ret));
  }
  ret = aclrtSynchronizeStream(runtime->Stream());
  if (ret != ACL_SUCCESS) {
    return ReportCase(name, false, "aclrtSynchronizeStream failed: " + std::to_string(ret));
  }

  std::vector<Complex> actual = self.CopyBack();
  std::vector<Complex> expected = ComputeMulExpected(self, other, self.Dtype(), self.Shape());
  std::string detail;
  bool passed = ValidateOutput(name, actual, expected, self.Dtype(), &detail);
  return ReportCase(name, passed, detail);
}

bool RunInplaceMulsExecuteExpectStatus(RuntimeContext* runtime,
                                       const std::string& name,
                                       const TensorSpec& self_spec,
                                       const ScalarSpec& scalar_spec,
                                       aclnnStatus expected_status) {
  TensorHandle self;
  ScalarHandle scalar;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(scalar.Init(scalar_spec) == ACL_SUCCESS, return ReportCase(name, false, "scalar init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulsGetWorkspaceSize(self.Tensor(), scalar.Scalar(), &workspace_size, &executor);
  if (ret != ACLNN_SUCCESS) {
    DestroyExecutor(&executor);
    return ReportCase(name, false, "aclnnInplaceMulsGetWorkspaceSize failed: " + std::to_string(ret));
  }

  WorkspaceHandle workspace;
  CHECK_RET(workspace.Init(workspace_size) == ACL_SUCCESS,
            DestroyExecutor(&executor); return ReportCase(name, false, "workspace init failed"));

  ret = aclnnInplaceMuls(workspace.Addr(), workspace.Size(), executor, runtime->Stream());
  return ExpectStatus(name, ret, expected_status);
}

bool RunInplaceMulsCase(RuntimeContext* runtime,
                        const std::string& name,
                        const TensorSpec& self_spec,
                        const ScalarSpec& scalar_spec) {
  TensorHandle self;
  ScalarHandle scalar;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(scalar.Init(scalar_spec) == ACL_SUCCESS, return ReportCase(name, false, "scalar init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulsGetWorkspaceSize(self.Tensor(), scalar.Scalar(), &workspace_size, &executor);
  if (ret != ACLNN_SUCCESS) {
    DestroyExecutor(&executor);
    return ReportCase(name, false, "aclnnInplaceMulsGetWorkspaceSize failed: " + std::to_string(ret));
  }

  WorkspaceHandle workspace;
  CHECK_RET(workspace.Init(workspace_size) == ACL_SUCCESS,
            DestroyExecutor(&executor); return ReportCase(name, false, "workspace init failed"));

  ret = aclnnInplaceMuls(workspace.Addr(), workspace.Size(), executor, runtime->Stream());
  if (ret != ACLNN_SUCCESS) {
    return ReportCase(name, false, "aclnnInplaceMuls failed: " + std::to_string(ret));
  }
  ret = aclrtSynchronizeStream(runtime->Stream());
  if (ret != ACL_SUCCESS) {
    return ReportCase(name, false, "aclrtSynchronizeStream failed: " + std::to_string(ret));
  }

  std::vector<Complex> actual = self.CopyBack();
  std::vector<Complex> expected = ComputeMulsExpected(self, scalar, self.Dtype(), self.Shape());
  std::string detail;
  bool passed = ValidateOutput(name, actual, expected, self.Dtype(), &detail);
  return ReportCase(name, passed, detail);
}

bool RunMulWorkspaceOnly(const std::string& name,
                         const TensorSpec& self_spec,
                         const TensorSpec& other_spec,
                         const TensorSpec& out_spec,
                         aclnnStatus expected_status) {
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(other.Init(other_spec) == ACL_SUCCESS, return ReportCase(name, false, "other init failed"));
  CHECK_RET(out.Init(out_spec) == ACL_SUCCESS, return ReportCase(name, false, "out init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnMulGetWorkspaceSize(self.Tensor(), other.Tensor(), out.Tensor(), &workspace_size, &executor);
  bool passed = ExpectStatus(name, ret, expected_status);
  DestroyExecutor(&executor);
  return passed;
}

bool RunMulsWorkspaceOnly(const std::string& name,
                          const TensorSpec& self_spec,
                          const ScalarSpec& scalar_spec,
                          const TensorSpec& out_spec,
                          aclnnStatus expected_status) {
  TensorHandle self;
  TensorHandle out;
  ScalarHandle scalar;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(scalar.Init(scalar_spec) == ACL_SUCCESS, return ReportCase(name, false, "scalar init failed"));
  CHECK_RET(out.Init(out_spec) == ACL_SUCCESS, return ReportCase(name, false, "out init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnMulsGetWorkspaceSize(self.Tensor(), scalar.Scalar(), out.Tensor(), &workspace_size, &executor);
  bool passed = ExpectStatus(name, ret, expected_status);
  DestroyExecutor(&executor);
  return passed;
}

bool RunInplaceMulWorkspaceOnly(const std::string& name,
                                const TensorSpec& self_spec,
                                const TensorSpec& other_spec,
                                aclnnStatus expected_status) {
  TensorHandle self;
  TensorHandle other;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(other.Init(other_spec) == ACL_SUCCESS, return ReportCase(name, false, "other init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnInplaceMulGetWorkspaceSize(self.Tensor(), other.Tensor(), &workspace_size, &executor);
  bool passed = ExpectStatus(name, ret, expected_status);
  DestroyExecutor(&executor);
  return passed;
}

bool RunInplaceMulsWorkspaceOnly(const std::string& name,
                                 const TensorSpec& self_spec,
                                 const ScalarSpec& scalar_spec,
                                 aclnnStatus expected_status) {
  TensorHandle self;
  ScalarHandle scalar;
  CHECK_RET(self.Init(self_spec) == ACL_SUCCESS, return ReportCase(name, false, "self init failed"));
  CHECK_RET(scalar.Init(scalar_spec) == ACL_SUCCESS, return ReportCase(name, false, "scalar init failed"));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnInplaceMulsGetWorkspaceSize(self.Tensor(), scalar.Scalar(), &workspace_size, &executor);
  bool passed = ExpectStatus(name, ret, expected_status);
  DestroyExecutor(&executor);
  return passed;
}

bool RunNullptrCase(const std::string& name, const std::function<aclnnStatus()>& invoke, aclnnStatus expected_status) {
  return ExpectStatus(name, invoke(), expected_status);
}

bool ShouldRunCase(const std::string& name) {
  const char* filter = std::getenv("MUL_CASE_FILTER");
  if (filter == nullptr || filter[0] == '\0') {
    return true;
  }
  std::string filter_text(filter);
  if (!filter_text.empty() && filter_text[0] == '=') {
    return name == filter_text.substr(1);
  }
  return name.find(filter_text) != std::string::npos;
}

template <typename Fn>
void RunCase(const std::string& name, Fn&& fn, int* total, int* failed) {
  if (!ShouldRunCase(name)) {
    return;
  }
  ++(*total);
  if (!fn()) {
    ++(*failed);
  }
}

}  // namespace

int main() {
  RuntimeContext runtime;
  auto ret = runtime.Init();
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int total = 0;
  int failed = 0;

  const std::vector<Complex> real_a = {1.5, -2.0, 0.5, 3.0, -4.5, 2.25};
  const std::vector<Complex> real_b = {2.0, -1.5, -3.0, 0.25, 4.0, -2.0};
  const std::vector<Complex> int_a = {2.0, -3.0, 4.0, -1.0, 0.0, 5.0};
  const std::vector<Complex> int_b = {-1.0, 2.0, 3.0, -4.0, 1.0, 2.0};
  const std::vector<Complex> uint_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const std::vector<Complex> uint_b = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  const std::vector<Complex> bool_a = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0};
  const std::vector<Complex> bool_b = {1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
  const std::vector<Complex> broadcast_b = {2.0, -1.0, 0.5};
  const std::vector<Complex> complex_a = {
      Complex(1.0, 2.0), Complex(-2.0, 0.5), Complex(0.0, -1.0),
      Complex(3.0, -4.0), Complex(2.0, 2.0), Complex(-1.0, -1.5)};
  const std::vector<Complex> complex_b = {
      Complex(0.5, -1.0), Complex(1.0, 1.0), Complex(-2.0, 0.0),
      Complex(0.0, 0.5), Complex(-1.5, 2.0), Complex(3.0, -0.5)};

  RunCase("mul_nullptr", [&]() {
    TensorHandle out;
    if (out.Init(MakeZeroTensor(ACL_FLOAT, {2, 3})) != ACL_SUCCESS) {
      return ReportCase("mul_nullptr", false, "out init failed");
    }
    return RunNullptrCase("mul_nullptr", [&]() {
      uint64_t workspace_size = 0;
      aclOpExecutor* executor = nullptr;
      aclnnStatus status = aclnnMulGetWorkspaceSize(nullptr, nullptr, out.Tensor(), &workspace_size, &executor);
      DestroyExecutor(&executor);
      return status;
    }, ACLNN_ERR_PARAM_NULLPTR);
  }, &total, &failed);

  RunCase("muls_nullptr", [&]() {
    TensorHandle out;
    if (out.Init(MakeZeroTensor(ACL_FLOAT, {2, 3})) != ACL_SUCCESS) {
      return ReportCase("muls_nullptr", false, "out init failed");
    }
    return RunNullptrCase("muls_nullptr", [&]() {
      uint64_t workspace_size = 0;
      aclOpExecutor* executor = nullptr;
      aclnnStatus status = aclnnMulsGetWorkspaceSize(nullptr, nullptr, out.Tensor(), &workspace_size, &executor);
      DestroyExecutor(&executor);
      return status;
    }, ACLNN_ERR_PARAM_NULLPTR);
  }, &total, &failed);

  RunCase("inplace_mul_nullptr", [&]() {
    return RunNullptrCase("inplace_mul_nullptr", [&]() {
      uint64_t workspace_size = 0;
      aclOpExecutor* executor = nullptr;
      aclnnStatus status = aclnnInplaceMulGetWorkspaceSize(nullptr, nullptr, &workspace_size, &executor);
      DestroyExecutor(&executor);
      return status;
    }, ACLNN_ERR_PARAM_NULLPTR);
  }, &total, &failed);

  RunCase("inplace_muls_nullptr", [&]() {
    return RunNullptrCase("inplace_muls_nullptr", [&]() {
      uint64_t workspace_size = 0;
      aclOpExecutor* executor = nullptr;
      aclnnStatus status = aclnnInplaceMulsGetWorkspaceSize(nullptr, nullptr, &workspace_size, &executor);
      DestroyExecutor(&executor);
      return status;
    }, ACLNN_ERR_PARAM_NULLPTR);
  }, &total, &failed);

  RunCase("mul_bad_dtype_uint16", [&]() {
    return RunMulWorkspaceOnly("mul_bad_dtype_uint16",
                               TensorSpec{ACL_UINT16, ACL_FORMAT_ND, {2, 3}, uint_a},
                               TensorSpec{ACL_UINT16, ACL_FORMAT_ND, {2, 3}, uint_b},
                               MakeZeroTensor(ACL_UINT16, {2, 3}),
                               ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("muls_bad_dtype_uint16", [&]() {
    return RunMulsWorkspaceOnly("muls_bad_dtype_uint16",
                                TensorSpec{ACL_UINT16, ACL_FORMAT_ND, {2, 3}, uint_a},
                                ScalarSpec{ACL_UINT16, Complex(2.0, 0.0)},
                                MakeZeroTensor(ACL_UINT16, {2, 3}),
                                ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("inplace_mul_bad_dtype_uint16", [&]() {
    return RunInplaceMulWorkspaceOnly("inplace_mul_bad_dtype_uint16",
                                      TensorSpec{ACL_UINT16, ACL_FORMAT_ND, {2, 3}, uint_a},
                                      TensorSpec{ACL_UINT16, ACL_FORMAT_ND, {2, 3}, uint_b},
                                      ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("inplace_muls_bad_dtype_uint16", [&]() {
    return RunInplaceMulsWorkspaceOnly("inplace_muls_bad_dtype_uint16",
                                       TensorSpec{ACL_UINT16, ACL_FORMAT_ND, {2, 3}, uint_a},
                                       ScalarSpec{ACL_UINT16, Complex(2.0, 0.0)},
                                       ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("mul_broadcast_invalid", [&]() {
    return RunMulWorkspaceOnly("mul_broadcast_invalid",
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 2}, {1.0, 2.0, 3.0, 4.0}},
                               MakeZeroTensor(ACL_FLOAT, {2, 3}),
                               ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("mul_out_shape_invalid", [&]() {
    return RunMulWorkspaceOnly("mul_out_shape_invalid",
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {3}, broadcast_b},
                               MakeZeroTensor(ACL_FLOAT, {2, 2}),
                               ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("muls_out_shape_invalid", [&]() {
    return RunMulsWorkspaceOnly("muls_out_shape_invalid",
                                TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                                ScalarSpec{ACL_FLOAT, Complex(1.5, 0.0)},
                                MakeZeroTensor(ACL_FLOAT, {3, 2}),
                                ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("inplace_mul_shape_invalid", [&]() {
    return RunInplaceMulWorkspaceOnly("inplace_mul_shape_invalid",
                                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {1, 3}, {1.0, 2.0, 3.0}},
                                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                                      ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  RunCase("mul_format_nchw_workspace", [&]() {
    return RunMulWorkspaceOnly("mul_format_nchw_workspace",
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_NCHW, {1, 1, 2, 3}, real_a},
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_NCHW, {1, 1, 2, 3}, real_b},
                               MakeZeroTensor(ACL_FLOAT, {1, 1, 2, 3}, ACL_FORMAT_NCHW),
                               ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("muls_format_nchw_workspace", [&]() {
    return RunMulsWorkspaceOnly("muls_format_nchw_workspace",
                                TensorSpec{ACL_FLOAT, ACL_FORMAT_NCHW, {1, 1, 2, 3}, real_a},
                                ScalarSpec{ACL_FLOAT, Complex(2.0, 0.0)},
                                MakeZeroTensor(ACL_FLOAT, {1, 1, 2, 3}, ACL_FORMAT_NCHW),
                                ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("mul_empty_workspace", [&]() {
    return RunMulWorkspaceOnly("mul_empty_workspace",
                               TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {0}, {}},
                               TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {0}, {}},
                               MakeZeroTensor(ACL_FLOAT16, {0}),
                               ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("muls_empty_workspace", [&]() {
    return RunMulsWorkspaceOnly("muls_empty_workspace",
                                TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {0}, {}},
                                ScalarSpec{ACL_FLOAT, Complex(2.0, 0.0)},
                                MakeZeroTensor(ACL_FLOAT16, {0}),
                                ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("inplace_mul_empty_workspace", [&]() {
    return RunInplaceMulWorkspaceOnly("inplace_mul_empty_workspace",
                                      TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {0}, {}},
                                      TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {0}, {}},
                                      ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("inplace_muls_empty_workspace", [&]() {
    return RunInplaceMulsWorkspaceOnly("inplace_muls_empty_workspace",
                                       TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {0}, {}},
                                       ScalarSpec{ACL_FLOAT, Complex(2.0, 0.0)},
                                       ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("mul_float_basic", [&]() {
    return RunMulCase(&runtime,
                      "mul_float_basic",
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_FLOAT, {2, 3}));
  }, &total, &failed);

  RunCase("mul_float_dim5_broadcast", [&]() {
    return RunMulCase(&runtime,
                      "mul_float_dim5_broadcast",
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {1, 1, 1, 2, 3}, real_a},
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {3}, broadcast_b},
                      MakeZeroTensor(ACL_FLOAT, {1, 1, 1, 2, 3}));
  }, &total, &failed);

  RunCase("mul_float16_float", [&]() {
    return RunMulCase(&runtime,
                      "mul_float16_float",
                      TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_FLOAT, {2, 3}));
  }, &total, &failed);

  RunCase("mul_float_float16", [&]() {
    return RunMulCase(&runtime,
                      "mul_float_float16",
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_FLOAT, {2, 3}));
  }, &total, &failed);

  RunCase("mul_bf16_float", [&]() {
    return RunMulCase(&runtime,
                      "mul_bf16_float",
                      TensorSpec{ACL_BF16, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_FLOAT, {2, 3}));
  }, &total, &failed);

  RunCase("mul_float_bf16", [&]() {
    return RunMulCase(&runtime,
                      "mul_float_bf16",
                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_BF16, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_FLOAT, {2, 3}));
  }, &total, &failed);

  RunCase("mul_int32_float_promote", [&]() {
    return RunMulWorkspaceOnly("mul_int32_float_promote",
                               TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_a},
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_b},
                               MakeZeroTensor(ACL_FLOAT, {2, 3}),
                               static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("mul_float_int32_promote", [&]() {
    return RunMulWorkspaceOnly("mul_float_int32_promote",
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                               TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_b},
                               MakeZeroTensor(ACL_FLOAT, {2, 3}),
                               static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("mul_float16_float16", [&]() {
    return RunMulCase(&runtime,
                      "mul_float16_float16",
                      TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_FLOAT16, {2, 3}));
  }, &total, &failed);

  RunCase("mul_bf16_bf16", [&]() {
    return RunMulCase(&runtime,
                      "mul_bf16_bf16",
                      TensorSpec{ACL_BF16, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_BF16, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_BF16, {2, 3}));
  }, &total, &failed);

  RunCase("mul_int8_int8", [&]() {
    return RunMulCase(&runtime,
                      "mul_int8_int8",
                      TensorSpec{ACL_INT8, ACL_FORMAT_ND, {2, 3}, int_a},
                      TensorSpec{ACL_INT8, ACL_FORMAT_ND, {2, 3}, int_b},
                      MakeZeroTensor(ACL_INT8, {2, 3}));
  }, &total, &failed);

  RunCase("mul_uint8_uint8", [&]() {
    return RunMulCase(&runtime,
                      "mul_uint8_uint8",
                      TensorSpec{ACL_UINT8, ACL_FORMAT_ND, {2, 3}, uint_a},
                      TensorSpec{ACL_UINT8, ACL_FORMAT_ND, {2, 3}, uint_b},
                      MakeZeroTensor(ACL_UINT8, {2, 3}));
  }, &total, &failed);

  RunCase("mul_bool_bool", [&]() {
    return RunMulCase(&runtime,
                      "mul_bool_bool",
                      TensorSpec{ACL_BOOL, ACL_FORMAT_ND, {2, 3}, bool_a},
                      TensorSpec{ACL_BOOL, ACL_FORMAT_ND, {2, 3}, bool_b},
                      MakeZeroTensor(ACL_BOOL, {2, 3}));
  }, &total, &failed);

  RunCase("mul_int16_int16", [&]() {
    return RunMulCase(&runtime,
                      "mul_int16_int16",
                      TensorSpec{ACL_INT16, ACL_FORMAT_ND, {2, 3}, int_a},
                      TensorSpec{ACL_INT16, ACL_FORMAT_ND, {2, 3}, int_b},
                      MakeZeroTensor(ACL_INT16, {2, 3}));
  }, &total, &failed);

  RunCase("mul_int32_int32", [&]() {
    return RunMulCase(&runtime,
                      "mul_int32_int32",
                      TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_a},
                      TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_b},
                      MakeZeroTensor(ACL_INT32, {2, 3}));
  }, &total, &failed);

  RunCase("mul_int64_int64", [&]() {
    return RunMulCase(&runtime,
                      "mul_int64_int64",
                      TensorSpec{ACL_INT64, ACL_FORMAT_ND, {2, 3}, int_a},
                      TensorSpec{ACL_INT64, ACL_FORMAT_ND, {2, 3}, int_b},
                      MakeZeroTensor(ACL_INT64, {2, 3}));
  }, &total, &failed);

  RunCase("mul_double_double", [&]() {
    return RunMulCase(&runtime,
                      "mul_double_double",
                      TensorSpec{ACL_DOUBLE, ACL_FORMAT_ND, {2, 3}, real_a},
                      TensorSpec{ACL_DOUBLE, ACL_FORMAT_ND, {2, 3}, real_b},
                      MakeZeroTensor(ACL_DOUBLE, {2, 3}));
  }, &total, &failed);

  RunCase("mul_complex64_complex64", [&]() {
    return RunMulCase(&runtime,
                      "mul_complex64_complex64",
                      TensorSpec{ACL_COMPLEX64, ACL_FORMAT_ND, {2, 3}, complex_a},
                      TensorSpec{ACL_COMPLEX64, ACL_FORMAT_ND, {2, 3}, complex_b},
                      MakeZeroTensor(ACL_COMPLEX64, {2, 3}));
  }, &total, &failed);

  RunCase("mul_complex128_complex128", [&]() {
    return RunMulWorkspaceOnly("mul_complex128_complex128",
                               TensorSpec{ACL_COMPLEX128, ACL_FORMAT_ND, {2, 3}, complex_a},
                               TensorSpec{ACL_COMPLEX128, ACL_FORMAT_ND, {2, 3}, complex_b},
                               MakeZeroTensor(ACL_COMPLEX128, {2, 3}),
                               static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("mul_float_complex64_promote", [&]() {
    return RunMulWorkspaceOnly("mul_float_complex64_promote",
                               TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                               TensorSpec{ACL_COMPLEX64, ACL_FORMAT_ND, {2, 3}, complex_b},
                               MakeZeroTensor(ACL_COMPLEX64, {2, 3}),
                               static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("muls_float_basic", [&]() {
    return RunMulsCase(&runtime,
                       "muls_float_basic",
                       TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                       ScalarSpec{ACL_FLOAT, Complex(1.25, 0.0)},
                       MakeZeroTensor(ACL_FLOAT, {2, 3}));
  }, &total, &failed);

  RunCase("muls_int32_float_promote", [&]() {
    return RunMulsWorkspaceOnly("muls_int32_float_promote",
                                TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_a},
                                ScalarSpec{ACL_FLOAT, Complex(1.5, 0.0)},
                                MakeZeroTensor(ACL_FLOAT, {2, 3}),
                                static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("muls_double_complex128_bad_promote", [&]() {
    return RunMulsWorkspaceOnly(
        "muls_double_complex128_bad_promote",
        TensorSpec{ACL_DOUBLE, ACL_FORMAT_ND, {2, 3}, real_a},
        ScalarSpec{ACL_COMPLEX128, Complex(1.25, -0.5)},
        MakeZeroTensor(ACL_COMPLEX128, {2, 3}),
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("muls_float16_exact_scalar", [&]() {
    return RunMulsWorkspaceOnly(
        "muls_float16_exact_scalar",
        TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_a},
        ScalarSpec{ACL_FLOAT, Complex(2.0, 0.0)},
        MakeZeroTensor(ACL_FLOAT16, {2, 3}),
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("muls_float16_inexact_scalar", [&]() {
    return RunMulsWorkspaceOnly(
        "muls_float16_inexact_scalar",
        TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_a},
        ScalarSpec{ACL_FLOAT, Complex(1.3, 0.0)},
        MakeZeroTensor(ACL_FLOAT16, {2, 3}),
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("muls_bf16_exact_scalar", [&]() {
    return RunMulsWorkspaceOnly(
        "muls_bf16_exact_scalar",
        TensorSpec{ACL_BF16, ACL_FORMAT_ND, {2, 3}, real_a},
        ScalarSpec{ACL_FLOAT, Complex(2.0, 0.0)},
        MakeZeroTensor(ACL_BF16, {2, 3}),
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("muls_int16_double_scalar", [&]() {
    return RunMulsWorkspaceOnly(
        "muls_int16_double_scalar",
        TensorSpec{ACL_INT16, ACL_FORMAT_ND, {2, 3}, int_a},
        ScalarSpec{ACL_DOUBLE, Complex(1.5, 0.0)},
        MakeZeroTensor(ACL_FLOAT, {2, 3}),
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("muls_float16_complex_scalar", [&]() {
    return RunMulsWorkspaceOnly(
        "muls_float16_complex_scalar",
        TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_a},
        ScalarSpec{ACL_COMPLEX64, Complex(1.25, -0.5)},
        MakeZeroTensor(ACL_COMPLEX64, {2, 3}),
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("inplace_mul_float_basic", [&]() {
    return RunInplaceMulCase(&runtime,
                             "inplace_mul_float_basic",
                             TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                             TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_b});
  }, &total, &failed);

  RunCase("inplace_mul_float16_float", [&]() {
    return RunInplaceMulWorkspaceOnly(
        "inplace_mul_float16_float",
        TensorSpec{ACL_FLOAT16, ACL_FORMAT_ND, {2, 3}, real_a},
        TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_b},
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("inplace_mul_int32_int32", [&]() {
    return RunInplaceMulCase(&runtime,
                             "inplace_mul_int32_int32",
                             TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_a},
                             TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_b});
  }, &total, &failed);

  RunCase("inplace_mul_int32_float_promote", [&]() {
    return RunInplaceMulWorkspaceOnly("inplace_mul_int32_float_promote",
                                      TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_a},
                                      TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_b},
                                      static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("inplace_muls_float_basic", [&]() {
    return RunInplaceMulsExecuteExpectStatus(&runtime,
                                             "inplace_muls_float_basic",
                                             TensorSpec{ACL_FLOAT, ACL_FORMAT_ND, {2, 3}, real_a},
                                             ScalarSpec{ACL_FLOAT, Complex(1.25, 0.0)},
                                             ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("inplace_muls_bf16_exact_scalar", [&]() {
    return RunInplaceMulsWorkspaceOnly(
        "inplace_muls_bf16_exact_scalar",
        TensorSpec{ACL_BF16, ACL_FORMAT_ND, {2, 3}, real_a},
        ScalarSpec{ACL_FLOAT, Complex(2.0, 0.0)},
        static_cast<aclnnStatus>(561103));
  }, &total, &failed);

  RunCase("inplace_muls_int32_int64_scalar", [&]() {
    return RunInplaceMulsExecuteExpectStatus(&runtime,
                                             "inplace_muls_int32_int64_scalar",
                                             TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_a},
                                             ScalarSpec{ACL_INT64, Complex(3.0, 0.0)},
                                             ACLNN_SUCCESS);
  }, &total, &failed);

  RunCase("inplace_muls_int32_float_promote", [&]() {
    return RunInplaceMulsWorkspaceOnly("inplace_muls_int32_float_promote",
                                       TensorSpec{ACL_INT32, ACL_FORMAT_ND, {2, 3}, int_a},
                                       ScalarSpec{ACL_FLOAT, Complex(1.5, 0.0)},
                                       ACLNN_ERR_PARAM_INVALID);
  }, &total, &failed);

  LOG_PRINT("Summary: %d total, %d passed, %d failed\n", total, total - failed, failed);
  runtime.Finalize();
  return failed == 0 ? 0 : 1;
}
