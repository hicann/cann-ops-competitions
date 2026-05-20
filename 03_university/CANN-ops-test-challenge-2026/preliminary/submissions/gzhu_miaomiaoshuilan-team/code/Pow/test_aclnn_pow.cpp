
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

namespace {

struct TestStats {
  int total = 0;
  int passed = 0;
};

void PrintCaseResult(const std::string& name, bool ok) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t rounded = (mant + (1u << (shift - 1))) >> shift;
    return static_cast<uint16_t>(sign | rounded);
  }

  if (exp >= 31) {
    if (mant == 0) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | 0x7c00u | (mant >> 13));
  }

  uint32_t rounded_mant = mant + 0x1000u;
  if (rounded_mant & 0x800000u) {
    rounded_mant = 0;
    exp += 1;
    if (exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded_mant >> 13));
}

float HalfBitsToFloat(uint16_t bits) {
  uint32_t sign = (static_cast<uint32_t>(bits & 0x8000u)) << 16;
  uint32_t exp = (bits >> 10) & 0x1fu;
  uint32_t mant = bits & 0x03ffu;
  uint32_t out = 0;

  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp -= 1;
      }
      mant &= 0x03ffu;
      uint32_t exp32 = exp + (127 - 15);
      out = sign | (exp32 << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    uint32_t exp32 = exp + (127 - 15);
    out = sign | (exp32 << 23) | (mant << 13);
  }

  float result = 0.0f;
  std::memcpy(&result, &out, sizeof(result));
  return result;
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t bits) {
  uint32_t out = static_cast<uint32_t>(bits) << 16;
  float result = 0.0f;
  std::memcpy(&result, &out, sizeof(result));
  return result;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    size *= shape[i];
  }
  return size;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  if (shape.empty()) {
    return strides;
  }
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

struct RuntimeContext {
  int32_t device_id = 0;
  aclrtStream stream = nullptr;
  bool ready = false;

  bool Init() {
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
      std::printf("aclInit failed: %d\n", ret);
      return false;
    }
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
      std::printf("aclrtSetDevice failed: %d\n", ret);
      aclFinalize();
      return false;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
      std::printf("aclrtCreateStream failed: %d\n", ret);
      aclrtResetDevice(device_id);
      aclFinalize();
      return false;
    }
    ready = true;
    return true;
  }

  ~RuntimeContext() {
    if (ready) {
      aclrtDestroyStream(stream);
      aclrtResetDevice(device_id);
      aclFinalize();
    }
  }
};

struct TensorHolder {
  void* device = nullptr;
  aclTensor* tensor = nullptr;
  aclDataType dtype = aclDataType::ACL_FLOAT;
  std::vector<int64_t> shape;
};

void DestroyTensorHolder(TensorHolder& holder) {
  if (holder.tensor != nullptr) {
    aclDestroyTensor(holder.tensor);
    holder.tensor = nullptr;
  }
  if (holder.device != nullptr) {
    aclrtFree(holder.device);
    holder.device = nullptr;
  }
}

template <typename T>
bool CreateTensorHolder(const std::vector<T>& host_data, const std::vector<int64_t>& shape,
                        aclDataType dtype, TensorHolder* holder) {
  holder->dtype = dtype;
  holder->shape = shape;
  size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
  auto ret = aclrtMalloc(&holder->device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMalloc failed: %d\n", ret); return false);

  ret = aclrtMemcpy(holder->device, bytes, host_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMemcpy H2D failed: %d\n", ret); return false);

  std::vector<int64_t> strides = MakeContiguousStrides(shape);
  holder->tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   shape.data(), shape.size(), holder->device);
  CHECK_RET(holder->tensor != nullptr, std::printf("aclCreateTensor failed\n"); return false);
  return true;
}

template <typename T>
bool CopyTensorToHost(const TensorHolder& holder, std::vector<T>* host_data) {
  size_t bytes = static_cast<size_t>(GetShapeSize(holder.shape)) * sizeof(T);
  host_data->assign(static_cast<size_t>(GetShapeSize(holder.shape)), T());
  auto ret = aclrtMemcpy(host_data->data(), bytes, holder.device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMemcpy D2H failed: %d\n", ret); return false);
  return true;
}

bool AlmostEqual(double actual, double expected, double atol, double rtol) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  if (std::isinf(expected)) {
    return std::isinf(actual) && ((expected > 0) == (actual > 0));
  }
  return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

bool CheckFloatVector(const std::vector<float>& actual, const std::vector<double>& expected,
                      double atol, double rtol) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqual(static_cast<double>(actual[i]), expected[i], atol, rtol)) {
      std::printf("float mismatch at %zu: actual=%f expected=%f\n", i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

bool CheckHalfVector(const std::vector<uint16_t>& actual_bits, const std::vector<double>& expected,
                     double atol, double rtol) {
  if (actual_bits.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual_bits.size(); ++i) {
    float actual = HalfBitsToFloat(actual_bits[i]);
    if (!AlmostEqual(static_cast<double>(actual), expected[i], atol, rtol)) {
      std::printf("fp16 mismatch at %zu: actual=%f expected=%f\n", i, actual, expected[i]);
      return false;
    }
  }
  return true;
}

bool CheckBf16Vector(const std::vector<uint16_t>& actual_bits, const std::vector<double>& expected,
                     double atol, double rtol) {
  if (actual_bits.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual_bits.size(); ++i) {
    float actual = Bf16BitsToFloat(actual_bits[i]);
    if (!AlmostEqual(static_cast<double>(actual), expected[i], atol, rtol)) {
      std::printf("bf16 mismatch at %zu: actual=%f expected=%f\n", i, actual, expected[i]);
      return false;
    }
  }
  return true;
}

template <typename T>
bool CheckExactVector(const std::vector<T>& actual, const std::vector<T>& expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      long long act = static_cast<long long>(actual[i]);
      long long exp = static_cast<long long>(expected[i]);
      std::printf("exact mismatch at %zu: actual=%lld expected=%lld\n", i, act, exp);
      return false;
    }
  }
  return true;
}

bool RunPowTensorScalar(const TensorHolder& self, aclScalar* exponent, TensorHolder& out, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspace_size, &executor);
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnPowTensorScalarGetWorkspaceSize failed: %d\n", ret); return false);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); return false);
  }

  ret = aclnnPowTensorScalar(workspace, workspace_size, executor, stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnPowTensorScalar failed: %d\n", ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("stream sync failed: %d\n", ret); return false);
  return true;
}

bool RunInplacePowTensorScalar(TensorHolder& self, aclScalar* exponent, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self.tensor, exponent, &workspace_size, &executor);
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnInplacePowTensorScalarGetWorkspaceSize failed: %d\n", ret); return false);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); return false);
  }

  ret = aclnnInplacePowTensorScalar(workspace, workspace_size, executor, stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnInplacePowTensorScalar failed: %d\n", ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("stream sync failed: %d\n", ret); return false);
  return true;
}

bool RunPowScalarTensor(aclScalar* self, const TensorHolder& exponent, TensorHolder& out, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent.tensor, out.tensor, &workspace_size, &executor);
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnPowScalarTensorGetWorkspaceSize failed: %d\n", ret); return false);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); return false);
  }

  ret = aclnnPowScalarTensor(workspace, workspace_size, executor, stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnPowScalarTensor failed: %d\n", ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("stream sync failed: %d\n", ret); return false);
  return true;
}

bool RunPowTensorTensor(const TensorHolder& self, const TensorHolder& exponent, TensorHolder& out, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspace_size, &executor);
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnPowTensorTensorGetWorkspaceSize failed: %d\n", ret); return false);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); return false);
  }

  ret = aclnnPowTensorTensor(workspace, workspace_size, executor, stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnPowTensorTensor failed: %d\n", ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("stream sync failed: %d\n", ret); return false);
  return true;
}

bool RunInplacePowTensorTensor(TensorHolder& self, const TensorHolder& exponent, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, &workspace_size, &executor);
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnInplacePowTensorTensorGetWorkspaceSize failed: %d\n", ret); return false);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); return false);
  }

  ret = aclnnInplacePowTensorTensor(workspace, workspace_size, executor, stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnInplacePowTensorTensor failed: %d\n", ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("stream sync failed: %d\n", ret); return false);
  return true;
}

bool RunExp2(const TensorHolder& self, TensorHolder& out, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspace_size, &executor);
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnExp2GetWorkspaceSize failed: %d\n", ret); return false);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); return false);
  }

  ret = aclnnExp2(workspace, workspace_size, executor, stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnExp2 failed: %d\n", ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("stream sync failed: %d\n", ret); return false);
  return true;
}

bool RunInplaceExp2(TensorHolder& self, aclrtStream stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceExp2GetWorkspaceSize(self.tensor, &workspace_size, &executor);
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnInplaceExp2GetWorkspaceSize failed: %d\n", ret); return false);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); return false);
  }

  ret = aclnnInplaceExp2(workspace, workspace_size, executor, stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  CHECK_RET(ret == ACLNN_SUCCESS, std::printf("aclnnInplaceExp2 failed: %d\n", ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, std::printf("stream sync failed: %d\n", ret); return false);
  return true;
}

bool TestPowTensorScalarFloatSquare(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> self_data = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -2.0f};
  std::vector<float> out_init(6, 0.0f);
  float exponent_value = 2.0f;

  TensorHolder self, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_FLOAT, &self) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&exponent_value, aclDataType::ACL_FLOAT);

  if (ok) {
    ok = exponent != nullptr && RunPowTensorScalar(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < self_data.size(); ++i) {
      expected.push_back(std::pow(static_cast<double>(self_data[i]), 2.0));
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  DestroyTensorHolder(self);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorScalarFloatGeneral(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<float> self_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out_init(4, 0.0f);
  float exponent_value = 3.0f;

  TensorHolder self, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_FLOAT, &self) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&exponent_value, aclDataType::ACL_FLOAT);

  if (ok) {
    ok = exponent != nullptr && RunPowTensorScalar(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < self_data.size(); ++i) {
      expected.push_back(std::pow(static_cast<double>(self_data[i]), 3.0));
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  DestroyTensorHolder(self);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorScalarIntNegativeExponentError() {
  std::vector<int64_t> shape = {4};
  std::vector<int32_t> self_data = {1, 2, 3, 4};
  std::vector<float> out_init(4, 0.0f);
  int32_t exponent_value = -1;

  TensorHolder self, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_INT32, &self) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&exponent_value, aclDataType::ACL_INT32);

  if (ok) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspace_size, &executor);
    ok = (ret != ACLNN_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  DestroyTensorHolder(self);
  DestroyTensorHolder(out);
  return ok;
}

bool TestInplacePowTensorScalarFloat(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<float> self_data = {2.0f, 3.0f, 4.0f, 0.5f};
  float exponent_value = 2.0f;

  TensorHolder self;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_FLOAT, &self);
  aclScalar* exponent = aclCreateScalar(&exponent_value, aclDataType::ACL_FLOAT);

  if (ok) {
    ok = exponent != nullptr && RunInplacePowTensorScalar(self, exponent, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(self, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < self_data.size(); ++i) {
      expected.push_back(std::pow(static_cast<double>(self_data[i]), 2.0));
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  DestroyTensorHolder(self);
  return ok;
}

bool TestPowScalarTensorFillOne(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {5};
  std::vector<float> exponent_data = {0.0f, 1.0f, 2.0f, 3.0f, 0.5f};
  std::vector<float> out_init(5, 0.0f);
  float self_value = 1.0f;

  TensorHolder exponent, out;
  bool ok = CreateTensorHolder(exponent_data, shape, aclDataType::ACL_FLOAT, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);
  aclScalar* self = aclCreateScalar(&self_value, aclDataType::ACL_FLOAT);

  if (ok) {
    ok = self != nullptr && RunPowScalarTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected(actual.size(), 1.0);
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  if (self != nullptr) {
    aclDestroyScalar(self);
  }
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowScalarTensorCompute(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<float> exponent_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> out_init(4, 0.0f);
  float self_value = 2.0f;

  TensorHolder exponent, out;
  bool ok = CreateTensorHolder(exponent_data, shape, aclDataType::ACL_FLOAT, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);
  aclScalar* self = aclCreateScalar(&self_value, aclDataType::ACL_FLOAT);

  if (ok) {
    ok = self != nullptr && RunPowScalarTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < exponent_data.size(); ++i) {
      expected.push_back(std::pow(2.0, static_cast<double>(exponent_data[i])));
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  if (self != nullptr) {
    aclDestroyScalar(self);
  }
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorTensorFloatBroadcast(RuntimeContext& ctx) {
  std::vector<int64_t> self_shape = {2, 3};
  std::vector<int64_t> exp_shape = {3};
  std::vector<int64_t> out_shape = {2, 3};
  std::vector<float> self_data = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 1.5f};
  std::vector<float> exp_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> out_init(6, 0.0f);

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, self_shape, aclDataType::ACL_FLOAT, &self) &&
            CreateTensorHolder(exp_data, exp_shape, aclDataType::ACL_FLOAT, &exponent) &&
            CreateTensorHolder(out_init, out_shape, aclDataType::ACL_FLOAT, &out);

  if (ok) {
    ok = RunPowTensorTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected;
    for (int64_t row = 0; row < 2; ++row) {
      for (int64_t col = 0; col < 3; ++col) {
        double base = static_cast<double>(self_data[static_cast<size_t>(row * 3 + col)]);
        double expv = static_cast<double>(exp_data[static_cast<size_t>(col)]);
        expected.push_back(std::pow(base, expv));
      }
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestInplacePowTensorTensorFloat(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<float> self_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> exp_data = {2.0f, 2.0f, 2.0f, 2.0f};

  TensorHolder self, exponent;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_FLOAT, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_FLOAT, &exponent);

  if (ok) {
    ok = RunInplacePowTensorTensor(self, exponent, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(self, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < self_data.size(); ++i) {
      expected.push_back(std::pow(static_cast<double>(self_data[i]), 2.0));
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  return ok;
}

bool TestPowTensorTensorFloat16(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<uint16_t> self_data = {
      FloatToHalfBits(1.0f), FloatToHalfBits(2.0f), FloatToHalfBits(3.0f), FloatToHalfBits(1.5f)};
  std::vector<uint16_t> exp_data = {
      FloatToHalfBits(0.0f), FloatToHalfBits(1.0f), FloatToHalfBits(2.0f), FloatToHalfBits(3.0f)};
  std::vector<uint16_t> out_init(4, FloatToHalfBits(0.0f));

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_FLOAT16, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_FLOAT16, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT16, &out);

  if (ok) {
    ok = RunPowTensorTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<uint16_t> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected = {1.0, 2.0, 9.0, 3.375};
    ok = ok && CheckHalfVector(actual, expected, 1e-3, 1e-3);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorTensorBf16(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<uint16_t> self_data = {
      FloatToBf16Bits(1.0f), FloatToBf16Bits(2.0f), FloatToBf16Bits(3.0f), FloatToBf16Bits(1.5f)};
  std::vector<uint16_t> exp_data = {
      FloatToBf16Bits(0.0f), FloatToBf16Bits(1.0f), FloatToBf16Bits(2.0f), FloatToBf16Bits(3.0f)};
  std::vector<uint16_t> out_init(4, FloatToBf16Bits(0.0f));

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_BF16, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_BF16, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_BF16, &out);

  if (ok) {
    ok = RunPowTensorTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<uint16_t> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected = {1.0, 2.0, 9.0, 3.375};
    ok = ok && CheckBf16Vector(actual, expected, 1e-2, 1e-2);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorTensorUint8(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<uint8_t> self_data = {1, 2, 3, 2};
  std::vector<uint8_t> exp_data = {0, 1, 2, 3};
  std::vector<uint8_t> out_init(4, 0);

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_UINT8, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_UINT8, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_UINT8, &out);

  if (ok) {
    ok = RunPowTensorTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<uint8_t> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<uint8_t> expected = {1, 2, 9, 8};
    ok = ok && CheckExactVector(actual, expected);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorTensorInt8(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<int8_t> self_data = {1, 2, 3, 2};
  std::vector<int8_t> exp_data = {0, 1, 2, 3};
  std::vector<int8_t> out_init(4, 0);

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_INT8, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_INT8, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_INT8, &out);

  if (ok) {
    ok = RunPowTensorTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<int8_t> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<int8_t> expected = {1, 2, 9, 8};
    ok = ok && CheckExactVector(actual, expected);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorTensorInt16(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<int16_t> self_data = {1, 2, 3, 2};
  std::vector<int16_t> exp_data = {0, 1, 2, 3};
  std::vector<int16_t> out_init(4, 0);

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_INT16, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_INT16, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_INT16, &out);

  if (ok) {
    ok = RunPowTensorTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<int16_t> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<int16_t> expected = {1, 2, 9, 8};
    ok = ok && CheckExactVector(actual, expected);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorTensorInt32(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<int32_t> self_data = {1, 2, 3, 2};
  std::vector<int32_t> exp_data = {0, 1, 2, 3};
  std::vector<int32_t> out_init(4, 0);

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_INT32, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_INT32, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_INT32, &out);

  if (ok) {
    ok = RunPowTensorTensor(self, exponent, out, ctx.stream);
  }

  if (ok) {
    std::vector<int32_t> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<int32_t> expected = {1, 2, 9, 8};
    ok = ok && CheckExactVector(actual, expected);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestPowTensorTensorBoolBoolError() {
  std::vector<int64_t> shape = {4};
  std::vector<bool> dummy; // unused, avoid vector<bool>
  std::vector<uint8_t> self_data = {1, 0, 1, 0};
  std::vector<uint8_t> exp_data = {0, 1, 0, 1};
  std::vector<float> out_init(4, 0.0f);

  TensorHolder self, exponent, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_BOOL, &self) &&
            CreateTensorHolder(exp_data, shape, aclDataType::ACL_BOOL, &exponent) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);

  if (ok) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspace_size, &executor);
    ok = (ret != ACLNN_SUCCESS);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(exponent);
  DestroyTensorHolder(out);
  return ok;
}

bool TestExp2Float(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<float> self_data = {-1.0f, 0.0f, 1.0f, 2.0f};
  std::vector<float> out_init(4, 0.0f);

  TensorHolder self, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_FLOAT, &self) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);

  if (ok) {
    ok = RunExp2(self, out, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < self_data.size(); ++i) {
      expected.push_back(std::pow(2.0, static_cast<double>(self_data[i])));
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(out);
  return ok;
}

bool TestInplaceExp2Float(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<float> self_data = {-1.0f, 0.0f, 1.0f, 2.0f};

  TensorHolder self;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_FLOAT, &self);

  if (ok) {
    ok = RunInplaceExp2(self, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(self, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < self_data.size(); ++i) {
      expected.push_back(std::pow(2.0, static_cast<double>(self_data[i])));
    }
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  DestroyTensorHolder(self);
  return ok;
}

bool TestExp2Int32ToFloat(RuntimeContext& ctx) {
  std::vector<int64_t> shape = {4};
  std::vector<int32_t> self_data = {0, 1, 2, 3};
  std::vector<float> out_init(4, 0.0f);

  TensorHolder self, out;
  bool ok = CreateTensorHolder(self_data, shape, aclDataType::ACL_INT32, &self) &&
            CreateTensorHolder(out_init, shape, aclDataType::ACL_FLOAT, &out);

  if (ok) {
    ok = RunExp2(self, out, ctx.stream);
  }

  if (ok) {
    std::vector<float> actual;
    ok = CopyTensorToHost(out, &actual);
    std::vector<double> expected = {1.0, 2.0, 4.0, 8.0};
    ok = ok && CheckFloatVector(actual, expected, 1e-5, 1e-5);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(out);
  return ok;
}

bool TestExp2ShapeError() {
  std::vector<int64_t> self_shape = {4};
  std::vector<int64_t> out_shape = {2, 2};
  std::vector<float> self_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> out_init(4, 0.0f);

  TensorHolder self, out;
  bool ok = CreateTensorHolder(self_data, self_shape, aclDataType::ACL_FLOAT, &self) &&
            CreateTensorHolder(out_init, out_shape, aclDataType::ACL_FLOAT, &out);

  if (ok) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspace_size, &executor);
    ok = (ret != ACLNN_SUCCESS);
  }

  DestroyTensorHolder(self);
  DestroyTensorHolder(out);
  return ok;
}

struct TestCase {
  const char* name;
  bool (*func)(RuntimeContext&);
};

}  // namespace

int main() {
  RuntimeContext ctx;
  if (!ctx.Init()) {
    return 1;
  }

  TestStats stats;

  const TestCase tests[] = {
      {"PowTensorScalar_Float_Square", TestPowTensorScalarFloatSquare},
      {"PowTensorScalar_Float_General", TestPowTensorScalarFloatGeneral},
      {"InplacePowTensorScalar_Float", TestInplacePowTensorScalarFloat},
      {"PowScalarTensor_FillOne", TestPowScalarTensorFillOne},
      {"PowScalarTensor_Compute", TestPowScalarTensorCompute},
      {"PowTensorTensor_Float_Broadcast", TestPowTensorTensorFloatBroadcast},
      {"InplacePowTensorTensor_Float", TestInplacePowTensorTensorFloat},
      {"PowTensorTensor_Float16", TestPowTensorTensorFloat16},
      {"PowTensorTensor_Bf16", TestPowTensorTensorBf16},
      {"PowTensorTensor_Uint8", TestPowTensorTensorUint8},
      {"PowTensorTensor_Int8", TestPowTensorTensorInt8},
      {"PowTensorTensor_Int16", TestPowTensorTensorInt16},
      {"PowTensorTensor_Int32", TestPowTensorTensorInt32},
      {"Exp2_Float", TestExp2Float},
      {"InplaceExp2_Float", TestInplaceExp2Float},
      {"Exp2_Int32_ToFloat", TestExp2Int32ToFloat},
  };

  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    ++stats.total;
    bool ok = tests[i].func(ctx);
    if (ok) {
      ++stats.passed;
    }
    PrintCaseResult(tests[i].name, ok);
  }

  struct ErrorCase {
    const char* name;
    bool (*func)();
  };

  const ErrorCase error_tests[] = {
      {"PowTensorScalar_Int_NegativeExponent_Error", TestPowTensorScalarIntNegativeExponentError},
      {"PowTensorTensor_BoolBool_Error", TestPowTensorTensorBoolBoolError},
      {"Exp2_Shape_Error", TestExp2ShapeError},
  };

  for (size_t i = 0; i < sizeof(error_tests) / sizeof(error_tests[0]); ++i) {
    ++stats.total;
    bool ok = error_tests[i].func();
    if (ok) {
      ++stats.passed;
    }
    PrintCaseResult(error_tests[i].name, ok);
  }

  std::printf("\n=== Pow Test Summary: total=%d passed=%d failed=%d ===\n",
              stats.total, stats.passed, stats.total - stats.passed);
  return (stats.total == stats.passed) ? 0 : 1;
}
