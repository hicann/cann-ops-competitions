#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

#define CHECK_RET(cond, expr) \
  do { \
    if (!(cond)) { \
      expr; \
    } \
  } while (0)

#define LOG_PRINT(fmt, ...) \
  do { \
    std::printf(fmt, ##__VA_ARGS__); \
  } while (0)

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    n *= shape[i];
  }
  return n;
}

uint32_t FloatAsUint(float x) {
  uint32_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

float UintAsFloat(uint32_t u) {
  float x = 0;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

uint16_t FloatToFp16Bits(float x) {
  uint32_t f = FloatAsUint(x);
  uint32_t sign = (f >> 16) & 0x8000u;
  uint32_t mantissa = f & 0x007fffffu;
  int32_t exp = static_cast<int32_t>((f >> 23) & 0xffu) - 127 + 15;

  if (((f >> 23) & 0xffu) == 0xffu) {
    if (mantissa == 0) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | 0x7e00u);
  }

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x00800000u;
    uint32_t shift = static_cast<uint32_t>(1 - exp);
    uint32_t half_mant = mantissa >> (shift + 13);
    uint32_t round_bit = (mantissa >> (shift + 12)) & 1u;
    if (round_bit) {
      half_mant += 1u;
    }
    return static_cast<uint16_t>(sign | half_mant);
  }

  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mantissa >> 13));
  if (mantissa & 0x00001000u) {
    half = static_cast<uint16_t>(half + 1);
  }
  return half;
}

float Fp16BitsToFloat(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x03ffu;

  if (exp == 0) {
    if (mantissa == 0) {
      return UintAsFloat(sign);
    }
    while ((mantissa & 0x0400u) == 0) {
      mantissa <<= 1;
      exp -= 1;
    }
    mantissa &= 0x03ffu;
    exp = 1;
    uint32_t fexp = exp + (127 - 15);
    uint32_t bits = sign | (fexp << 23) | (mantissa << 13);
    return UintAsFloat(bits);
  }

  if (exp == 31) {
    uint32_t bits = sign | 0x7f800000u | (mantissa << 13);
    return UintAsFloat(bits);
  }

  uint32_t fexp = exp + (127 - 15);
  uint32_t bits = sign | (fexp << 23) | (mantissa << 13);
  return UintAsFloat(bits);
}

struct DeviceTensor {
  void* deviceAddr = nullptr;
  aclTensor* tensor = nullptr;
  size_t bytes = 0;

  void Destroy() {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
      tensor = nullptr;
    }
    if (deviceAddr != nullptr) {
      aclrtFree(deviceAddr);
      deviceAddr = nullptr;
    }
    bytes = 0;
  }
};

struct DeviceScalar {
  aclScalar* scalar = nullptr;

  void Destroy() {
    if (scalar != nullptr) {
      aclDestroyScalar(scalar);
      scalar = nullptr;
    }
  }
};

struct Runtime {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  bool ready = false;

  int Init() {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ready = true;
    return 0;
  }

  void Destroy() {
    if (stream != nullptr) {
      aclrtDestroyStream(stream);
      stream = nullptr;
    }
    if (ready) {
      aclrtResetDevice(deviceId);
      aclFinalize();
      ready = false;
    }
  }
};

template <typename T>
bool CreateAclTensorFromVector(const std::vector<T>& hostData,
                               const std::vector<int64_t>& shape,
                               aclDataType dataType,
                               DeviceTensor* out) {
  out->Destroy();
  size_t size = hostData.size() * sizeof(T);
  out->bytes = size;

  if (size > 0) {
    auto ret = aclrtMalloc(&out->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
      return false;
    }
    ret = aclrtMemcpy(out->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret);
      out->Destroy();
      return false;
    }
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }

  out->tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                                strides.empty() ? nullptr : strides.data(),
                                0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), out->deviceAddr);
  if (out->tensor == nullptr) {
    LOG_PRINT("aclCreateTensor failed.\n");
    out->Destroy();
    return false;
  }
  return true;
}

template <typename T>
bool CopyTensorToHost(const DeviceTensor& tensor, std::vector<T>* hostData) {
  if (hostData->empty()) {
    return true;
  }
  auto ret = aclrtMemcpy(hostData->data(), hostData->size() * sizeof(T),
                         tensor.deviceAddr, hostData->size() * sizeof(T),
                         ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret);
    return false;
  }
  return true;
}

template <typename T>
bool CreateScalar(const T& value, aclDataType dataType, DeviceScalar* out) {
  out->Destroy();
  out->scalar = aclCreateScalar(const_cast<T*>(&value), dataType);
  if (out->scalar == nullptr) {
    LOG_PRINT("aclCreateScalar failed.\n");
    return false;
  }
  return true;
}

bool RunTwoPhase(const std::function<aclnnStatus(uint64_t*, aclOpExecutor**)>& phase1,
                 const std::function<aclnnStatus(void*, uint64_t, aclOpExecutor*)>& phase2,
                 aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = phase1(&workspaceSize, &executor);
  if (ret != ACLNN_SUCCESS) {
    LOG_PRINT("phase1 failed. status=%d\n", static_cast<int>(ret));
    return false;
  }

  void* workspace = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return false;
    }
  }

  auto runRet = phase2(workspace, workspaceSize, executor);
  if (runRet != ACLNN_SUCCESS) {
    LOG_PRINT("phase2 failed. status=%d\n", static_cast<int>(runRet));
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    return false;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    return false;
  }

  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  return true;
}

bool AlmostEqualFloat(float a, float b, float atol, float rtol) {
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::isinf(a) || std::isinf(b)) {
    return a == b;
  }
  return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

bool AlmostEqualComplex(const std::complex<float>& a,
                        const std::complex<float>& b,
                        float atol,
                        float rtol) {
  return AlmostEqualFloat(a.real(), b.real(), atol, rtol) &&
         AlmostEqualFloat(a.imag(), b.imag(), atol, rtol);
}

bool CheckFloatVector(const std::vector<float>& actual,
                      const std::vector<float>& expected,
                      float atol,
                      float rtol) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqualFloat(actual[i], expected[i], atol, rtol)) {
      LOG_PRINT("float mismatch at %zu: actual=%f expected=%f\n", i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

bool CheckFp16Vector(const std::vector<uint16_t>& actual,
                     const std::vector<float>& expected,
                     float atol,
                     float rtol) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    float a = Fp16BitsToFloat(actual[i]);
    if (!AlmostEqualFloat(a, expected[i], atol, rtol)) {
      LOG_PRINT("fp16 mismatch at %zu: actual=%f expected=%f\n", i, a, expected[i]);
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
      LOG_PRINT("exact mismatch at %zu: actual=%lld expected=%lld\n",
                i,
                static_cast<long long>(actual[i]),
                static_cast<long long>(expected[i]));
      return false;
    }
  }
  return true;
}

bool CheckComplexVector(const std::vector<std::complex<float>>& actual,
                        const std::vector<std::complex<float>>& expected,
                        float atol,
                        float rtol) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqualComplex(actual[i], expected[i], atol, rtol)) {
      LOG_PRINT("complex mismatch at %zu: actual=(%f,%f) expected=(%f,%f)\n",
                i, actual[i].real(), actual[i].imag(), expected[i].real(), expected[i].imag());
      return false;
    }
  }
  return true;
}

std::vector<float> ExpandLastDimFloat(const std::vector<float>& src,
                                      const std::vector<int64_t>& srcShape,
                                      const std::vector<int64_t>& dstShape) {
  std::vector<float> dst(static_cast<size_t>(GetShapeSize(dstShape)), 0.0f);
  if (srcShape.size() == 1 && dstShape.size() == 2 && srcShape[0] == dstShape[1]) {
    int64_t rows = dstShape[0];
    int64_t cols = dstShape[1];
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        dst[static_cast<size_t>(r * cols + c)] = src[static_cast<size_t>(c)];
      }
    }
  }
  return dst;
}

struct Tester {
  Runtime* runtime;
  int passed = 0;
  int failed = 0;

  explicit Tester(Runtime* r) : runtime(r) {}

  void Run(const std::string& name, const std::function<bool()>& fn) {
    bool ok = fn();
    if (ok) {
      ++passed;
      LOG_PRINT("[PASS] %s\n", name.c_str());
    } else {
      ++failed;
      LOG_PRINT("[FAIL] %s\n", name.c_str());
    }
  }
};

bool TestAddFloatSameShapeAlpha1(Runtime* runtime) {
  std::vector<int64_t> shape = {4, 2};
  std::vector<float> self = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> other = {1, -1, 1, 2, -2, 2, 3, 3};
  std::vector<float> expected(self.size(), 0.0f);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + other[i];
  }

  DeviceTensor selfT, otherT, outT;
  DeviceScalar alpha;
  float alphaValue = 1.0f;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_FLOAT, &selfT) &&
            CreateAclTensorFromVector(other, shape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(self.size(), 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddGetWorkspaceSize(selfT.tensor, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdd(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(self.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddFloatAxpyAlphaNonOne(Runtime* runtime) {
  std::vector<int64_t> shape = {4, 2};
  std::vector<float> self = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> other = {1, 1, -1, 2, -2, 2, 3, -3};
  float alphaValue = 1.5f;
  std::vector<float> expected(self.size(), 0.0f);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + alphaValue * other[i];
  }

  DeviceTensor selfT, otherT, outT;
  DeviceScalar alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_FLOAT, &selfT) &&
            CreateAclTensorFromVector(other, shape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(self.size(), 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddGetWorkspaceSize(selfT.tensor, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdd(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(self.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddInt32AxpyV2(Runtime* runtime) {
  std::vector<int64_t> shape = {4, 2};
  std::vector<int32_t> self = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> other = {1, -1, 1, 2, -2, 2, 3, 3};
  int32_t alphaValue = 2;
  std::vector<int32_t> expected(self.size(), 0);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + alphaValue * other[i];
  }

  DeviceTensor selfT, otherT, outT;
  DeviceScalar alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_INT32, &selfT) &&
            CreateAclTensorFromVector(other, shape, ACL_INT32, &otherT) &&
            CreateScalar(alphaValue, ACL_INT32, &alpha) &&
            CreateAclTensorFromVector(std::vector<int32_t>(self.size(), 0), shape, ACL_INT32, &outT);
  if (!ok) {
    selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddGetWorkspaceSize(selfT.tensor, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdd(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<int32_t> actual(self.size(), 0);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckExactVector(actual, expected);
  }

  selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddComplex64FallbackMulThenAdd(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<std::complex<float>> self = {
      {1.0f, 2.0f}, {3.0f, -1.0f}, {-2.0f, 0.5f}, {0.0f, -3.0f}};
  std::vector<std::complex<float>> other = {
      {0.5f, -1.0f}, {2.0f, 0.0f}, {-1.0f, 1.0f}, {4.0f, 2.0f}};
  float alphaValue = 2.0f;
  std::vector<std::complex<float>> expected(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + alphaValue * other[i];
  }

  DeviceTensor selfT, otherT, outT;
  DeviceScalar alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_COMPLEX64, &selfT) &&
            CreateAclTensorFromVector(other, shape, ACL_COMPLEX64, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<std::complex<float>>(self.size(), {0.0f, 0.0f}), shape, ACL_COMPLEX64, &outT);
  if (!ok) {
    selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddGetWorkspaceSize(selfT.tensor, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdd(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<std::complex<float>> actual(self.size(), {0.0f, 0.0f});
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckComplexVector(actual, expected, 1e-4f, 1e-4f);
  }

  selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddMixDtypeFloat16Float(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfFp32 = {0.5f, 1.0f, -2.0f, 3.5f, 4.0f, -5.5f, 6.25f, 7.0f};
  std::vector<uint16_t> self;
  for (size_t i = 0; i < selfFp32.size(); ++i) {
    self.push_back(FloatToFp16Bits(selfFp32[i]));
  }
  std::vector<float> other = {1.0f, -1.0f, 2.0f, 0.5f, -4.0f, 5.5f, -6.25f, 3.0f};
  float alphaValue = 1.0f;
  std::vector<float> expected(other.size(), 0.0f);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = Fp16BitsToFloat(self[i]) + other[i];
  }

  DeviceTensor selfT, otherT, outT;
  DeviceScalar alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_FLOAT16, &selfT) &&
            CreateAclTensorFromVector(other, shape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(other.size(), 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddGetWorkspaceSize(selfT.tensor, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdd(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(expected.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckFloatVector(actual, expected, 2e-3f, 2e-3f);
  }

  selfT.Destroy(); otherT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddsFloatScalarAlpha1(Runtime* runtime) {
  std::vector<int64_t> shape = {4, 2};
  std::vector<float> self = {0, 1, 2, 3, 4, 5, 6, 7};
  float otherValue = 2.5f;
  float alphaValue = 1.0f;
  std::vector<float> expected(self.size(), 0.0f);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + otherValue;
  }

  DeviceTensor selfT, outT;
  DeviceScalar other, alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_FLOAT, &selfT) &&
            CreateScalar(otherValue, ACL_FLOAT, &other) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(self.size(), 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddsGetWorkspaceSize(selfT.tensor, other.scalar, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdds(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(self.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddsFloatAxpy(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> self = {0, 1, 2, 3, 4, 5, 6, 7};
  float otherValue = -1.25f;
  float alphaValue = 2.0f;
  std::vector<float> expected(self.size(), 0.0f);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + alphaValue * otherValue;
  }

  DeviceTensor selfT, outT;
  DeviceScalar other, alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_FLOAT, &selfT) &&
            CreateScalar(otherValue, ACL_FLOAT, &other) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(self.size(), 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddsGetWorkspaceSize(selfT.tensor, other.scalar, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdds(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(self.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddsInt32AxpyV2(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  std::vector<int32_t> self = {0, 1, 2, 3, 4, 5, 6, 7};
  int32_t otherValue = -3;
  int32_t alphaValue = 2;
  std::vector<int32_t> expected(self.size(), 0);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + alphaValue * otherValue;
  }

  DeviceTensor selfT, outT;
  DeviceScalar other, alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_INT32, &selfT) &&
            CreateScalar(otherValue, ACL_INT32, &other) &&
            CreateScalar(alphaValue, ACL_INT32, &alpha) &&
            CreateAclTensorFromVector(std::vector<int32_t>(self.size(), 0), shape, ACL_INT32, &outT);
  if (!ok) {
    selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddsGetWorkspaceSize(selfT.tensor, other.scalar, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdds(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<int32_t> actual(self.size(), 0);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckExactVector(actual, expected);
  }

  selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddsBoolSpecialCast(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  std::vector<uint8_t> self = {0, 1, 0, 1, 1, 0, 0, 1};
  uint8_t otherValue = 1;
  uint8_t alphaValue = 1;
  std::vector<int32_t> expected = {1, 1, 1, 1, 1, 1, 1, 1};

  DeviceTensor selfT, outT;
  DeviceScalar other, alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_BOOL, &selfT) &&
            CreateScalar(otherValue, ACL_BOOL, &other) &&
            CreateScalar(alphaValue, ACL_BOOL, &alpha) &&
            CreateAclTensorFromVector(std::vector<int32_t>(expected.size(), 0), shape, ACL_INT32, &outT);
  if (!ok) {
    selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddsGetWorkspaceSize(selfT.tensor, other.scalar, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAdds(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<int32_t> actual(expected.size(), 0);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckExactVector(actual, expected);
  }

  selfT.Destroy(); outT.Destroy(); other.Destroy(); alpha.Destroy();
  return ok;
}

bool TestInplaceAddBroadcast(Runtime* runtime) {
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<float> self = {1, 2, 3, 4, 5, 6};
  std::vector<float> other = {10, -1, 2};
  float alphaValue = 0.5f;
  std::vector<float> otherExpanded = ExpandLastDimFloat(other, otherShape, selfShape);
  std::vector<float> expected(self.size(), 0.0f);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + alphaValue * otherExpanded[i];
  }

  DeviceTensor selfT, otherT;
  DeviceScalar alpha;
  bool ok = CreateAclTensorFromVector(self, selfShape, ACL_FLOAT, &selfT) &&
            CreateAclTensorFromVector(other, otherShape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha);
  if (!ok) {
    selfT.Destroy(); otherT.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnInplaceAddGetWorkspaceSize(selfT.tensor, otherT.tensor, alpha.scalar, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnInplaceAdd(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(self.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(selfT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  selfT.Destroy(); otherT.Destroy(); alpha.Destroy();
  return ok;
}

bool TestInplaceAddsFloat(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> self = {0, 1, 2, 3, 4, 5, 6, 7};
  float otherValue = 1.25f;
  float alphaValue = -2.0f;
  std::vector<float> expected(self.size(), 0.0f);
  for (size_t i = 0; i < self.size(); ++i) {
    expected[i] = self[i] + alphaValue * otherValue;
  }

  DeviceTensor selfT;
  DeviceScalar other, alpha;
  bool ok = CreateAclTensorFromVector(self, shape, ACL_FLOAT, &selfT) &&
            CreateScalar(otherValue, ACL_FLOAT, &other) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha);
  if (!ok) {
    selfT.Destroy(); other.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnInplaceAddsGetWorkspaceSize(selfT.tensor, other.scalar, alpha.scalar, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnInplaceAdds(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(self.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(selfT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  selfT.Destroy(); other.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddV3Alpha1(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  float selfScalar = 3.0f;
  std::vector<float> other = {0, 1, 2, 3, 4, 5, 6, 7};
  float alphaValue = 1.0f;
  std::vector<float> expected(other.size(), 0.0f);
  for (size_t i = 0; i < other.size(); ++i) {
    expected[i] = selfScalar + other[i];
  }

  DeviceTensor otherT, outT;
  DeviceScalar selfS, alpha;
  bool ok = CreateScalar(selfScalar, ACL_FLOAT, &selfS) &&
            CreateAclTensorFromVector(other, shape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(other.size(), 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    otherT.Destroy(); outT.Destroy(); selfS.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddV3GetWorkspaceSize(selfS.scalar, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAddV3(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(other.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  otherT.Destroy(); outT.Destroy(); selfS.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddV3Axpy(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  float selfScalar = 1.25f;
  std::vector<float> other = {0, 1, -2, 3, 4, -5, 6, 7};
  float alphaValue = 2.0f;
  std::vector<float> expected(other.size(), 0.0f);
  for (size_t i = 0; i < other.size(); ++i) {
    expected[i] = selfScalar + alphaValue * other[i];
  }

  DeviceTensor otherT, outT;
  DeviceScalar selfS, alpha;
  bool ok = CreateScalar(selfScalar, ACL_FLOAT, &selfS) &&
            CreateAclTensorFromVector(other, shape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(other.size(), 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    otherT.Destroy(); outT.Destroy(); selfS.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddV3GetWorkspaceSize(selfS.scalar, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAddV3(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(other.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  otherT.Destroy(); outT.Destroy(); selfS.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddV3Int8FallbackMulThenAdd(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  int8_t selfScalar = 3;
  std::vector<int8_t> other = {1, -1, 2, 0, -3, 4, 5, -2};
  int8_t alphaValue = 2;
  std::vector<int8_t> expected(other.size(), 0);
  for (size_t i = 0; i < other.size(); ++i) {
    expected[i] = static_cast<int8_t>(selfScalar + alphaValue * other[i]);
  }

  DeviceTensor otherT, outT;
  DeviceScalar selfS, alpha;
  bool ok = CreateScalar(selfScalar, ACL_INT8, &selfS) &&
            CreateAclTensorFromVector(other, shape, ACL_INT8, &otherT) &&
            CreateScalar(alphaValue, ACL_INT8, &alpha) &&
            CreateAclTensorFromVector(std::vector<int8_t>(other.size(), 0), shape, ACL_INT8, &outT);
  if (!ok) {
    otherT.Destroy(); outT.Destroy(); selfS.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnAddV3GetWorkspaceSize(selfS.scalar, otherT.tensor, alpha.scalar, outT.tensor, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnAddV3(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<int8_t> actual(other.size(), 0);
  if (ok) {
    ok = CopyTensorToHost(outT, &actual) && CheckExactVector(actual, expected);
  }

  otherT.Destroy(); outT.Destroy(); selfS.Destroy(); alpha.Destroy();
  return ok;
}

bool TestInplaceAddV3(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 4};
  float selfScalar = 2.0f;
  std::vector<float> other = {0, 1, 2, 3, 4, 5, 6, 7};
  float alphaValue = 0.5f;
  std::vector<float> expected(other.size(), 0.0f);
  for (size_t i = 0; i < other.size(); ++i) {
    expected[i] = selfScalar + alphaValue * other[i];
  }

  DeviceTensor otherT;
  DeviceScalar selfS, alpha;
  bool ok = CreateScalar(selfScalar, ACL_FLOAT, &selfS) &&
            CreateAclTensorFromVector(other, shape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha);
  if (!ok) {
    otherT.Destroy(); selfS.Destroy(); alpha.Destroy();
    return false;
  }

  ok = RunTwoPhase(
      [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnInplaceAddV3GetWorkspaceSize(selfS.scalar, otherT.tensor, alpha.scalar, workspaceSize, executor);
      },
      [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor) {
        return aclnnInplaceAddV3(workspace, workspaceSize, executor, runtime->stream);
      },
      runtime->stream);

  std::vector<float> actual(other.size(), 0.0f);
  if (ok) {
    ok = CopyTensorToHost(otherT, &actual) && CheckFloatVector(actual, expected, 1e-5f, 1e-5f);
  }

  otherT.Destroy(); selfS.Destroy(); alpha.Destroy();
  return ok;
}

bool TestAddNullptrError() {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  float alphaValue = 1.0f;
  DeviceScalar alpha;
  if (!CreateScalar(alphaValue, ACL_FLOAT, &alpha)) {
    return false;
  }
  aclnnStatus ret = aclnnAddGetWorkspaceSize(nullptr, nullptr, alpha.scalar, nullptr, &workspaceSize, &executor);
  alpha.Destroy();
  return ret != ACLNN_SUCCESS;
}

bool TestAddBadOutShapeError(Runtime* runtime) {
  std::vector<int64_t> inShape = {2, 3};
  std::vector<int64_t> badOutShape = {3, 2};
  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> b = {6, 5, 4, 3, 2, 1};
  DeviceTensor aT, bT, outT;
  DeviceScalar alpha;
  float alphaValue = 1.0f;
  bool ok = CreateAclTensorFromVector(a, inShape, ACL_FLOAT, &aT) &&
            CreateAclTensorFromVector(b, inShape, ACL_FLOAT, &bT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(6, 0.0f), badOutShape, ACL_FLOAT, &outT);
  if (!ok) {
    aT.Destroy(); bT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnAddGetWorkspaceSize(aT.tensor, bT.tensor, alpha.scalar, outT.tensor, &workspaceSize, &executor);

  aT.Destroy(); bT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ret != ACLNN_SUCCESS;
}

bool TestAddBadBroadcastError(Runtime* runtime) {
  std::vector<int64_t> aShape = {2, 2};
  std::vector<int64_t> bShape = {3};
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7};
  DeviceTensor aT, bT, outT;
  DeviceScalar alpha;
  float alphaValue = 1.0f;
  bool ok = CreateAclTensorFromVector(a, aShape, ACL_FLOAT, &aT) &&
            CreateAclTensorFromVector(b, bShape, ACL_FLOAT, &bT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(4, 0.0f), aShape, ACL_FLOAT, &outT);
  if (!ok) {
    aT.Destroy(); bT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnAddGetWorkspaceSize(aT.tensor, bT.tensor, alpha.scalar, outT.tensor, &workspaceSize, &executor);

  aT.Destroy(); bT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ret != ACLNN_SUCCESS;
}

bool TestAddsNullScalarError(Runtime* runtime) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> a = {1, 2, 3, 4};
  DeviceTensor aT, outT;
  DeviceScalar alpha;
  float alphaValue = 1.0f;
  bool ok = CreateAclTensorFromVector(a, shape, ACL_FLOAT, &aT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha) &&
            CreateAclTensorFromVector(std::vector<float>(4, 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    aT.Destroy(); outT.Destroy(); alpha.Destroy();
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnAddsGetWorkspaceSize(aT.tensor, nullptr, alpha.scalar, outT.tensor, &workspaceSize, &executor);

  aT.Destroy(); outT.Destroy(); alpha.Destroy();
  return ret != ACLNN_SUCCESS;
}

bool TestInplaceAddBadShapeError(Runtime* runtime) {
  std::vector<int64_t> selfShape = {2, 1};
  std::vector<int64_t> otherShape = {3};
  std::vector<float> self = {1, 2};
  std::vector<float> other = {3, 4, 5};
  DeviceTensor selfT, otherT;
  DeviceScalar alpha;
  float alphaValue = 1.0f;
  bool ok = CreateAclTensorFromVector(self, selfShape, ACL_FLOAT, &selfT) &&
            CreateAclTensorFromVector(other, otherShape, ACL_FLOAT, &otherT) &&
            CreateScalar(alphaValue, ACL_FLOAT, &alpha);
  if (!ok) {
    selfT.Destroy(); otherT.Destroy(); alpha.Destroy();
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = aclnnInplaceAddGetWorkspaceSize(selfT.tensor, otherT.tensor, alpha.scalar, &workspaceSize, &executor);

  selfT.Destroy(); otherT.Destroy(); alpha.Destroy();
  return ret != ACLNN_SUCCESS;
}

bool TestAddV3NullScalarError() {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  DeviceTensor outT;
  std::vector<int64_t> shape = {2, 2};
  bool ok = CreateAclTensorFromVector(std::vector<float>(4, 0.0f), shape, ACL_FLOAT, &outT);
  if (!ok) {
    outT.Destroy();
    return false;
  }
  float alphaValue = 1.0f;
  DeviceScalar alpha;
  ok = CreateScalar(alphaValue, ACL_FLOAT, &alpha);
  if (!ok) {
    outT.Destroy();
    return false;
  }

  aclnnStatus ret = aclnnAddV3GetWorkspaceSize(nullptr, nullptr, alpha.scalar, outT.tensor, &workspaceSize, &executor);
  outT.Destroy();
  alpha.Destroy();
  return ret != ACLNN_SUCCESS;
}

}  // namespace

int main() {
  Runtime runtime;
  auto ret = runtime.Init();
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  Tester t(&runtime);

  t.Run("Add_Float_SameShape_Alpha1", [&]() { return TestAddFloatSameShapeAlpha1(&runtime); });
  t.Run("Add_Float_Axpy_AlphaNonOne", [&]() { return TestAddFloatAxpyAlphaNonOne(&runtime); });
  t.Run("Add_Int32_AxpyV2", [&]() { return TestAddInt32AxpyV2(&runtime); });
  t.Run("Add_Complex64_FallbackMulThenAdd", [&]() { return TestAddComplex64FallbackMulThenAdd(&runtime); });
  t.Run("Add_MixDtype_Float16_Float", [&]() { return TestAddMixDtypeFloat16Float(&runtime); });

  t.Run("Adds_Float_Scalar_Alpha1", [&]() { return TestAddsFloatScalarAlpha1(&runtime); });
  t.Run("Adds_Float_Axpy", [&]() { return TestAddsFloatAxpy(&runtime); });
  t.Run("Adds_Int32_AxpyV2", [&]() { return TestAddsInt32AxpyV2(&runtime); });
  t.Run("Adds_Bool_SpecialCast", [&]() { return TestAddsBoolSpecialCast(&runtime); });

  t.Run("InplaceAdd_Broadcast", [&]() { return TestInplaceAddBroadcast(&runtime); });
  t.Run("InplaceAdds_Float", [&]() { return TestInplaceAddsFloat(&runtime); });

  t.Run("AddV3_Alpha1", [&]() { return TestAddV3Alpha1(&runtime); });
  t.Run("AddV3_Axpy", [&]() { return TestAddV3Axpy(&runtime); });
  t.Run("AddV3_Int8_FallbackMulThenAdd", [&]() { return TestAddV3Int8FallbackMulThenAdd(&runtime); });
  t.Run("InplaceAddV3", [&]() { return TestInplaceAddV3(&runtime); });

  t.Run("Error_Add_Nullptr", [&]() { return TestAddNullptrError(); });
  t.Run("Error_Add_BadOutShape", [&]() { return TestAddBadOutShapeError(&runtime); });
  t.Run("Error_Add_BadBroadcast", [&]() { return TestAddBadBroadcastError(&runtime); });
  t.Run("Error_Adds_NullScalar", [&]() { return TestAddsNullScalarError(&runtime); });
  t.Run("Error_InplaceAdd_BadShape", [&]() { return TestInplaceAddBadShapeError(&runtime); });
  t.Run("Error_AddV3_NullScalar", [&]() { return TestAddV3NullScalarError(); });

  LOG_PRINT("\n=== Test Summary: total=%d passed=%d failed=%d ===\n", t.passed + t.failed, t.passed, t.failed);

  runtime.Destroy();
  return t.failed == 0 ? 0 : 1;
}
