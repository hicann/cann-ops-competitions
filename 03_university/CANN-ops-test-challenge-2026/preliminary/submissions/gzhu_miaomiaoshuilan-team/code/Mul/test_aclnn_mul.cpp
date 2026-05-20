#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, expr) \
  do { \
    if (!(cond)) { \
      expr; \
    } \
  } while (0)

#define LOG_PRINT(fmt, ...) \
  do { \
    printf(fmt, ##__VA_ARGS__); \
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId); aclFinalize(); return ret);
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

bool AlmostEqualFloat(float a, float b, float atol, float rtol) {
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::isinf(a) || std::isinf(b)) {
    return a == b;
  }
  return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

bool AlmostEqualDouble(double a, double b, double atol, double rtol) {
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::isinf(a) || std::isinf(b)) {
    return a == b;
  }
  return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

bool AlmostEqualComplex(const std::complex<double>& a,
                        const std::complex<double>& b,
                        double atol,
                        double rtol) {
  return AlmostEqualDouble(a.real(), b.real(), atol, rtol) &&
         AlmostEqualDouble(a.imag(), b.imag(), atol, rtol);
}

void PrintVector(const std::vector<float>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    LOG_PRINT("%s%.6f", (i == 0 ? "[" : ", "), v[i]);
  }
  LOG_PRINT("]\n");
}

bool CheckFloatVector(const std::vector<float>& actual,
                      const std::vector<float>& expected,
                      float atol,
                      float rtol) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqualFloat(actual[i], expected[i], atol, rtol)) {
      LOG_PRINT("mismatch at %zu: actual=%.8f expected=%.8f\n", i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

bool CheckComplexVector(const std::vector<std::complex<double>>& actual,
                        const std::vector<std::complex<double>>& expected,
                        double atol,
                        double rtol) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("complex size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqualComplex(actual[i], expected[i], atol, rtol)) {
      LOG_PRINT("complex mismatch at %zu: actual=(%.8f, %.8f) expected=(%.8f, %.8f)\n",
                i,
                actual[i].real(), actual[i].imag(),
                expected[i].real(), expected[i].imag());
      return false;
    }
  }
  return true;
}

bool RunMul(Runtime& rt,
            aclTensor* self,
            aclTensor* other,
            aclTensor* out,
            aclnnStatus* apiStatus) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  *apiStatus = ret;
  if (ret != ACL_SUCCESS) {
    return false;
  }

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("workspace alloc failed. ERROR: %d\n", ret);
      return false;
    }
  }
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, rt.stream);
  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(rt.stream);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  *apiStatus = ret;
  return ret == ACL_SUCCESS;
}

bool RunMuls(Runtime& rt,
             aclTensor* self,
             aclScalar* scalar,
             aclTensor* out,
             aclnnStatus* apiStatus) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  *apiStatus = ret;
  if (ret != ACL_SUCCESS) {
    return false;
  }

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("workspace alloc failed. ERROR: %d\n", ret);
      return false;
    }
  }
  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, rt.stream);
  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(rt.stream);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  *apiStatus = ret;
  return ret == ACL_SUCCESS;
}

bool RunInplaceMul(Runtime& rt,
                   aclTensor* selfRef,
                   aclTensor* other,
                   aclnnStatus* apiStatus) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
  *apiStatus = ret;
  if (ret != ACL_SUCCESS) {
    return false;
  }

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("workspace alloc failed. ERROR: %d\n", ret);
      return false;
    }
  }
  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, rt.stream);
  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(rt.stream);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  *apiStatus = ret;
  return ret == ACL_SUCCESS;
}

bool RunInplaceMuls(Runtime& rt,
                    aclTensor* selfRef,
                    aclScalar* scalar,
                    aclnnStatus* apiStatus) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulsGetWorkspaceSize(selfRef, scalar, &workspaceSize, &executor);
  *apiStatus = ret;
  if (ret != ACL_SUCCESS) {
    return false;
  }

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("workspace alloc failed. ERROR: %d\n", ret);
      return false;
    }
  }
  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, rt.stream);
  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(rt.stream);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  *apiStatus = ret;
  return ret == ACL_SUCCESS;
}

using TestFn = std::function<bool(Runtime&)>;

bool TestMulFloatSameShape(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  DeviceTensor out;
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> a = {0.f, 1.f, 2.f, -3.f, 4.5f, -5.f};
  std::vector<float> b = {1.f, -2.f, 3.f, 2.f, -1.f, 0.5f};
  std::vector<float> expect = {0.f, -2.f, 6.f, -6.f, -4.5f, -2.5f};
  std::vector<float> actual(expect.size(), 0.f);

  bool ok = CreateAclTensorFromVector(a, shape, ACL_FLOAT, &self) &&
            CreateAclTensorFromVector(b, shape, ACL_FLOAT, &other) &&
            CreateAclTensorFromVector(std::vector<float>(expect.size(), 0.f), shape, ACL_FLOAT, &out);
  if (!ok) {
    self.Destroy(); other.Destroy(); out.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunMul(rt, self.tensor, other.tensor, out.tensor, &st) && CopyTensorToHost(out, &actual) &&
       CheckFloatVector(actual, expect, 1e-5f, 1e-5f);

  self.Destroy(); other.Destroy(); out.Destroy();
  return ok;
}

bool TestMulFloatBroadcast(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  DeviceTensor out;
  std::vector<float> a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> b = {10.f, -1.f, 0.5f};
  std::vector<float> expect = {10.f, -2.f, 1.5f, 40.f, -5.f, 3.f};
  std::vector<float> actual(expect.size(), 0.f);

  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 3}, ACL_FLOAT, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{3}, ACL_FLOAT, &other) &&
            CreateAclTensorFromVector(std::vector<float>(expect.size(), 0.f), std::vector<int64_t>{2, 3}, ACL_FLOAT, &out);
  if (!ok) {
    self.Destroy(); other.Destroy(); out.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunMul(rt, self.tensor, other.tensor, out.tensor, &st) && CopyTensorToHost(out, &actual) &&
       CheckFloatVector(actual, expect, 1e-5f, 1e-5f);

  self.Destroy(); other.Destroy(); out.Destroy();
  return ok;
}

bool TestMulFloat16FloatMix(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  DeviceTensor out;
  std::vector<uint16_t> a;
  a.push_back(FloatToFp16Bits(1.0f));
  a.push_back(FloatToFp16Bits(-2.0f));
  a.push_back(FloatToFp16Bits(0.5f));
  a.push_back(FloatToFp16Bits(8.0f));
  std::vector<float> b = {2.f, -3.f, 4.f, 0.25f};
  std::vector<float> expect = {2.f, 6.f, 2.f, 2.f};
  std::vector<float> actual(expect.size(), 0.f);

  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT16, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other) &&
            CreateAclTensorFromVector(std::vector<float>(expect.size(), 0.f), std::vector<int64_t>{2, 2}, ACL_FLOAT, &out);
  if (!ok) {
    self.Destroy(); other.Destroy(); out.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunMul(rt, self.tensor, other.tensor, out.tensor, &st) && CopyTensorToHost(out, &actual) &&
       CheckFloatVector(actual, expect, 1e-4f, 1e-4f);

  self.Destroy(); other.Destroy(); out.Destroy();
  return ok;
}

bool TestMulPromoteCastInt32Float(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  DeviceTensor out;
  std::vector<int32_t> a = {1, 2, -3, 4};
  std::vector<float> b = {0.5f, -1.f, 2.f, -0.25f};
  std::vector<float> expect = {0.5f, -2.f, -6.f, -1.f};
  std::vector<float> actual(expect.size(), 0.f);

  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_INT32, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other) &&
            CreateAclTensorFromVector(std::vector<float>(expect.size(), 0.f), std::vector<int64_t>{2, 2}, ACL_FLOAT, &out);
  if (!ok) {
    self.Destroy(); other.Destroy(); out.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunMul(rt, self.tensor, other.tensor, out.tensor, &st) && CopyTensorToHost(out, &actual) &&
       CheckFloatVector(actual, expect, 1e-5f, 1e-5f);

  self.Destroy(); other.Destroy(); out.Destroy();
  return ok;
}

bool TestMulComplex128AiCpu(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  DeviceTensor out;
  std::vector<std::complex<double> > a;
  std::vector<std::complex<double> > b;
  a.push_back(std::complex<double>(1.0, 2.0));
  a.push_back(std::complex<double>(-3.0, 0.5));
  a.push_back(std::complex<double>(0.0, -1.0));
  a.push_back(std::complex<double>(2.0, 2.0));
  b.push_back(std::complex<double>(2.0, -1.0));
  b.push_back(std::complex<double>(0.5, 4.0));
  b.push_back(std::complex<double>(3.0, 0.0));
  b.push_back(std::complex<double>(-1.0, 1.0));

  std::vector<std::complex<double> > expect(4);
  for (size_t i = 0; i < 4; ++i) {
    expect[i] = a[i] * b[i];
  }
  std::vector<std::complex<double> > actual(4);

  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_COMPLEX128, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{2, 2}, ACL_COMPLEX128, &other) &&
            CreateAclTensorFromVector(std::vector<std::complex<double> >(4, std::complex<double>(0.0, 0.0)),
                                      std::vector<int64_t>{2, 2}, ACL_COMPLEX128, &out);
  if (!ok) {
    self.Destroy(); other.Destroy(); out.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunMul(rt, self.tensor, other.tensor, out.tensor, &st) && CopyTensorToHost(out, &actual) &&
       CheckComplexVector(actual, expect, 1e-10, 1e-10);

  self.Destroy(); other.Destroy(); out.Destroy();
  return ok;
}

bool TestMulsFloatScalar(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor out;
  std::vector<float> a = {1.f, -2.f, 3.f, -4.f};
  float alpha = 1.5f;
  aclScalar* scalar = aclCreateScalar(&alpha, ACL_FLOAT);
  std::vector<float> expect = {1.5f, -3.f, 4.5f, -6.f};
  std::vector<float> actual(expect.size(), 0.f);

  bool ok = (scalar != nullptr) &&
            CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self) &&
            CreateAclTensorFromVector(std::vector<float>(expect.size(), 0.f), std::vector<int64_t>{2, 2}, ACL_FLOAT, &out);
  if (!ok) {
    if (scalar != nullptr) aclDestroyScalar(scalar);
    self.Destroy(); out.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunMuls(rt, self.tensor, scalar, out.tensor, &st) && CopyTensorToHost(out, &actual) &&
       CheckFloatVector(actual, expect, 1e-5f, 1e-5f);

  aclDestroyScalar(scalar);
  self.Destroy(); out.Destroy();
  return ok;
}

bool TestMulsFloat16CanUseMuls(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor out;
  std::vector<uint16_t> a;
  a.push_back(FloatToFp16Bits(1.0f));
  a.push_back(FloatToFp16Bits(-2.0f));
  a.push_back(FloatToFp16Bits(0.5f));
  a.push_back(FloatToFp16Bits(4.0f));
  float alpha = 1.5f;
  aclScalar* scalar = aclCreateScalar(&alpha, ACL_FLOAT);
  std::vector<float> expect = {1.5f, -3.0f, 0.75f, 6.0f};
  std::vector<uint16_t> actualBits(4, 0);
  std::vector<float> actual(4, 0.f);

  bool ok = (scalar != nullptr) &&
            CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT16, &self) &&
            CreateAclTensorFromVector(std::vector<uint16_t>(4, 0), std::vector<int64_t>{2, 2}, ACL_FLOAT16, &out);
  if (!ok) {
    if (scalar != nullptr) aclDestroyScalar(scalar);
    self.Destroy(); out.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunMuls(rt, self.tensor, scalar, out.tensor, &st) && CopyTensorToHost(out, &actualBits);
  if (ok) {
    for (size_t i = 0; i < actualBits.size(); ++i) {
      actual[i] = Fp16BitsToFloat(actualBits[i]);
    }
    ok = CheckFloatVector(actual, expect, 1e-3f, 1e-3f);
  }

  aclDestroyScalar(scalar);
  self.Destroy(); out.Destroy();
  return ok;
}

bool TestInplaceMulBroadcast(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  std::vector<float> a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> b = {2.f, -1.f, 0.5f};
  std::vector<float> expect = {2.f, -2.f, 1.5f, 8.f, -5.f, 3.f};
  std::vector<float> actual(expect.size(), 0.f);

  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 3}, ACL_FLOAT, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{3}, ACL_FLOAT, &other);
  if (!ok) {
    self.Destroy(); other.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunInplaceMul(rt, self.tensor, other.tensor, &st) && CopyTensorToHost(self, &actual) &&
       CheckFloatVector(actual, expect, 1e-5f, 1e-5f);

  self.Destroy(); other.Destroy();
  return ok;
}

bool TestInplaceMulFloat16FloatMix(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  std::vector<uint16_t> a;
  a.push_back(FloatToFp16Bits(1.0f));
  a.push_back(FloatToFp16Bits(-2.0f));
  a.push_back(FloatToFp16Bits(0.5f));
  a.push_back(FloatToFp16Bits(4.0f));
  std::vector<float> b = {2.f, -3.f, 4.f, 0.25f};
  std::vector<float> expect = {2.f, 6.f, 2.f, 1.f};
  std::vector<uint16_t> actualBits(4, 0);
  std::vector<float> actual(4, 0.f);

  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT16, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other);
  if (!ok) {
    self.Destroy(); other.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunInplaceMul(rt, self.tensor, other.tensor, &st) && CopyTensorToHost(self, &actualBits);
  if (ok) {
    for (size_t i = 0; i < actualBits.size(); ++i) {
      actual[i] = Fp16BitsToFloat(actualBits[i]);
    }
    ok = CheckFloatVector(actual, expect, 1e-3f, 1e-3f);
  }

  self.Destroy(); other.Destroy();
  return ok;
}

bool TestInplaceMulsFloatScalar(Runtime& rt) {
  DeviceTensor self;
  std::vector<float> a = {1.f, -2.f, 3.f, -4.f};
  float alpha = -2.f;
  aclScalar* scalar = aclCreateScalar(&alpha, ACL_FLOAT);
  std::vector<float> expect = {-2.f, 4.f, -6.f, 8.f};
  std::vector<float> actual(expect.size(), 0.f);

  bool ok = (scalar != nullptr) &&
            CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self);
  if (!ok) {
    if (scalar != nullptr) aclDestroyScalar(scalar);
    self.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunInplaceMuls(rt, self.tensor, scalar, &st) && CopyTensorToHost(self, &actual) &&
       CheckFloatVector(actual, expect, 1e-5f, 1e-5f);

  aclDestroyScalar(scalar);
  self.Destroy();
  return ok;
}

bool TestInplaceMulsFloat16CanUseMuls(Runtime& rt) {
  DeviceTensor self;
  std::vector<uint16_t> a;
  a.push_back(FloatToFp16Bits(1.0f));
  a.push_back(FloatToFp16Bits(-2.0f));
  a.push_back(FloatToFp16Bits(0.5f));
  a.push_back(FloatToFp16Bits(4.0f));
  float alpha = 1.5f;
  aclScalar* scalar = aclCreateScalar(&alpha, ACL_FLOAT);
  std::vector<float> expect = {1.5f, -3.0f, 0.75f, 6.0f};
  std::vector<uint16_t> actualBits(4, 0);
  std::vector<float> actual(4, 0.f);

  bool ok = (scalar != nullptr) &&
            CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT16, &self);
  if (!ok) {
    if (scalar != nullptr) aclDestroyScalar(scalar);
    self.Destroy();
    return false;
  }

  aclnnStatus st = ACL_SUCCESS;
  ok = RunInplaceMuls(rt, self.tensor, scalar, &st) && CopyTensorToHost(self, &actualBits);
  if (ok) {
    for (size_t i = 0; i < actualBits.size(); ++i) {
      actual[i] = Fp16BitsToFloat(actualBits[i]);
    }
    ok = CheckFloatVector(actual, expect, 1e-3f, 1e-3f);
  }

  aclDestroyScalar(scalar);
  self.Destroy();
  return ok;
}

bool TestMulNullptrFail(Runtime&) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus st = aclnnMulGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor);
  return st != ACL_SUCCESS;
}

bool TestMulBadOutShapeFail(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  DeviceTensor out;
  std::vector<float> a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> b = {10.f, -1.f, 0.5f};
  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 3}, ACL_FLOAT, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{3}, ACL_FLOAT, &other) &&
            CreateAclTensorFromVector(std::vector<float>(4, 0.f), std::vector<int64_t>{2, 2}, ACL_FLOAT, &out);
  if (!ok) {
    self.Destroy(); other.Destroy(); out.Destroy();
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus st = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  self.Destroy(); other.Destroy(); out.Destroy();
  return st != ACL_SUCCESS;
}

bool TestMulsShapeMismatchFail(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor out;
  std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
  float alpha = 2.f;
  aclScalar* scalar = aclCreateScalar(&alpha, ACL_FLOAT);
  bool ok = (scalar != nullptr) &&
            CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self) &&
            CreateAclTensorFromVector(std::vector<float>(4, 0.f), std::vector<int64_t>{4, 1}, ACL_FLOAT, &out);
  if (!ok) {
    if (scalar != nullptr) aclDestroyScalar(scalar);
    self.Destroy(); out.Destroy();
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus st = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  aclDestroyScalar(scalar);
  self.Destroy(); out.Destroy();
  return st != ACL_SUCCESS;
}

bool TestInplaceMulBadBroadcastFail(Runtime& rt) {
  DeviceTensor self;
  DeviceTensor other;
  std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> b = {1.f, 2.f, 3.f};
  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self) &&
            CreateAclTensorFromVector(b, std::vector<int64_t>{3}, ACL_FLOAT, &other);
  if (!ok) {
    self.Destroy(); other.Destroy();
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus st = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  self.Destroy(); other.Destroy();
  return st != ACL_SUCCESS;
}

bool TestInplaceMulsNullScalarFail(Runtime& rt) {
  DeviceTensor self;
  std::vector<float> a = {1.f, 2.f, 3.f, 4.f};
  bool ok = CreateAclTensorFromVector(a, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self);
  if (!ok) {
    self.Destroy();
    return false;
  }
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus st = aclnnInplaceMulsGetWorkspaceSize(self.tensor, nullptr, &workspaceSize, &executor);
  self.Destroy();
  return st != ACL_SUCCESS;
}

struct TestCase {
  const char* name;
  TestFn fn;
};

}  // namespace

int main() {
  Runtime rt;
  int ret = rt.Init();
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<TestCase> tests;
  tests.push_back({"Mul_Float_SameShape", TestMulFloatSameShape});
  tests.push_back({"Mul_Float_Broadcast", TestMulFloatBroadcast});
  tests.push_back({"Mul_Float16_Float_MixDtype", TestMulFloat16FloatMix});
  tests.push_back({"Mul_Int32_Float_PromoteCast", TestMulPromoteCastInt32Float});
  tests.push_back({"Mul_Complex128_AiCpu", TestMulComplex128AiCpu});
  tests.push_back({"Muls_Float_Scalar", TestMulsFloatScalar});
  tests.push_back({"Muls_Float16_Scalar_CanUseMuls", TestMulsFloat16CanUseMuls});
  tests.push_back({"InplaceMul_Float_Broadcast", TestInplaceMulBroadcast});
  tests.push_back({"InplaceMul_Float16_Float_MixDtype", TestInplaceMulFloat16FloatMix});
  tests.push_back({"InplaceMuls_Float_Scalar", TestInplaceMulsFloatScalar});
  tests.push_back({"InplaceMuls_Float16_Scalar_CanUseMuls", TestInplaceMulsFloat16CanUseMuls});
  tests.push_back({"Mul_Nullptr_Fail", TestMulNullptrFail});
  tests.push_back({"Mul_BadOutShape_Fail", TestMulBadOutShapeFail});
  tests.push_back({"Muls_ShapeMismatch_Fail", TestMulsShapeMismatchFail});
  tests.push_back({"InplaceMul_BadBroadcast_Fail", TestInplaceMulBadBroadcastFail});
  tests.push_back({"InplaceMuls_NullScalar_Fail", TestInplaceMulsNullScalarFail});

  int passed = 0;
  int failed = 0;
  for (size_t i = 0; i < tests.size(); ++i) {
    bool ok = false;
    try {
      ok = tests[i].fn(rt);
    } catch (...) {
      ok = false;
    }
    if (ok) {
      ++passed;
      LOG_PRINT("[PASS] %s\n", tests[i].name);
    } else {
      ++failed;
      LOG_PRINT("[FAIL] %s\n", tests[i].name);
    }
  }

  LOG_PRINT("===== Test Summary: total=%zu passed=%d failed=%d =====\n", tests.size(), passed, failed);
  rt.Destroy();
  return failed == 0 ? 0 : 1;
}
