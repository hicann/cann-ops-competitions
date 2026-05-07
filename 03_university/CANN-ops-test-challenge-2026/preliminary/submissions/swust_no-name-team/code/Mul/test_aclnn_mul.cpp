#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do {                                \
    if (!(cond)) {                    \
      return_expr;                    \
    }                                 \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
  const auto elemCount = GetShapeSize(shape);
  const auto size = elemCount * static_cast<int64_t>(sizeof(T));

  auto ret = aclrtMalloc(deviceAddr, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  if (size > 0) {
    ret = aclrtMemcpy(*deviceAddr, static_cast<size_t>(size), hostData.data(), static_cast<size_t>(size),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

template <typename T>
void DestroyTensorPair(void* dev, aclTensor* tensor)
{
  if (tensor != nullptr) {
    aclDestroyTensor(tensor);
  }
  if (dev != nullptr) {
    aclrtFree(dev);
  }
}

bool AlmostEqual(double expected, double actual, double atol, double rtol)
{
  if (std::isnan(expected) && std::isnan(actual)) {
    return true;
  }
  if (std::isinf(expected) && std::isinf(actual)) {
    return (expected > 0) == (actual > 0);
  }
  return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

float Fp16BitsToFloat(uint16_t bits)
{
  const uint16_t sign = (bits >> 15) & 0x1;
  const uint16_t exp = (bits >> 10) & 0x1F;
  const uint16_t frac = bits & 0x3FF;

  if (exp == 0) {
    if (frac == 0) {
      return sign ? -0.0f : 0.0f;
    }
    const float m = static_cast<float>(frac) / 1024.0f;
    const float v = std::ldexp(m, -14);
    return sign ? -v : v;
  }

  if (exp == 31) {
    if (frac == 0) {
      return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::quiet_NaN();
  }

  const float m = 1.0f + static_cast<float>(frac) / 1024.0f;
  const float v = std::ldexp(m, static_cast<int>(exp) - 15);
  return sign ? -v : v;
}

float Bf16BitsToFloat(uint16_t bits)
{
  union {
    uint32_t u;
    float f;
  } cvt;
  cvt.u = static_cast<uint32_t>(bits) << 16;
  return cvt.f;
}

int RunMulFloatCase(const char* name, const std::vector<float>& x1, const std::vector<float>& x2,
                    const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2,
                    const std::vector<int64_t>& outShape, double atol, double rtol, aclrtStream stream)
{
  const int64_t nOut = GetShapeSize(outShape);
  void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
  aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;

  auto ret = CreateAclTensor(x1, shape1, &x1Dev, ACL_FLOAT, &x1T);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(x2, shape2, &x2Dev, ACL_FLOAT, &x2T);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(x1Dev, x1T); return 1);

  std::vector<float> outHost(static_cast<size_t>(nOut), 0.0f);
  ret = CreateAclTensor(outHost, outShape, &outDev, ACL_FLOAT, &outT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<float>(x1Dev, x1T); DestroyTensorPair<float>(x2Dev, x2T); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize=%d\n", name, ret);
    DestroyTensorPair<float>(x1Dev, x1T);
    DestroyTensorPair<float>(x2Dev, x2T);
    DestroyTensorPair<float>(outDev, outT);
    return 1;
  }

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: alloc workspace failed\n", name);
              DestroyTensorPair<float>(x1Dev, x1T); DestroyTensorPair<float>(x2Dev, x2T);
              DestroyTensorPair<float>(outDev, outT); return 1);
  }

  ret = aclnnMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMul=%d\n", name, ret);
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<float>(x1Dev, x1T); DestroyTensorPair<float>(x2Dev, x2T);
            DestroyTensorPair<float>(outDev, outT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclrtSynchronizeStream=%d\n", name, ret);
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<float>(x1Dev, x1T); DestroyTensorPair<float>(x2Dev, x2T);
            DestroyTensorPair<float>(outDev, outT); return 1);

  if (nOut > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(nOut * sizeof(float)), outDev,
                      static_cast<size_t>(nOut * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: copy out failed\n", name);
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<float>(x1Dev, x1T); DestroyTensorPair<float>(x2Dev, x2T);
              DestroyTensorPair<float>(outDev, outT); return 1);
  }

  int failed = 0;
  if (shape1 == shape2 && shape1 == outShape) {
    for (int64_t i = 0; i < nOut; ++i) {
      const double expected = static_cast<double>(x1[static_cast<size_t>(i)]) * static_cast<double>(x2[static_cast<size_t>(i)]);
      if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], atol, rtol)) {
        ++failed;
      }
    }
  } else if (shape1.size() == 2 && shape2.size() == 1 && shape1[1] == shape2[0] && outShape == shape1) {
    for (int64_t i = 0; i < shape1[0]; ++i) {
      for (int64_t j = 0; j < shape1[1]; ++j) {
        const double expected = static_cast<double>(x1[static_cast<size_t>(i * shape1[1] + j)]) *
                                static_cast<double>(x2[static_cast<size_t>(j)]);
        if (!AlmostEqual(expected, outHost[static_cast<size_t>(i * shape1[1] + j)], atol, rtol)) {
          ++failed;
        }
      }
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<float>(x1Dev, x1T);
  DestroyTensorPair<float>(x2Dev, x2T);
  DestroyTensorPair<float>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

int RunMulIntCaseI32(const char* name, const std::vector<int32_t>& x1, const std::vector<int32_t>& x2,
                     const std::vector<int64_t>& shape, aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
  aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;

  auto ret = CreateAclTensor(x1, shape, &x1Dev, ACL_INT32, &x1T);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(x2, shape, &x2Dev, ACL_INT32, &x2T);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<int32_t>(x1Dev, x1T); return 1);

  std::vector<int32_t> outHost(static_cast<size_t>(n), 0);
  ret = CreateAclTensor(outHost, shape, &outDev, ACL_INT32, &outT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<int32_t>(x1Dev, x1T); DestroyTensorPair<int32_t>(x2Dev, x2T); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize=%d\n", name, ret);
            DestroyTensorPair<int32_t>(x1Dev, x1T); DestroyTensorPair<int32_t>(x2Dev, x2T);
            DestroyTensorPair<int32_t>(outDev, outT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<int32_t>(x1Dev, x1T); DestroyTensorPair<int32_t>(x2Dev, x2T);
              DestroyTensorPair<int32_t>(outDev, outT); return 1);
  }

  ret = aclnnMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<int32_t>(x1Dev, x1T); DestroyTensorPair<int32_t>(x2Dev, x2T);
            DestroyTensorPair<int32_t>(outDev, outT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<int32_t>(x1Dev, x1T); DestroyTensorPair<int32_t>(x2Dev, x2T);
            DestroyTensorPair<int32_t>(outDev, outT); return 1);

  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(int32_t)), outDev,
                      static_cast<size_t>(n * sizeof(int32_t)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<int32_t>(x1Dev, x1T); DestroyTensorPair<int32_t>(x2Dev, x2T);
              DestroyTensorPair<int32_t>(outDev, outT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t expected = static_cast<int64_t>(x1[static_cast<size_t>(i)]) *
                             static_cast<int64_t>(x2[static_cast<size_t>(i)]);
    if (expected != outHost[static_cast<size_t>(i)]) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<int32_t>(x1Dev, x1T);
  DestroyTensorPair<int32_t>(x2Dev, x2T);
  DestroyTensorPair<int32_t>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

template <typename T>
int RunMulIntegralCase(const char* name, const std::vector<T>& x1, const std::vector<T>& x2,
                       const std::vector<int64_t>& shape, aclDataType dtype, aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
  aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;

  auto ret = CreateAclTensor(x1, shape, &x1Dev, dtype, &x1T);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(x2, shape, &x2Dev, dtype, &x2T);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<T>(x1Dev, x1T); return 1);

  std::vector<T> outHost(static_cast<size_t>(n), static_cast<T>(0));
  ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<T>(x1Dev, x1T); DestroyTensorPair<T>(x2Dev, x2T); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize=%d\n", name, ret);
            DestroyTensorPair<T>(x1Dev, x1T); DestroyTensorPair<T>(x2Dev, x2T); DestroyTensorPair<T>(outDev, outT);
            return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<T>(x1Dev, x1T); DestroyTensorPair<T>(x2Dev, x2T); DestroyTensorPair<T>(outDev, outT);
              return 1);
  }

  ret = aclnnMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<T>(x1Dev, x1T); DestroyTensorPair<T>(x2Dev, x2T); DestroyTensorPair<T>(outDev, outT);
            return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<T>(x1Dev, x1T); DestroyTensorPair<T>(x2Dev, x2T); DestroyTensorPair<T>(outDev, outT);
            return 1);

  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(T)), outDev, static_cast<size_t>(n * sizeof(T)),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<T>(x1Dev, x1T); DestroyTensorPair<T>(x2Dev, x2T); DestroyTensorPair<T>(outDev, outT);
              return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const long long expected = static_cast<long long>(x1[static_cast<size_t>(i)]) *
                               static_cast<long long>(x2[static_cast<size_t>(i)]);
    if (static_cast<long long>(outHost[static_cast<size_t>(i)]) != expected) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<T>(x1Dev, x1T);
  DestroyTensorPair<T>(x2Dev, x2T);
  DestroyTensorPair<T>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

int RunMulDoubleCase(const char* name, const std::vector<double>& x1, const std::vector<double>& x2,
                     const std::vector<int64_t>& shape, aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
  aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;

  auto ret = CreateAclTensor(x1, shape, &x1Dev, ACL_DOUBLE, &x1T);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(x2, shape, &x2Dev, ACL_DOUBLE, &x2T);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<double>(x1Dev, x1T); return 1);

  std::vector<double> outHost(static_cast<size_t>(n), 0.0);
  ret = CreateAclTensor(outHost, shape, &outDev, ACL_DOUBLE, &outT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<double>(x1Dev, x1T); DestroyTensorPair<double>(x2Dev, x2T); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize=%d\n", name, ret);
            DestroyTensorPair<double>(x1Dev, x1T); DestroyTensorPair<double>(x2Dev, x2T);
            DestroyTensorPair<double>(outDev, outT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<double>(x1Dev, x1T); DestroyTensorPair<double>(x2Dev, x2T);
              DestroyTensorPair<double>(outDev, outT); return 1);
  }

  ret = aclnnMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<double>(x1Dev, x1T); DestroyTensorPair<double>(x2Dev, x2T);
            DestroyTensorPair<double>(outDev, outT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<double>(x1Dev, x1T); DestroyTensorPair<double>(x2Dev, x2T);
            DestroyTensorPair<double>(outDev, outT); return 1);

  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(double)), outDev,
                      static_cast<size_t>(n * sizeof(double)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<double>(x1Dev, x1T); DestroyTensorPair<double>(x2Dev, x2T);
              DestroyTensorPair<double>(outDev, outT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double expected = x1[static_cast<size_t>(i)] * x2[static_cast<size_t>(i)];
    if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-12, 1e-12)) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<double>(x1Dev, x1T);
  DestroyTensorPair<double>(x2Dev, x2T);
  DestroyTensorPair<double>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

int RunMulBoolCase(const char* name, const std::vector<uint8_t>& x1, const std::vector<uint8_t>& x2,
                   const std::vector<int64_t>& shape, aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
  aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;

  auto ret = CreateAclTensor(x1, shape, &x1Dev, ACL_BOOL, &x1T);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(x2, shape, &x2Dev, ACL_BOOL, &x2T);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<uint8_t>(x1Dev, x1T); return 1);

  std::vector<uint8_t> outHost(static_cast<size_t>(n), 0);
  ret = CreateAclTensor(outHost, shape, &outDev, ACL_BOOL, &outT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<uint8_t>(x1Dev, x1T); DestroyTensorPair<uint8_t>(x2Dev, x2T); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize=%d\n", name, ret);
            DestroyTensorPair<uint8_t>(x1Dev, x1T); DestroyTensorPair<uint8_t>(x2Dev, x2T);
            DestroyTensorPair<uint8_t>(outDev, outT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<uint8_t>(x1Dev, x1T); DestroyTensorPair<uint8_t>(x2Dev, x2T);
              DestroyTensorPair<uint8_t>(outDev, outT); return 1);
  }

  ret = aclnnMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<uint8_t>(x1Dev, x1T); DestroyTensorPair<uint8_t>(x2Dev, x2T);
            DestroyTensorPair<uint8_t>(outDev, outT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<uint8_t>(x1Dev, x1T); DestroyTensorPair<uint8_t>(x2Dev, x2T);
            DestroyTensorPair<uint8_t>(outDev, outT); return 1);

  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(uint8_t)), outDev,
                      static_cast<size_t>(n * sizeof(uint8_t)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<uint8_t>(x1Dev, x1T); DestroyTensorPair<uint8_t>(x2Dev, x2T);
              DestroyTensorPair<uint8_t>(outDev, outT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const uint8_t expected = static_cast<uint8_t>((x1[static_cast<size_t>(i)] != 0) && (x2[static_cast<size_t>(i)] != 0));
    const uint8_t got = static_cast<uint8_t>(outHost[static_cast<size_t>(i)] != 0);
    if (expected != got) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<uint8_t>(x1Dev, x1T);
  DestroyTensorPair<uint8_t>(x2Dev, x2T);
  DestroyTensorPair<uint8_t>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

int RunMulRaw16Case(const char* name, const std::vector<uint16_t>& x1, const std::vector<uint16_t>& x2,
                    const std::vector<uint16_t>& expected, const std::vector<int64_t>& shape, aclDataType dtype,
                    aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
  aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;

  auto ret = CreateAclTensor(x1, shape, &x1Dev, dtype, &x1T);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(x2, shape, &x2Dev, dtype, &x2T);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<uint16_t>(x1Dev, x1T); return 1);

  std::vector<uint16_t> outHost(static_cast<size_t>(n), 0);
  ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<uint16_t>(x1Dev, x1T); DestroyTensorPair<uint16_t>(x2Dev, x2T); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize=%d\n", name, ret);
            DestroyTensorPair<uint16_t>(x1Dev, x1T); DestroyTensorPair<uint16_t>(x2Dev, x2T);
            DestroyTensorPair<uint16_t>(outDev, outT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<uint16_t>(x1Dev, x1T); DestroyTensorPair<uint16_t>(x2Dev, x2T);
              DestroyTensorPair<uint16_t>(outDev, outT); return 1);
  }

  ret = aclnnMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<uint16_t>(x1Dev, x1T); DestroyTensorPair<uint16_t>(x2Dev, x2T);
            DestroyTensorPair<uint16_t>(outDev, outT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<uint16_t>(x1Dev, x1T); DestroyTensorPair<uint16_t>(x2Dev, x2T);
            DestroyTensorPair<uint16_t>(outDev, outT); return 1);

  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(uint16_t)), outDev,
                      static_cast<size_t>(n * sizeof(uint16_t)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<uint16_t>(x1Dev, x1T); DestroyTensorPair<uint16_t>(x2Dev, x2T);
              DestroyTensorPair<uint16_t>(outDev, outT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (outHost[static_cast<size_t>(i)] != expected[static_cast<size_t>(i)]) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<uint16_t>(x1Dev, x1T);
  DestroyTensorPair<uint16_t>(x2Dev, x2T);
  DestroyTensorPair<uint16_t>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

int RunMulMix16ToFloatCase(const char* name, const std::vector<uint16_t>& x16, const std::vector<float>& xf,
                           const std::vector<int64_t>& shape, aclDataType in16Type, bool isFp16,
                           aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *x16Dev = nullptr, *xfDev = nullptr, *outDev = nullptr;
  aclTensor *x16T = nullptr, *xfT = nullptr, *outT = nullptr;

  auto ret = CreateAclTensor(x16, shape, &x16Dev, in16Type, &x16T);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(xf, shape, &xfDev, ACL_FLOAT, &xfT);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<uint16_t>(x16Dev, x16T); return 1);

  std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
  ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<uint16_t>(x16Dev, x16T); DestroyTensorPair<float>(xfDev, xfT); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(x16T, xfT, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize=%d\n", name, ret);
            DestroyTensorPair<uint16_t>(x16Dev, x16T); DestroyTensorPair<float>(xfDev, xfT);
            DestroyTensorPair<float>(outDev, outT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<uint16_t>(x16Dev, x16T); DestroyTensorPair<float>(xfDev, xfT);
              DestroyTensorPair<float>(outDev, outT); return 1);
  }

  ret = aclnnMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<uint16_t>(x16Dev, x16T); DestroyTensorPair<float>(xfDev, xfT);
            DestroyTensorPair<float>(outDev, outT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<uint16_t>(x16Dev, x16T); DestroyTensorPair<float>(xfDev, xfT);
            DestroyTensorPair<float>(outDev, outT); return 1);

  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), outDev,
                      static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<uint16_t>(x16Dev, x16T); DestroyTensorPair<float>(xfDev, xfT);
              DestroyTensorPair<float>(outDev, outT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const float a = isFp16 ? Fp16BitsToFloat(x16[static_cast<size_t>(i)]) : Bf16BitsToFloat(x16[static_cast<size_t>(i)]);
    const float expected = a * xf[static_cast<size_t>(i)];
    if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-3, 1e-3)) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<uint16_t>(x16Dev, x16T);
  DestroyTensorPair<float>(xfDev, xfT);
  DestroyTensorPair<float>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

int RunMulsFloatCase(const char* name, const std::vector<float>& self, const std::vector<int64_t>& shape,
                     float scalarValue, aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *selfDev = nullptr, *outDev = nullptr;
  aclTensor *selfT = nullptr, *outT = nullptr;
  auto ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
  ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(selfDev, selfT); return 1);

  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr,
            DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(outDev, outT); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulsGetWorkspaceSize(selfT, scalar, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnMulsGetWorkspaceSize=%d\n", name, ret);
            aclDestroyScalar(scalar);
            DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(outDev, outT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              aclDestroyScalar(scalar);
              DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(outDev, outT); return 1);
  }

  ret = aclnnMuls(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            aclDestroyScalar(scalar);
            DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(outDev, outT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            aclDestroyScalar(scalar);
            DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(outDev, outT); return 1);

  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), outDev,
                      static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              aclDestroyScalar(scalar);
              DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(outDev, outT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double expected = static_cast<double>(self[static_cast<size_t>(i)]) * static_cast<double>(scalarValue);
    if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  aclDestroyScalar(scalar);
  DestroyTensorPair<float>(selfDev, selfT);
  DestroyTensorPair<float>(outDev, outT);
  return failed == 0 ? 0 : 1;
}

int RunInplaceMulFloatCase(const char* name, const std::vector<float>& self, const std::vector<float>& other,
                           const std::vector<int64_t>& shape, aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void *selfDev = nullptr, *otherDev = nullptr;
  aclTensor *selfT = nullptr, *otherT = nullptr;

  auto ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(other, shape, &otherDev, ACL_FLOAT, &otherT);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(selfDev, selfT); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnInplaceMulGetWorkspaceSize(selfT, otherT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnInplaceMulGetWorkspaceSize=%d\n", name, ret);
            DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(otherDev, otherT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(otherDev, otherT); return 1);
  }

  ret = aclnnInplaceMul(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(otherDev, otherT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(otherDev, otherT); return 1);

  std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), selfDev,
                      static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              DestroyTensorPair<float>(selfDev, selfT); DestroyTensorPair<float>(otherDev, otherT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double expected = static_cast<double>(self[static_cast<size_t>(i)]) *
                            static_cast<double>(other[static_cast<size_t>(i)]);
    if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  DestroyTensorPair<float>(selfDev, selfT);
  DestroyTensorPair<float>(otherDev, otherT);
  return failed == 0 ? 0 : 1;
}

int RunInplaceMulsFloatCase(const char* name, const std::vector<float>& self, const std::vector<int64_t>& shape,
                            float scalarValue, aclrtStream stream)
{
  const int64_t n = GetShapeSize(shape);
  void* selfDev = nullptr;
  aclTensor* selfT = nullptr;

  auto ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorPair<float>(selfDev, selfT); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnInplaceMulsGetWorkspaceSize(selfT, scalar, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnInplaceMulsGetWorkspaceSize=%d\n", name, ret);
            aclDestroyScalar(scalar); DestroyTensorPair<float>(selfDev, selfT); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) {
    ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, aclDestroyScalar(scalar); DestroyTensorPair<float>(selfDev, selfT); return 1);
  }

  ret = aclnnInplaceMuls(wsAddr, wsSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            aclDestroyScalar(scalar); DestroyTensorPair<float>(selfDev, selfT); return 1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            if (wsAddr != nullptr) { aclrtFree(wsAddr); }
            aclDestroyScalar(scalar); DestroyTensorPair<float>(selfDev, selfT); return 1);

  std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
  if (n > 0) {
    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), selfDev,
                      static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (wsAddr != nullptr) { aclrtFree(wsAddr); }
              aclDestroyScalar(scalar); DestroyTensorPair<float>(selfDev, selfT); return 1);
  }

  int failed = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double expected = static_cast<double>(self[static_cast<size_t>(i)]) * static_cast<double>(scalarValue);
    if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
      ++failed;
    }
  }

  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr != nullptr) {
    aclrtFree(wsAddr);
  }
  aclDestroyScalar(scalar);
  DestroyTensorPair<float>(selfDev, selfT);
  return failed == 0 ? 0 : 1;
}

int RunExpectedFailureCases(aclrtStream stream)
{
  (void)stream;
  int failed = 0;

  std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape2x2 = {2, 2};
  std::vector<int64_t> shape1x4 = {1, 4};

  void *aDev = nullptr, *bDev = nullptr, *oDev = nullptr;
  aclTensor *aT = nullptr, *bT = nullptr, *oT = nullptr;

  int ret = CreateAclTensor(base, shape2x2, &aDev, ACL_FLOAT, &aT);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(base, shape2x2, &bDev, ACL_FLOAT, &bT);
  CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(aDev, aT); return 1);
  ret = CreateAclTensor(base, shape2x2, &oDev, ACL_FLOAT, &oT);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<float>(aDev, aT); DestroyTensorPair<float>(bDev, bT); return 1);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;

  auto st = aclnnMulGetWorkspaceSize(nullptr, bT, oT, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] mul_null_self should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] mul_null_self\n");
  }

  st = aclnnMulGetWorkspaceSize(aT, nullptr, oT, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] mul_null_other should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] mul_null_other\n");
  }

  st = aclnnMulGetWorkspaceSize(aT, bT, nullptr, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] mul_null_out should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] mul_null_out\n");
  }

  void* badOutDev = nullptr;
  aclTensor* badOut = nullptr;
  std::vector<float> badOutData(4, 0.0f);
  ret = CreateAclTensor(badOutData, shape1x4, &badOutDev, ACL_FLOAT, &badOut);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorPair<float>(aDev, aT); DestroyTensorPair<float>(bDev, bT); DestroyTensorPair<float>(oDev, oT);
            return 1);

  st = aclnnMulGetWorkspaceSize(aT, bT, badOut, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] mul_bad_out_shape should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] mul_bad_out_shape\n");
  }

  float scalarVal = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
  st = aclnnMulsGetWorkspaceSize(aT, nullptr, oT, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] muls_null_scalar should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] muls_null_scalar\n");
  }

  st = aclnnInplaceMulsGetWorkspaceSize(nullptr, scalar, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] inplace_muls_null_self should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] inplace_muls_null_self\n");
  }

  st = aclnnInplaceMulGetWorkspaceSize(aT, nullptr, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] inplace_mul_null_other should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] inplace_mul_null_other\n");
  }

  void* badOtherDev = nullptr;
  aclTensor* badOther = nullptr;
  std::vector<float> badOtherData(4, 1.0f);
  ret = CreateAclTensor(badOtherData, shape1x4, &badOtherDev, ACL_FLOAT, &badOther);
  CHECK_RET(ret == ACL_SUCCESS,
            if (scalar != nullptr) { aclDestroyScalar(scalar); }
            DestroyTensorPair<float>(badOutDev, badOut);
            DestroyTensorPair<float>(aDev, aT); DestroyTensorPair<float>(bDev, bT); DestroyTensorPair<float>(oDev, oT);
            return 1);

  st = aclnnInplaceMulGetWorkspaceSize(aT, badOther, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] inplace_mul_bad_shape should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] inplace_mul_bad_shape\n");
  }

  std::vector<uint16_t> u16 = {1, 2, 3, 4};
  void* u16Dev = nullptr;
  aclTensor* u16T = nullptr;
  ret = CreateAclTensor(u16, shape2x2, &u16Dev, ACL_UINT16, &u16T);
  CHECK_RET(ret == ACL_SUCCESS,
            if (scalar != nullptr) { aclDestroyScalar(scalar); }
            DestroyTensorPair<float>(badOtherDev, badOther); DestroyTensorPair<float>(badOutDev, badOut);
            DestroyTensorPair<float>(aDev, aT); DestroyTensorPair<float>(bDev, bT); DestroyTensorPair<float>(oDev, oT);
            return 1);

  st = aclnnMulGetWorkspaceSize(u16T, u16T, u16T, &wsSize, &executor);
  if (st == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] mul_unsupported_dtype_uint16 should fail\n");
    ++failed;
  } else {
    LOG_PRINT("[PASS] mul_unsupported_dtype_uint16\n");
  }

  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
  }
  DestroyTensorPair<uint16_t>(u16Dev, u16T);
  DestroyTensorPair<float>(badOtherDev, badOther);
  DestroyTensorPair<float>(badOutDev, badOut);
  DestroyTensorPair<float>(aDev, aT);
  DestroyTensorPair<float>(bDev, bT);
  DestroyTensorPair<float>(oDev, oT);

  return failed;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int totalFailed = 0;

  totalFailed += RunMulFloatCase("mul_float_basic", {0, 1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 2, 2, 2, 3, 3},
                                 {4, 2}, {4, 2}, {4, 2}, 1e-5, 1e-5, stream);

  totalFailed += RunMulFloatCase("mul_float_neg_zero", {-1.0f, 0.0f, 3.5f, -2.0f}, {2.0f, 5.0f, -1.0f, 0.0f},
                                 {2, 2}, {2, 2}, {2, 2}, 1e-5, 1e-5, stream);

  totalFailed += RunMulFloatCase("mul_float_special", {std::numeric_limits<float>::infinity(),
                                                         -std::numeric_limits<float>::infinity(),
                                                         std::numeric_limits<float>::quiet_NaN(), 1e20f},
                                 {2.0f, 2.0f, 2.0f, 1e20f}, {2, 2}, {2, 2}, {2, 2}, 1e-5, 1e-5, stream);

  totalFailed += RunMulFloatCase("mul_float_broadcast", {1, 2, 3, 4, 5, 6}, {1, 2, 3},
                                 {2, 3}, {3}, {2, 3}, 1e-5, 1e-5, stream);

  totalFailed += RunMulFloatCase("mul_float_empty", {}, {}, {0}, {0}, {0}, 1e-5, 1e-5, stream);

  totalFailed += RunMulIntCaseI32("mul_int32", {-1, 0, 10, 100}, {2, 5, -1, 3}, {2, 2}, stream);

  totalFailed += RunMulIntegralCase<int8_t>("mul_int8", {-1, 0, 10, 20}, {2, 5, -1, 3}, {2, 2}, ACL_INT8, stream);

  totalFailed += RunMulIntegralCase<uint8_t>("mul_uint8", {1, 2, 3, 4}, {2, 3, 4, 5}, {2, 2}, ACL_UINT8, stream);

  totalFailed += RunMulIntegralCase<int16_t>("mul_int16", {-2, 3, -4, 5}, {6, -7, 8, -9}, {2, 2}, ACL_INT16, stream);

  totalFailed += RunMulIntegralCase<int64_t>("mul_int64", {-2, 3, -4, 5}, {6, -7, 8, -9}, {2, 2}, ACL_INT64, stream);

  totalFailed += RunMulDoubleCase("mul_double", {1.25, -2.5, 3.0, -4.0}, {2.0, 3.0, -4.0, -5.0}, {2, 2}, stream);

  totalFailed += RunMulBoolCase("mul_bool", {1, 0, 1, 0}, {1, 1, 0, 0}, {2, 2}, stream);

  totalFailed += RunMulRaw16Case("mul_fp16", {15360, 16384, 16896, 17408}, {16384, 16384, 16384, 16384},
                                 {16384, 17408, 17920, 18432}, {2, 2}, ACL_FLOAT16, stream);

  totalFailed += RunMulRaw16Case("mul_bf16", {16256, 16384, 16448, 16512}, {16384, 16384, 16384, 16384},
                                 {16384, 16512, 16576, 16640}, {2, 2}, ACL_BF16, stream);

  totalFailed += RunMulMix16ToFloatCase("mul_mix_fp16_float", {15360, 16384, 16896, 17408},
                                        {2.0f, 2.0f, 2.0f, 2.0f}, {2, 2}, ACL_FLOAT16, true, stream);

  totalFailed += RunMulMix16ToFloatCase("mul_mix_bf16_float", {16256, 16384, 16448, 16512},
                                        {2.0f, 2.0f, 2.0f, 2.0f}, {2, 2}, ACL_BF16, false, stream);

  totalFailed += RunMulsFloatCase("muls_float", {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 2.5f, stream);

  totalFailed += RunInplaceMulFloatCase("inplace_mul_float", {2, 3, 4, 5}, {1, 2, 3, 4}, {2, 2}, stream);

  totalFailed += RunInplaceMulsFloatCase("inplace_muls_float", {2, 3, 4, 5}, {2, 2}, 2.0f, stream);

  totalFailed += RunExpectedFailureCases(stream);

  LOG_PRINT("\n=== Summary: %d failed ===\n", totalFailed);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return totalFailed;
}
