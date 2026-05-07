/**
 * Add-family examples test.
 *
 * Principles used in this version:
 * 1. Learn mul's reusable runner style only where it actually reduces duplication.
 * 2. Keep all existing add test ideas/cases; do not delete coverage-oriented samples.
 * 3. Only keep 3 generic runners with clear reuse value:
 *      - RunAddTypedTest
 *      - RunAddsTypedTest
 *      - RunAddV3TypedTest
 * 4. Other paths (inplace, empty/probe/failure/platform-sensitive) stay as dedicated helpers.
 * 5. Generic runners support expectSuccess to cover both normal and expected-failure flows.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int Finalize(int32_t deviceId, aclrtStream stream) {
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}

float bf16_to_fp32(uint16_t h) {
  union { float f; uint32_t i; } u;
  u.i = (uint32_t)h << 16;
  return u.f;
}

template <typename T>
bool AlmostEqual(T a, T b, double atol = 1e-5, double rtol = 1e-5) {
  if constexpr (std::is_same_v<T, bool> || std::is_integral_v<T>) {
    return a == b;
  } else {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
  }
}

template <typename T>
void LogMismatch(size_t i, const T& res, const T& exp) {
  LOG_PRINT("mismatch at index %zu, result=%f expected=%f\n", i, (double)res, (double)exp);
}

template <typename T>
void LogMismatch(size_t i, const std::complex<T>& res, const std::complex<T>& exp) {
  LOG_PRINT("mismatch at index %zu, result=(%f, %f) expected=(%f, %f)\n",
            i, (double)res.real(), (double)res.imag(),
            (double)exp.real(), (double)exp.imag());
}

template <typename TOut, typename TExp>
bool CheckResultTyped(const std::vector<TOut>& resultData,
                      const std::vector<TExp>& expectedData,
                      double atol = 1e-5,
                      double rtol = 1e-5) {
  if (resultData.size() != expectedData.size()) {
    LOG_PRINT("result size mismatch, result=%zu expected=%zu\n", resultData.size(), expectedData.size());
    return false;
  }
  for (size_t i = 0; i < resultData.size(); ++i) {
    if (!AlmostEqual(resultData[i], expectedData[i], atol, rtol)) {
      LogMismatch(i, resultData[i], expectedData[i]);
      return false;
    }
  }
  return true;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape,
                    void** deviceAddr,
                    aclDataType dataType,
                    aclTensor** tensor) {
  auto elemCount = GetShapeSize(shape);
  auto size = elemCount * (int64_t)sizeof(T);
  auto strides = MakeStrides(shape);

  if (size == 0) {
    *deviceAddr = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), nullptr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed for empty tensor.\n"); return -1);
    return 0;
  }

  auto ret = aclrtMalloc(deviceAddr, (size_t)size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, (size_t)size, hostData.data(), (size_t)size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
  return 0;
}

template <typename T>
int CreateAclTensorWithFormat(const std::vector<T>& hostData,
                              const std::vector<int64_t>& viewShape,
                              const std::vector<int64_t>& storageShape,
                              void** deviceAddr,
                              aclDataType dataType,
                              aclFormat format,
                              aclTensor** tensor) {
  auto elemCount = GetShapeSize(storageShape);
  auto size = elemCount * (int64_t)sizeof(T);
  auto strides = MakeStrides(viewShape);

  if (size == 0) {
    *deviceAddr = nullptr;
    *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType,
                              strides.data(), 0, format,
                              storageShape.data(), storageShape.size(), nullptr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed for empty tensor.\n"); return -1);
    return 0;
  }

  auto ret = aclrtMalloc(deviceAddr, (size_t)size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, (size_t)size, hostData.data(), (size_t)size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType,
                            strides.data(), 0, format,
                            storageShape.data(), storageShape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
  return 0;
}

int AllocateWorkspace(uint64_t workspaceSize, void** workspaceAddr) {
  *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    auto ret = aclrtMalloc(workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  return 0;
}

// ---------------- 1) Valuable generic runners ----------------

template <typename TSelf, typename TOther, typename TOut, typename TAlpha, typename TExp>
bool RunAddTypedTest(const char* testName,
                     aclrtStream stream,
                     const std::vector<TSelf>& selfHostData,
                     const std::vector<TOther>& otherHostData,
                     const std::vector<int64_t>& selfShape,
                     const std::vector<int64_t>& otherShape,
                     const std::vector<int64_t>& outShape,
                     aclDataType selfType,
                     aclDataType otherType,
                     aclDataType outType,
                     TAlpha alphaValue,
                     aclDataType alphaType,
                     const std::vector<TExp>& expectedData,
                     bool expectSuccess,
                     double atol = 1e-5,
                     double rtol = 1e-5) {
  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* alpha = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(GetShapeSize(outShape), static_cast<TOut>(0));
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, alphaType);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");
    goto cleanup;
  }
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  ret = AllocateWorkspace(workspaceSize, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, pass = false; goto cleanup);

  ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnAdd failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  {
    std::vector<TOut> resultData(GetShapeSize(outShape), static_cast<TOut>(0));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(TOut),
                      outDeviceAddr, resultData.size() * sizeof(TOut),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret);
              pass = false; goto cleanup);

    pass = CheckResultTyped(resultData, expectedData, atol, rtol);
    LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");
  }

cleanup:
  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
  return pass;
}

template <typename TSelf, typename TOtherScalar, typename TAlpha, typename TOut, typename TExp>
bool RunAddsTypedTest(const char* testName,
                      aclrtStream stream,
                      const std::vector<TSelf>& selfHostData,
                      const std::vector<int64_t>& shape,
                      TOtherScalar otherValue,
                      aclDataType otherType,
                      TAlpha alphaValue,
                      aclDataType alphaType,
                      aclDataType selfType,
                      aclDataType outType,
                      const std::vector<TExp>& expectedData,
                      bool expectSuccess,
                      double atol = 1e-5,
                      double rtol = 1e-5) {
  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* other = nullptr;
  aclScalar* alpha = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(GetShapeSize(shape), static_cast<TOut>(0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  other = aclCreateScalar(&otherValue, otherType);
  alpha = aclCreateScalar(&alphaValue, alphaType);
  CHECK_RET(other != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);

  ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");
    goto cleanup;
  }
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  ret = AllocateWorkspace(workspaceSize, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, pass = false; goto cleanup);

  ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnAdds failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  {
    std::vector<TOut> resultData(GetShapeSize(shape), static_cast<TOut>(0));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(TOut),
                      outDeviceAddr, resultData.size() * sizeof(TOut),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret);
              pass = false; goto cleanup);

    pass = CheckResultTyped(resultData, expectedData, atol, rtol);
    LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");
  }

cleanup:
  if (alpha) aclDestroyScalar(alpha);
  if (other) aclDestroyScalar(other);
  if (self) aclDestroyTensor(self);
  if (out) aclDestroyTensor(out);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
  return pass;
}

template <typename TSelfScalar, typename TOther, typename TOut, typename TAlpha, typename TExp>
bool RunAddV3TypedTest(const char* testName,
                       aclrtStream stream,
                       TSelfScalar selfValue,
                       aclDataType selfType,
                       const std::vector<TOther>& otherHostData,
                       const std::vector<int64_t>& shape,
                       aclDataType otherType,
                       TAlpha alphaValue,
                       aclDataType alphaType,
                       aclDataType outType,
                       const std::vector<TExp>& expectedData,
                       bool expectSuccess,
                       double atol = 1e-5,
                       double rtol = 1e-5) {
  int ret = 0;
  bool pass = true;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* self = nullptr;
  aclScalar* alpha = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(GetShapeSize(shape), static_cast<TOut>(0));
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  self = aclCreateScalar(&selfValue, selfType);
  alpha = aclCreateScalar(&alphaValue, alphaType);
  CHECK_RET(self != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);

  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");
    goto cleanup;
  }
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  ret = AllocateWorkspace(workspaceSize, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, pass = false; goto cleanup);

  ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnAddV3 failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
            pass = false; goto cleanup);

  {
    std::vector<TOut> resultData(GetShapeSize(shape), static_cast<TOut>(0));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(TOut),
                      outDeviceAddr, resultData.size() * sizeof(TOut),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret);
              pass = false; goto cleanup);

    pass = CheckResultTyped(resultData, expectedData, atol, rtol);
    LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");
  }

cleanup:
  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyScalar(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
  return pass;
}

template <typename TSelf, typename TOther, typename TAlpha, typename TExp>
bool RunInplaceAddTypedTest(const char* testName,
                            aclrtStream stream,
                            const std::vector<TSelf>& selfHostData,
                            const std::vector<TOther>& otherHostData,
                            const std::vector<int64_t>& shape,
                            aclDataType selfType,
                            aclDataType otherType,
                            TAlpha alphaValue,
                            aclDataType alphaType,
                            const std::vector<TExp>& expectedData,
                            bool expectSuccess,
                            double atol = 1e-5,
                            double rtol = 1e-5) {
  int ret = 0; bool pass = true; void* selfDeviceAddr = nullptr; void* otherDeviceAddr = nullptr; void* workspaceAddr = nullptr;
  aclTensor* self = nullptr; aclTensor* other = nullptr; aclScalar* alpha = nullptr; uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, alphaType);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);
  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  if (!expectSuccess) { pass = (ret != ACL_SUCCESS); LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL"); goto cleanup; }
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  ret = AllocateWorkspace(workspaceSize, &workspaceAddr); CHECK_RET(ret == ACL_SUCCESS, pass = false; goto cleanup);
  ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream); CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceAdd failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  ret = aclrtSynchronizeStream(stream); CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  { std::vector<TSelf> resultData(GetShapeSize(shape), static_cast<TSelf>(0));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(TSelf), selfDeviceAddr, resultData.size() * sizeof(TSelf), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: copy result failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
    pass = CheckResultTyped(resultData, expectedData, atol, rtol); LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL"); }
cleanup:
  if (alpha) aclDestroyScalar(alpha); if (self) aclDestroyTensor(self); if (other) aclDestroyTensor(other);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr); if (otherDeviceAddr) aclrtFree(otherDeviceAddr); if (workspaceAddr) aclrtFree(workspaceAddr); return pass; }

template <typename TSelf, typename TOtherScalar, typename TAlpha, typename TExp>
bool RunInplaceAddsTypedTest(const char* testName,
                             aclrtStream stream,
                             const std::vector<TSelf>& selfHostData,
                             const std::vector<int64_t>& shape,
                             TOtherScalar otherValue,
                             aclDataType otherType,
                             TAlpha alphaValue,
                             aclDataType alphaType,
                             aclDataType selfType,
                             const std::vector<TExp>& expectedData,
                             bool expectSuccess,
                             double atol = 1e-5,
                             double rtol = 1e-5) {
  int ret = 0; bool pass = true; void* selfDeviceAddr = nullptr; void* workspaceAddr = nullptr;
  aclTensor* self = nullptr; aclScalar* other = nullptr; aclScalar* alpha = nullptr; uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  other = aclCreateScalar(&otherValue, otherType); alpha = aclCreateScalar(&alphaValue, alphaType);
  CHECK_RET(other != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);
  ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  if (!expectSuccess) { pass = (ret != ACL_SUCCESS); LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL"); goto cleanup; }
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  ret = AllocateWorkspace(workspaceSize, &workspaceAddr); CHECK_RET(ret == ACL_SUCCESS, pass = false; goto cleanup);
  ret = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream); CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceAdds failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  ret = aclrtSynchronizeStream(stream); CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  { std::vector<TSelf> resultData(GetShapeSize(shape), static_cast<TSelf>(0));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(TSelf), selfDeviceAddr, resultData.size() * sizeof(TSelf), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: copy result failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
    pass = CheckResultTyped(resultData, expectedData, atol, rtol); LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL"); }
cleanup:
  if (alpha) aclDestroyScalar(alpha); if (other) aclDestroyScalar(other); if (self) aclDestroyTensor(self);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr); if (workspaceAddr) aclrtFree(workspaceAddr); return pass; }

template <typename TSelfScalar, typename TOther, typename TAlpha, typename TExp>
bool RunInplaceAddV3TypedTest(const char* testName,
                              aclrtStream stream,
                              TSelfScalar selfValue,
                              aclDataType selfType,
                              const std::vector<TOther>& otherHostData,
                              const std::vector<int64_t>& shape,
                              aclDataType otherType,
                              TAlpha alphaValue,
                              aclDataType alphaType,
                              const std::vector<TExp>& expectedData,
                              bool expectSuccess,
                              double atol = 1e-5,
                              double rtol = 1e-5) {
  int ret = 0; bool pass = true; void* otherDeviceAddr = nullptr; void* workspaceAddr = nullptr;
  aclTensor* other = nullptr; aclScalar* self = nullptr; aclScalar* alpha = nullptr; uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  self = aclCreateScalar(&selfValue, selfType); alpha = aclCreateScalar(&alphaValue, alphaType);
  CHECK_RET(self != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);
  ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  if (!expectSuccess) { pass = (ret != ACL_SUCCESS); LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL"); goto cleanup; }
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceAddV3GetWorkspaceSize failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  ret = AllocateWorkspace(workspaceSize, &workspaceAddr); CHECK_RET(ret == ACL_SUCCESS, pass = false; goto cleanup);
  ret = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream); CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceAddV3 failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  ret = aclrtSynchronizeStream(stream); CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
  { std::vector<TOther> resultData(GetShapeSize(shape), static_cast<TOther>(0));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(TOther), otherDeviceAddr, resultData.size() * sizeof(TOther), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: copy result failed. ERROR: %d\n", testName, ret); pass = false; goto cleanup);
    pass = CheckResultTyped(resultData, expectedData, atol, rtol); LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL"); }
cleanup:
  if (alpha) aclDestroyScalar(alpha); if (self) aclDestroyScalar(self); if (other) aclDestroyTensor(other);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr); if (workspaceAddr) aclrtFree(workspaceAddr); return pass; }

// ---------------- 2) Dedicated helpers kept for non-reusable cases ----------------

bool RunAddEmptyTensorProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 999;
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> empty;
  float alphaValue = 1.0f;

  ret = CreateAclTensor(empty, shape, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(empty, shape, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(empty, shape, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret == ACL_SUCCESS && workspaceSize == 0);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  return pass;
}

bool RunAddInvalidBroadcastProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {1, 2, 3, 4};
  std::vector<float> outHostData(6, 0);
  float alphaValue = 1.0f;

  ret = CreateAclTensor(selfHostData, {2, 3}, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, {2, 2}, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, {2, 3}, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  return pass;
}

bool RunAddInvalidOutShapeProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {10, 20, 30, 40, 50, 60};
  std::vector<float> outHostData(6, 0);
  float alphaValue = 1.0f;

  ret = CreateAclTensor(selfHostData, {2, 3}, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, {2, 3}, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, {3, 2}, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  return pass;
}

bool RunAddNullParamProbe(const char* testName, bool nullSelf, bool nullOther, bool nullAlpha, bool nullOut) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> hostData = {1, 2, 3, 4};
  float alphaValue = 1.0f;

  if (!nullSelf) {
    ret = CreateAclTensor(hostData, {2, 2}, &selfDeviceAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  }
  if (!nullOther) {
    ret = CreateAclTensor(hostData, {2, 2}, &otherDeviceAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  }
  if (!nullOut) {
    ret = CreateAclTensor(hostData, {2, 2}, &outDeviceAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  }
  if (!nullAlpha) {
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);
  }

  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  return pass;
}

bool RunAddNonNdFormatProbe(const char* testName, aclrtStream stream) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;

  std::vector<int64_t> shape = {1, 1, 2, 2};
  std::vector<float> selfHostData = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> otherHostData = {10.f, 20.f, 30.f, 40.f};
  std::vector<float> outHostData(4, 0.f);
  float alphaValue = 1.0f;

  ret = CreateAclTensorWithFormat(selfHostData, shape, shape, &selfDeviceAddr, ACL_FLOAT, ACL_FORMAT_NCHW, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensorWithFormat(otherHostData, shape, shape, &otherDeviceAddr, ACL_FLOAT, ACL_FORMAT_NCHW, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);
  ret = AllocateWorkspace(workspaceSize, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, return false);
  ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnAdd failed. ERROR: %d\n", testName, ret); return false);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  LOG_PRINT("[%s] [PASS]\n", testName);

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
  return true;
}

bool RunAddsEmptyTensorProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* other = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 999;
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> empty;
  float otherValue = 1.0f;
  float alphaValue = 1.0f;

  ret = CreateAclTensor(empty, shape, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(empty, shape, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  other = aclCreateScalar(&otherValue, ACL_FLOAT);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(other != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);

  ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret == ACL_SUCCESS && workspaceSize == 0);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (other) aclDestroyScalar(other);
  if (self) aclDestroyTensor(self);
  if (out) aclDestroyTensor(out);
  return pass;
}

bool RunAddsNonNdFormatProbe(const char* testName, aclrtStream stream) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* other = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;

  std::vector<int64_t> shape = {1, 1, 2, 2};
  std::vector<float> selfHostData = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> outHostData(4, 0.f);
  float otherValue = 1.0f;
  float alphaValue = 1.0f;

  ret = CreateAclTensorWithFormat(selfHostData, shape, shape, &selfDeviceAddr, ACL_FLOAT, ACL_FORMAT_NCHW, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  other = aclCreateScalar(&otherValue, ACL_FLOAT);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(other != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);

  ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);
  ret = AllocateWorkspace(workspaceSize, &workspaceAddr);
  CHECK_RET(ret == ACL_SUCCESS, return false);
  ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnAdds failed. ERROR: %d\n", testName, ret); return false);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  LOG_PRINT("[%s] [PASS]\n", testName);

  if (alpha) aclDestroyScalar(alpha);
  if (other) aclDestroyScalar(other);
  if (self) aclDestroyTensor(self);
  if (out) aclDestroyTensor(out);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
  return true;
}

bool RunAddsNonNdDoubleScalarPromoteCase(const char* testName, aclrtStream stream) {
  return RunAddsTypedTest<int64_t, double, double, float, float>(
      testName, stream,
      {1, 2, 3, 4}, {2, 2},
      2.0, ACL_DOUBLE,
      2.0, ACL_DOUBLE,
      ACL_INT64, ACL_FLOAT,
      {5.f, 6.f, 7.f, 8.f}, true, 1e-5, 1e-5);
}

bool RunAddsBoolSpecialCastCase(const char* testName, aclrtStream stream) {
  return RunAddsTypedTest<uint8_t, bool, bool, uint16_t, uint16_t>(
      testName, stream,
      {1, 0, 1, 0}, {2, 2},
      true, ACL_BOOL,
      true, ACL_BOOL,
      ACL_BOOL, ACL_FLOAT16,
      {0x3c00, 0x3c00, 0x3c00, 0x3c00}, true, 0.0, 0.0);
}

bool RunInplaceAddCaseFloat(const char* testName, aclrtStream stream) {
  return RunInplaceAddTypedTest<float, float, float, float>(
      testName, stream,
      {1.f, 2.f, 3.f, 4.f}, {10.f, 20.f, 30.f, 40.f},
      {2, 2}, ACL_FLOAT, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {11.f, 22.f, 33.f, 44.f}, true);
}

bool RunInplaceAddCaseDouble(const char* testName, aclrtStream stream) {
  return RunInplaceAddTypedTest<double, double, double, double>(
      testName, stream,
      {1.0, 2.0, 3.0, 4.0}, {10.0, 20.0, 30.0, 40.0},
      {2, 2}, ACL_DOUBLE, ACL_DOUBLE,
      1.0, ACL_DOUBLE,
      {11.0, 22.0, 33.0, 44.0}, true, 1e-9, 1e-9);
}

bool RunInplaceAddCaseInt8(const char* testName, aclrtStream stream) {
  return RunInplaceAddTypedTest<int8_t, int8_t, int8_t, int8_t>(
      testName, stream,
      {1, 2, 3, 4}, {10, 20, 30, 40},
      {2, 2}, ACL_INT8, ACL_INT8,
      1, ACL_INT8,
      {11, 22, 33, 44}, true);
}

bool RunInplaceAddInvalidShapeProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {1, 2, 3};
  std::vector<float> otherHostData = {10, 20, 30, 40, 50, 60};
  float alphaValue = 1.0f;

  ret = CreateAclTensor(selfHostData, {1, 3}, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, {2, 3}, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  return pass;
}

bool RunInplaceAddBroadcastFailProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {10, 20, 30, 40};
  float alphaValue = 1.0f;

  ret = CreateAclTensor(selfHostData, {2, 3}, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, {2, 2}, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  return pass;
}

bool RunInplaceAddOtherShapeMismatchProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {10, 20, 30};
  float alphaValue = 1.0f;

  ret = CreateAclTensor(selfHostData, {2, 3}, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, {1, 3}, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  return pass;
}

bool RunInplaceAddMixedInvalidProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<uint16_t> otherHostData = {0x3c00, 0x4000, 0x4200, 0x4400};
  float alphaValue = 1.0f;

  ret = CreateAclTensor(selfHostData, {2, 2}, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, {2, 2}, &otherDeviceAddr, ACL_FLOAT16, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  return pass;
}

bool RunInplaceAddMixedBf16InvalidProbe(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<uint16_t> otherHostData = {0x4120, 0x41a0, 0x41f0, 0x4220};
  float alphaValue = 1.0f;

  ret = CreateAclTensor(selfHostData, {2, 2}, &selfDeviceAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, {2, 2}, &otherDeviceAddr, ACL_BF16, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);

  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  return pass;
}

bool RunInplaceAddsCaseFloat(const char* testName, aclrtStream stream) {
  return RunInplaceAddsTypedTest<float, float, float, float>(
      testName, stream,
      {1.f, 2.f, 3.f, 4.f}, {2, 2},
      10.0f, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT,
      {11.f, 12.f, 13.f, 14.f}, true);
}

bool RunAddV3EmptyTensorProbe(const char* testName) {
  int ret = 0;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* self = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 999;
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> empty;
  float selfValue = 1.0f;
  float alphaValue = 1.0f;

  ret = CreateAclTensor(empty, shape, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(empty, shape, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  self = aclCreateScalar(&selfValue, ACL_FLOAT);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(self != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);

  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret == ACL_SUCCESS && workspaceSize == 0);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyScalar(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  return pass;
}

bool RunAddV3InvalidShapeProbe(const char* testName) {
  int ret = 0;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* self = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> otherHostData = {10.f, 20.f, 30.f, 40.f};
  std::vector<float> outHostData(4, 0.f);
  float selfValue = 1.0f;
  float alphaValue = 1.0f;

  ret = CreateAclTensor(otherHostData, {2, 2}, &otherDeviceAddr, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, {4}, &outDeviceAddr, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  self = aclCreateScalar(&selfValue, ACL_FLOAT);
  alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  CHECK_RET(self != nullptr && alpha != nullptr, LOG_PRINT("[%s] FAIL: create scalar failed\n", testName); return false);

  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyScalar(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  return pass;
}

bool RunAddV3NullParamProbe(const char* testName, bool nullSelf, bool nullOther, bool nullAlpha, bool nullOut) {
  int ret = 0;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* self = nullptr;
  aclScalar* alpha = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> hostData = {10.f, 20.f, 30.f, 40.f};
  float selfValue = 1.0f;
  float alphaValue = 1.0f;

  if (!nullOther) {
    ret = CreateAclTensor(hostData, {2, 2}, &otherDeviceAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  }
  if (!nullOut) {
    ret = CreateAclTensor(hostData, {2, 2}, &outDeviceAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  }
  if (!nullSelf) {
    self = aclCreateScalar(&selfValue, ACL_FLOAT);
    CHECK_RET(self != nullptr, LOG_PRINT("[%s] FAIL: create self scalar failed\n", testName); return false);
  }
  if (!nullAlpha) {
    alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, LOG_PRINT("[%s] FAIL: create alpha failed\n", testName); return false);
  }

  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");

  if (alpha) aclDestroyScalar(alpha);
  if (self) aclDestroyScalar(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  return pass;
}

bool RunAddV3FallbackCaseInt8(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<int8_t, int8_t, int8_t, int8_t, int8_t>(
      testName, stream,
      1, ACL_INT8,
      {10, 20, 30, 40}, {2, 2}, ACL_INT8,
      2, ACL_INT8,
      ACL_INT8,
      {21, 41, 61, 81},
      true, 0.0, 0.0);
}

bool RunAddV3FallbackCaseInt8Line200(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<int8_t, int8_t, int8_t, int8_t, int8_t>(
      testName, stream,
      3, ACL_INT8,
      {2, 4, 6, 8}, {2, 2}, ACL_INT8,
      3, ACL_INT8,
      ACL_INT8,
      {9, 15, 21, 27},
      true, 0.0, 0.0);
}

bool RunAddV3AxpyCaseFloat(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<float, float, float, float, float>(
      testName, stream,
      1.0f, ACL_FLOAT,
      {10.f, 20.f, 30.f, 40.f}, {2, 2}, ACL_FLOAT,
      2.0f, ACL_FLOAT,
      ACL_FLOAT,
      {21.f, 41.f, 61.f, 81.f},
      true);
}

bool RunAddV3PromoteFromFloatingSelfCase(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<float, int32_t, float, float, float>(
      testName, stream,
      1.5f, ACL_FLOAT,
      {10, 20, 30, 40}, {2, 2}, ACL_INT32,
      1.0f, ACL_FLOAT,
      ACL_FLOAT,
      {11.5f, 21.5f, 31.5f, 41.5f},
      true, 1e-5, 1e-5);
}

bool RunAddV3ComplexSelfScalarCase(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<std::complex<float>, float, std::complex<float>, std::complex<float>, std::complex<float>>(
      testName, stream,
      std::complex<float>(1.0f, 2.0f), ACL_COMPLEX64,
      {10.f, 20.f, 30.f, 40.f}, {2, 2}, ACL_FLOAT,
      std::complex<float>(1.0f, 0.0f), ACL_COMPLEX64,
      ACL_COMPLEX64,
      {
          std::complex<float>(11.0f, 2.0f),
          std::complex<float>(21.0f, 2.0f),
          std::complex<float>(31.0f, 2.0f),
          std::complex<float>(41.0f, 2.0f),
      },
      true, 1e-4, 1e-4);
}

bool RunAddV3DoubleSelfOutFloatCase(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<double, int32_t, float, float, float>(
      testName, stream,
      1.5, ACL_DOUBLE,
      {10, 20, 30, 40}, {2, 2}, ACL_INT32,
      1.0f, ACL_FLOAT,
      ACL_FLOAT,
      {11.5f, 21.5f, 31.5f, 41.5f},
      true, 1e-5, 1e-5);
}

bool RunAddV3OtherBf16FloatingCase(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<int32_t, uint16_t, float, float, float>(
      testName, stream,
      1, ACL_INT32,
      {0x4120, 0x41a0, 0x41f0, 0x4220}, {2, 2}, ACL_BF16,
      1.0f, ACL_FLOAT,
      ACL_FLOAT,
      {11.f, 21.f, 31.f, 41.f},
      true, 1e-3, 1e-3);
}

bool RunAddV3OtherFloat16FloatingCase(const char* testName, aclrtStream stream) {
  return RunAddV3TypedTest<int32_t, uint16_t, float, float, float>(
      testName, stream,
      1, ACL_INT32,
      {0x4900, 0x4d00, 0x4f80, 0x5100}, {2, 2}, ACL_FLOAT16,
      1.0f, ACL_FLOAT,
      ACL_FLOAT,
      {11.f, 21.f, 31.f, 41.f},
      true, 1e-3, 1e-3);
}

bool RunInplaceAddV3CaseFloat(const char* testName, aclrtStream stream) {
  return RunInplaceAddV3TypedTest<float, float, float, float>(
      testName, stream,
      1.0f, ACL_FLOAT,
      {10.f, 20.f, 30.f, 40.f}, {2, 2}, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {11.f, 21.f, 31.f, 41.f}, true);
}


int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  bool allPass = true;
  int totalCases = 0;
  int failedCases = 0;
  auto RecordCase = [&](bool pass) { ++totalCases; if (!pass) ++failedCases; allPass = allPass && pass; };
  std::vector<int64_t> shape = {2, 2};

  RecordCase(RunAddTypedTest<float, float, float, float, float>(
      "CASE_01_add_same_dtype_alpha1_direct", stream,
      {1.f, 2.f, 3.f, 4.f}, {10.f, 20.f, 30.f, 40.f},
      shape, shape, shape,
      ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {11.f, 22.f, 33.f, 44.f}, true));

  RecordCase(RunAddTypedTest<double, double, double, double, double>(
      "CASE_02_add_same_dtype_alpha1_contiguous_else_double", stream,
      {1.0, 2.0, 3.0, 4.0}, {10.0, 20.0, 30.0, 40.0},
      shape, shape, shape,
      ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE,
      1.0, ACL_DOUBLE,
      {11.0, 22.0, 33.0, 44.0}, true, 1e-9, 1e-9));

  RecordCase(RunAddTypedTest<uint16_t, float, float, float, float>(
      "CASE_03_add_mixed_dtype_alpha1_fastpath", stream,
      {0x3c00, 0x4000, 0x4200, 0x4400}, {10.f, 20.f, 30.f, 40.f},
      shape, shape, shape,
      ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {11.f, 22.f, 33.f, 44.f}, true, 1e-4, 1e-4));

  RecordCase(RunAddTypedTest<float, uint16_t, float, float, float>(
      "CASE_Add_Mixed_Float_BF16", stream,
      {1.f, 2.f, 3.f, 4.f}, {0x4120, 0x41a0, 0x41f0, 0x4220},
      shape, shape, shape,
      ACL_FLOAT, ACL_BF16, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {11.f, 22.f, 33.f, 44.f}, true, 1e-3, 1e-3));

  RecordCase(RunAddTypedTest<uint16_t, float, float, float, float>(
      "CASE_Add_Mixed_BF16_Float", stream,
      {0x3f80, 0x4000, 0x4040, 0x4080}, {10.f, 20.f, 30.f, 40.f},
      shape, shape, shape,
      ACL_BF16, ACL_FLOAT, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {11.f, 22.f, 33.f, 44.f}, true, 1e-3, 1e-3));

  RecordCase(RunAddTypedTest<float, uint16_t, float, float, float>(
      "CASE_Add_Mixed_Float_F16", stream,
      {1.f, 2.f, 3.f, 4.f}, {0x4900, 0x4d00, 0x4f80, 0x5100},
      shape, shape, shape,
      ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {11.f, 22.f, 33.f, 44.f}, true, 1e-3, 1e-3));

  RecordCase(RunAddTypedTest<int16_t, int32_t, int32_t, int32_t, int32_t>(
      "CASE_04_add_promote_else_alpha1", stream,
      {1, 2, 3, 4}, {10, 20, 30, 40},
      shape, shape, shape,
      ACL_INT16, ACL_INT32, ACL_INT32,
      1, ACL_INT32,
      {11, 22, 33, 44}, true));

  RecordCase(RunAddTypedTest<int32_t, int32_t, int32_t, int32_t, int32_t>(
      "CASE_Add_Tiling_Int32_Alpha1", stream,
      {1, 2, 3, 4}, {10, 20, 30, 40},
      shape, shape, shape,
      ACL_INT32, ACL_INT32, ACL_INT32,
      1, ACL_INT32,
      {11, 22, 33, 44}, true));

  RecordCase(RunAddTypedTest<int64_t, int64_t, int64_t, int64_t, int64_t>(
      "CASE_Add_Tiling_Int64_Alpha1", stream,
      {1, 2, 3, 4}, {10, 20, 30, 40},
      shape, shape, shape,
      ACL_INT64, ACL_INT64, ACL_INT64,
      1, ACL_INT64,
      {11, 22, 33, 44}, true));

  RecordCase(RunAddTypedTest<uint8_t, uint8_t, uint8_t, uint8_t, uint8_t>(
      "CASE_Add_Tiling_Uint8_Alpha1", stream,
      {1, 2, 3, 4}, {10, 20, 30, 40},
      shape, shape, shape,
      ACL_UINT8, ACL_UINT8, ACL_UINT8,
      static_cast<uint8_t>(1), ACL_UINT8,
      {11, 22, 33, 44}, true));

  RecordCase(RunAddTypedTest<float, float, float, float, float>(
      "CASE_05_add_alpha_ne_one_axpy_or_fallback", stream,
      {1.f, 2.f, 3.f, 4.f}, {10.f, 20.f, 30.f, 40.f},
      shape, shape, shape,
      ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
      2.0f, ACL_FLOAT,
      {21.f, 42.f, 63.f, 84.f}, true));

  RecordCase(RunAddTypedTest<int32_t, int32_t, int32_t, int32_t, int32_t>(
      "CASE_Add_AxpyV2_Int32", stream,
      {1, 2, 3, 4}, {10, 20, 30, 40},
      shape, shape, shape,
      ACL_INT32, ACL_INT32, ACL_INT32,
      2, ACL_INT32,
      {21, 42, 63, 84}, true));

  RecordCase(RunAddTypedTest<double, double, double, double, double>(
      "CASE_Add_FinalElse_Double", stream,
      {1.0, 2.0, 3.0, 4.0}, {10.0, 20.0, 30.0, 40.0},
      shape, shape, shape,
      ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE,
      2.0, ACL_DOUBLE,
      {21.0, 42.0, 63.0, 84.0}, true, 1e-9, 1e-9));

  RecordCase(RunAddTypedTest<float, float, float, float, float>(
      "CASE_Add_NegativeValues", stream,
      {-1.f, -2.f, 3.f, 4.f}, {10.f, -20.f, -30.f, 40.f},
      shape, shape, shape,
      ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {9.f, -22.f, -27.f, 44.f}, true));

  RecordCase(RunAddTypedTest<float, float, float, float, float>(
      "CASE_Add_ZeroAndSmallValues", stream,
      {0.f, 1e-6f, -1e-6f, 0.f}, {0.f, -1e-6f, 1e-6f, 0.f},
      shape, shape, shape,
      ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {0.f, 0.f, 0.f, 0.f}, true, 1e-7, 1e-7));

  RecordCase(RunAddTypedTest<int32_t, int32_t, int32_t, int32_t, int32_t>(
      "CASE_Add_Int32_NearBoundary", stream,
      {2147483000, -2147483000, 100, -100}, {100, -100, -100, 100},
      shape, shape, shape,
      ACL_INT32, ACL_INT32, ACL_INT32,
      1, ACL_INT32,
      {2147483100, -2147483100, 0, 0}, true));

  RecordCase(RunAddEmptyTensorProbe("CASE_Add_EmptyTensor_Float"));
  RecordCase(RunAddInvalidBroadcastProbe("CASE_Add_InvalidBroadcast"));
  RecordCase(RunAddInvalidOutShapeProbe("CASE_Add_InvalidOutShape"));
  RecordCase(RunAddTypedTest<uint16_t, float, uint16_t, float, uint16_t>(
      "CASE_Add_Tiling_MixedNonFloatOut_Fail", stream,
      {0x3c00, 0x4000, 0x4200, 0x4400}, {10.f, 20.f, 30.f, 40.f},
      shape, shape, shape,
      ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16,
      1.0f, ACL_FLOAT,
      {}, false));
  RecordCase(RunAddTypedTest<uint16_t, uint16_t, float, float, float>(
      "CASE_Add_Tiling_OutputMismatch_Fail", stream,
      {0x3c00, 0x4000, 0x4200, 0x4400}, {0x3c00, 0x4000, 0x4200, 0x4400},
      shape, shape, shape,
      ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      {}, false));
  RecordCase(RunAddNullParamProbe("CASE_Add_NullSelf", true, false, false, false));
  RecordCase(RunAddNullParamProbe("CASE_Add_NullOther", false, true, false, false));
  RecordCase(RunAddNullParamProbe("CASE_Add_NullAlpha", false, false, true, false));
  RecordCase(RunAddNullParamProbe("CASE_Add_NullOut", false, false, false, true));
  RecordCase(RunAddNonNdFormatProbe("CASE_Add_NonND_Format_Warning", stream));
  RecordCase(RunAddTypedTest<uint8_t, uint8_t, uint8_t, float, uint8_t>(
      "CASE_Add_CheckPromoteType_BOOL", stream,
      {0, 1, 0, 1}, {1, 0, 1, 0},
      shape, shape, shape,
      ACL_BOOL, ACL_BOOL, ACL_BOOL,
      1.0f, ACL_FLOAT,
      {}, false));
  RecordCase(RunAddTypedTest<int32_t, int32_t, int32_t, float, int32_t>(
      "CASE_Add_CheckPromoteType_AlphaCastFail", stream,
      {1, 2, 3, 4}, {10, 20, 30, 40},
      shape, shape, shape,
      ACL_INT32, ACL_INT32, ACL_INT32,
      1.0f, ACL_FLOAT,
      {}, false));
  RecordCase(RunAddTypedTest<float, float, int32_t, float, int32_t>(
      "CASE_Add_CheckPromoteType_OutCastFail", stream,
      {1.f, 2.f, 3.f, 4.f}, {10.f, 20.f, 30.f, 40.f},
      shape, shape, shape,
      ACL_FLOAT, ACL_FLOAT, ACL_INT32,
      1.0f, ACL_FLOAT,
      {}, false));
  RecordCase(RunAddTypedTest<double, double, double, double, double>(
      "CASE_Add_Tiling_UnsupportedDouble_Fail", stream,
      {1.0, 2.0, 3.0, 4.0}, {10.0, 20.0, 30.0, 40.0},
      shape, shape, shape,
      ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE,
      1.0, ACL_DOUBLE,
      {}, false));

  RecordCase(RunAddsTypedTest<float, float, float, float, float>(
      "CASE_Adds_Basic", stream,
      {1.f, 2.f, 3.f, 4.f}, shape,
      10.0f, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT, ACL_FLOAT,
      {11.f, 12.f, 13.f, 14.f}, true));

  RecordCase(RunAddsTypedTest<float, float, float, float, float>(
      "CASE_Adds_Axpy_Float_Alpha2", stream,
      {1.f, 2.f, 3.f, 4.f}, shape,
      10.0f, ACL_FLOAT,
      2.0f, ACL_FLOAT,
      ACL_FLOAT, ACL_FLOAT,
      {21.f, 22.f, 23.f, 24.f}, true));

  RecordCase(RunAddsTypedTest<float, float, float, float, float>(
      "CASE_Adds_NegativeScalar", stream,
      {-1.f, 2.f, -3.f, 4.f}, shape,
      -10.0f, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT, ACL_FLOAT,
      {-11.f, -8.f, -13.f, -6.f}, true));

  RecordCase(RunAddsTypedTest<int32_t, int32_t, int32_t, int32_t, int32_t>(
      "CASE_Adds_AxpyV2_Int32_Alpha2", stream,
      {1, 2, 3, 4}, shape,
      10, ACL_INT32,
      2, ACL_INT32,
      ACL_INT32, ACL_INT32,
      {21, 22, 23, 24}, true));

  RecordCase(RunAddsTypedTest<int16_t, int16_t, int16_t, int16_t, int16_t>(
      "CASE_Adds_FinalElse_Int16_Alpha2", stream,
      {1, 2, 3, 4}, shape,
      10, ACL_INT16,
      2, ACL_INT16,
      ACL_INT16, ACL_INT16,
      {21, 22, 23, 24}, true));

  RecordCase(RunAddsTypedTest<uint16_t, float, float, uint16_t, uint16_t>(
      "CASE_Adds_BF16", stream,
      {0x3f80, 0x4000, 0x4040, 0x4080}, shape,
      1.0f, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_BF16, ACL_BF16,
      {0, 0, 0, 0}, true));

  RecordCase(RunAddsEmptyTensorProbe("CASE_Adds_EmptyTensor"));
  RecordCase(RunAddsNonNdFormatProbe("CASE_Adds_NonND_Format_Warning", stream));
  RecordCase(RunAddsNonNdDoubleScalarPromoteCase("CASE_Adds_DoubleScalar_OutFloat", stream));
  RecordCase(RunAddsBoolSpecialCastCase("CASE_Adds_BoolSpecialCast", stream));
  RecordCase(RunAddsTypedTest<uint8_t, bool, bool, int32_t, int32_t>(
      "CASE_Adds_Bool_Guard", stream,
      {0, 1, 0, 1}, shape,
      true, ACL_BOOL,
      true, ACL_BOOL,
      ACL_BOOL, ACL_INT32,
      {}, false));

  RecordCase(RunInplaceAddCaseFloat("CASE_InplaceAdd_Float_Basic", stream));
  RecordCase(RunInplaceAddCaseDouble("CASE_InplaceAdd_Double_Basic", stream));
  RecordCase(RunInplaceAddCaseInt8("CASE_InplaceAdd_Int8_Basic", stream));
  RecordCase(RunInplaceAddInvalidShapeProbe("CASE_InplaceAdd_InvalidShape"));
  RecordCase(RunInplaceAddBroadcastFailProbe("CASE_InplaceAdd_BroadcastFail"));
  RecordCase(RunInplaceAddOtherShapeMismatchProbe("CASE_InplaceAdd_OtherShapeMismatch"));
  RecordCase(RunInplaceAddMixedInvalidProbe("CASE_InplaceAdd_MixedInvalid"));
  RecordCase(RunInplaceAddMixedBf16InvalidProbe("CASE_InplaceAdd_MixedBf16Invalid"));
  RecordCase(RunInplaceAddsCaseFloat("CASE_13_inplace_adds_basic", stream));

  RecordCase(RunAddV3TypedTest<float, float, float, float, float>(
      "CASE_14_addv3_basic", stream,
      1.0f, ACL_FLOAT,
      {10.f, 20.f, 30.f, 40.f}, shape, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT,
      {11.f, 21.f, 31.f, 41.f}, true));

  RecordCase(RunAddV3TypedTest<float, float, float, float, float>(
      "CASE_AddV3_NegativeSelf", stream,
      -1.5f, ACL_FLOAT,
      {10.f, -20.f, 30.f, -40.f}, shape, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT,
      {8.5f, -21.5f, 28.5f, -41.5f}, true));

  RecordCase(RunAddV3FallbackCaseInt8("CASE_16_addv3_alpha_ne_one_fallback", stream));
  RecordCase(RunAddV3FallbackCaseInt8Line200("CASE_AddV3_Fallback_Line200_Int8", stream));
  RecordCase(RunAddV3AxpyCaseFloat("CASE_AddV3_Axpy_Float_Alpha2", stream));
  RecordCase(RunAddV3PromoteFromFloatingSelfCase("CASE_AddV3_SelfFloat_OtherInt32_Promote", stream));
  RecordCase(RunAddV3OtherBf16FloatingCase("CASE_AddV3_OtherBF16_FloatingBranch", stream));
  RecordCase(RunAddV3OtherFloat16FloatingCase("CASE_AddV3_OtherFloat16_FloatingBranch", stream));
  RecordCase(RunAddV3ComplexSelfScalarCase("CASE_AddV3_SelfComplex64_OtherFloat", stream));
  RecordCase(RunAddV3DoubleSelfOutFloatCase("CASE_AddV3_SelfDouble_OutFloat", stream));
  RecordCase(RunAddV3EmptyTensorProbe("CASE_15_addv3_empty_tensor"));
  RecordCase(RunAddV3NullParamProbe("CASE_AddV3_NullSelf", true, false, false, false));
  RecordCase(RunAddV3NullParamProbe("CASE_AddV3_NullOther", false, true, false, false));
  RecordCase(RunAddV3NullParamProbe("CASE_AddV3_NullAlpha", false, false, true, false));
  RecordCase(RunAddV3NullParamProbe("CASE_AddV3_NullOut", false, false, false, true));
  RecordCase(RunAddV3TypedTest<int32_t, int32_t, int32_t, float, int32_t>(
      "CASE_AddV3_AlphaCastFail_Int32", stream,
      1, ACL_INT32,
      {10, 20, 30, 40}, shape, ACL_INT32,
      1.0f, ACL_FLOAT,
      ACL_INT32,
      {}, false));
  RecordCase(RunAddV3TypedTest<float, float, int32_t, float, int32_t>(
      "CASE_AddV3_OutCastFail_FloatToInt32", stream,
      1.0f, ACL_FLOAT,
      {10.f, 20.f, 30.f, 40.f}, shape, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_INT32,
      {}, false));
  RecordCase(RunAddV3TypedTest<int32_t, uint8_t, int32_t, int32_t, int32_t>(
      "CASE_AddV3_OtherBoolNotSupport", stream,
      1, ACL_INT32,
      {0, 1, 0, 1}, shape, ACL_BOOL,
      1, ACL_INT32,
      ACL_INT32,
      {}, false));
  RecordCase(RunAddV3InvalidShapeProbe("CASE_AddV3_InvalidShape"));
  RecordCase(RunInplaceAddV3CaseFloat("CASE_17_inplace_addv3_basic", stream));

  Finalize(deviceId, stream);
  LOG_PRINT("Summary: total=%d failed=%d passed=%d\n", totalCases, failedCases, totalCases - failedCases);
  LOG_PRINT("Overall result: [%s]\n", failedCases == 0 ? "PASS" : "FAIL");
  return failedCases == 0 ? 0 : 1;
}
