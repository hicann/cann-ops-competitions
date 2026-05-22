/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

bool AlmostEqual(double expected, double actual, double atol, double rtol) {
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual))
        return (expected > 0) == (actual > 0);
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

int TestMul_Float32_Basic(aclrtStream stream) {
    LOG_PRINT("[TEST] Mul_Float32_Basic\n");
    std::vector<int64_t> shape = {4, 2};
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;

    std::vector<float> x1 = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> x2 = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<float> outHost(GetShapeSize(shape), 0);

    CreateAclTensor(x1, shape, &x1Dev, ACL_FLOAT, &x1T);
    CreateAclTensor(x2, shape, &x2Dev, ACL_FLOAT, &x2T);
    CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  GetWorkspaceSize failed: %d\n", ret);
        return 1;
    }

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape);
    std::vector<float> result(n);
    aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double expected = (double)x1[i] * (double)x2[i];
        if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) {
            if (failed < 3) LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, result[i]);
            failed++;
        }
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL] %d mismatches\n", failed);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

int TestMul_Float32_NegativeZero(aclrtStream stream) {
    LOG_PRINT("[TEST] Mul_Float32_NegativeZero\n");
    std::vector<int64_t> shape = {2, 2};
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;

    std::vector<float> x1 = {-1.0f, 0.0f, 3.5f, -2.0f};
    std::vector<float> x2 = {2.0f, 5.0f, -1.0f, 0.0f};
    std::vector<float> outHost(GetShapeSize(shape), 0);

    CreateAclTensor(x1, shape, &x1Dev, ACL_FLOAT, &x1T);
    CreateAclTensor(x2, shape, &x2Dev, ACL_FLOAT, &x2T);
    CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return 1;

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape);
    std::vector<float> result(n);
    aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double expected = (double)x1[i] * (double)x2[i];
        if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL]\n");

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

int TestMul_Float32_Broadcast(aclrtStream stream) {
    LOG_PRINT("[TEST] Mul_Float32_Broadcast\n");
    std::vector<int64_t> shape1 = {2, 3};
    std::vector<int64_t> shape2 = {1, 3};
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;

    std::vector<float> x1 = {1, 2, 3, 4, 5, 6};
    std::vector<float> x2 = {2, 3, 4};
    std::vector<float> outHost(GetShapeSize(shape1), 0);

    CreateAclTensor(x1, shape1, &x1Dev, ACL_FLOAT, &x1T);
    CreateAclTensor(x2, shape2, &x2Dev, ACL_FLOAT, &x2T);
    CreateAclTensor(outHost, shape1, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return 1;

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape1);
    std::vector<float> result(n);
    aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> expected = {2, 6, 12, 8, 15, 24};
    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!AlmostEqual(expected[i], result[i], 1e-5, 1e-5)) failed++;
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL]\n");

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

int TestMul_Int32_Basic(aclrtStream stream) {
    LOG_PRINT("[TEST] Mul_Int32_Basic\n");
    std::vector<int64_t> shape = {4};
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;

    std::vector<int32_t> x1 = {-10, -1, 0, 2};
    std::vector<int32_t> x2 = {3, 5, 100, -7};
    std::vector<int32_t> outHost(GetShapeSize(shape), 0);

    CreateAclTensor(x1, shape, &x1Dev, ACL_INT32, &x1T);
    CreateAclTensor(x2, shape, &x2Dev, ACL_INT32, &x2T);
    CreateAclTensor(outHost, shape, &outDev, ACL_INT32, &outT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return 1;

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape);
    std::vector<int32_t> result(n);
    aclrtMemcpy(result.data(), n*sizeof(int32_t), outDev, n*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t expected = x1[i] * x2[i];
        if (result[i] != expected) failed++;
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL]\n");

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

int TestMuls_Float32(aclrtStream stream) {
    LOG_PRINT("[TEST] Muls_Float32\n");
    std::vector<int64_t> shape = {2, 2};
    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *selfT=nullptr, *outT=nullptr;

    std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 2.5f;
    std::vector<float> outHost(GetShapeSize(shape), 0);

    CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
    CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);

    aclScalar* aclScalar = aclCreateScalar(&scalar, ACL_FLOAT);
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(selfT, aclScalar, outT, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return 1;

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMuls(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape);
    std::vector<float> result(n);
    aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double expected = (double)self[i] * (double)scalar;
        if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL]\n");

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(aclScalar);
    aclrtFree(selfDev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

int TestMuls_Int32(aclrtStream stream) {
    LOG_PRINT("[TEST] Muls_Int32\n");
    std::vector<int64_t> shape = {3};
    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *selfT=nullptr, *outT=nullptr;

    std::vector<int32_t> self = {5, -3, 0};
    int32_t scalar = 7;
    std::vector<int32_t> outHost(GetShapeSize(shape), 0);

    CreateAclTensor(self, shape, &selfDev, ACL_INT32, &selfT);
    CreateAclTensor(outHost, shape, &outDev, ACL_INT32, &outT);

    aclScalar* aclScalar = aclCreateScalar(&scalar, ACL_INT32);
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(selfT, aclScalar, outT, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return 1;

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMuls(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape);
    std::vector<int32_t> result(n);
    aclrtMemcpy(result.data(), n*sizeof(int32_t), outDev, n*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t expected = self[i] * scalar;
        if (result[i] != expected) failed++;
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL]\n");

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(aclScalar);
    aclrtFree(selfDev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

int TestInplaceMul_Float32(aclrtStream stream) {
    LOG_PRINT("[TEST] InplaceMul_Float32\n");
    std::vector<int64_t> shape = {2, 2};
    void *selfDev=nullptr, *otherDev=nullptr;
    aclTensor *selfT=nullptr, *otherT=nullptr;

    std::vector<float> self = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> other = {3.0f, 2.0f, 1.0f, 4.0f};

    CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
    CreateAclTensor(other, shape, &otherDev, ACL_FLOAT, &otherT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceMulGetWorkspaceSize(selfT, otherT, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return 1;

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape);
    std::vector<float> result(n);
    aclrtMemcpy(result.data(), n*sizeof(float), selfDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double expected = (double)self[i] * (double)other[i];
        if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL]\n");

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyTensor(otherT);
    aclrtFree(selfDev); aclrtFree(otherDev);
    return failed > 0 ? 1 : 0;
}

int TestInplaceMuls_Float32(aclrtStream stream) {
    LOG_PRINT("[TEST] InplaceMuls_Float32\n");
    std::vector<int64_t> shape = {4};
    void *selfDev=nullptr;
    aclTensor *selfT=nullptr;

    std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 5.0f;

    CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);

    aclScalar* aclScalar = aclCreateScalar(&scalar, ACL_FLOAT);
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceMulsGetWorkspaceSize(selfT, aclScalar, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return 1;

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceMuls(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int64_t n = GetShapeSize(shape);
    std::vector<float> result(n);
    aclrtMemcpy(result.data(), n*sizeof(float), selfDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double expected = (double)self[i] * (double)scalar;
        if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
    }
    LOG_PRINT(failed == 0 ? "  [PASS]\n" : "  [FAIL]\n");

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyScalar(aclScalar);
    aclrtFree(selfDev);
    return failed > 0 ? 1 : 0;
}

int TestInvalid_Nullptr(aclrtStream stream) {
    LOG_PRINT("[TEST] Invalid_Nullptr\n");
    aclTensor* tensor = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnMulGetWorkspaceSize(nullptr, tensor, tensor, &wsSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [PASS] Correctly rejected nullptr\n");
        return 0;
    }
    LOG_PRINT("  [FAIL] Should have rejected nullptr\n");
    return 1;
}

int TestInvalid_Dtype(aclrtStream stream) {
    LOG_PRINT("[TEST] Invalid_Dtype\n");
    std::vector<int64_t> shape = {2};
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;

    std::vector<uint32_t> x1 = {1, 2};
    std::vector<uint32_t> x2 = {3, 4};
    std::vector<uint32_t> outHost(GetShapeSize(shape), 0);

    CreateAclTensor(x1, shape, &x1Dev, ACL_UINT32, &x1T);
    CreateAclTensor(x2, shape, &x2Dev, ACL_UINT32, &x2T);
    CreateAclTensor(outHost, shape, &outDev, ACL_UINT32, &outT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);

    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [PASS] Correctly rejected UINT32\n");
        return 0;
    }
    LOG_PRINT("  [FAIL] Should have rejected UINT32\n");
    return 1;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int totalFailed = 0;

    totalFailed += TestMul_Float32_Basic(stream);
    totalFailed += TestMul_Float32_NegativeZero(stream);
    totalFailed += TestMul_Float32_Broadcast(stream);
    totalFailed += TestMul_Int32_Basic(stream);
    totalFailed += TestMuls_Float32(stream);
    totalFailed += TestMuls_Int32(stream);
    totalFailed += TestInplaceMul_Float32(stream);
    totalFailed += TestInplaceMuls_Float32(stream);
    totalFailed += TestInvalid_Nullptr(stream);
    totalFailed += TestInvalid_Dtype(stream);

    LOG_PRINT("\n=== Summary: %d tests failed ===\n", totalFailed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFailed;
}
