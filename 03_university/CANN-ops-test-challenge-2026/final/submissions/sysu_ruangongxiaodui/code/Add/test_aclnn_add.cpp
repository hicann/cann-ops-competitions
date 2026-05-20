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
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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

void CleanupStream(aclrtStream stream, int32_t deviceId) {
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
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

aclScalar* CreateAclScalar(float value, aclDataType dataType) {
    return aclCreateScalar(&value, dataType);
}

bool AlmostEqual(double expected, double actual, double atol, double rtol) {
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual)) {
        return (expected > 0) == (actual > 0);
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

void Cleanup(void* x1Addr, void* x2Addr, void* outAddr, void* wsAddr,
             aclTensor* x1, aclTensor* x2, aclScalar* alpha, aclTensor* out) {
    if (x1) aclDestroyTensor(x1);
    if (x2) aclDestroyTensor(x2);
    if (alpha) aclDestroyScalar(alpha);
    if (out) aclDestroyTensor(out);
    if (x1Addr) aclrtFree(x1Addr);
    if (x2Addr) aclrtFree(x2Addr);
    if (outAddr) aclrtFree(outAddr);
    if (wsAddr) aclrtFree(wsAddr);
}

template <typename T1, typename T2, typename Tout>
int RunAddTest(const char* name, const std::vector<T1>& x1Data, const std::vector<int64_t>& x1Shape,
               const std::vector<T2>& x2Data, const std::vector<int64_t>& x2Shape,
               float alpha, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
               aclrtStream stream, double atol, double rtol, bool verbose) {
    void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
    aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;
    aclScalar* alphaScalar = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    void* wsAddr = nullptr;
    int ret;
    std::vector<int64_t> outShape;
    int64_t outSize;
    std::vector<Tout> outHost;
    std::vector<Tout> result;
    int failed;
    int64_t i;
    double expected, actual, abs_err, rel_err;

    outShape = x1Shape.size() >= x2Shape.size() ? x1Shape : x2Shape;
    outSize = GetShapeSize(outShape);

    if (CreateAclTensor(x1Data, x1Shape, &x1Dev, dt1, &x1T) != 0) goto cleanup;
    if (CreateAclTensor(x2Data, x2Shape, &x2Dev, dt2, &x2T) != 0) goto cleanup;

    outHost.resize(outSize, 0);
    if (CreateAclTensor(outHost, outShape, &outDev, dtOut, &outT) != 0) goto cleanup;

    alphaScalar = CreateAclScalar(alpha, dt2);
    CHECK_RET(alphaScalar != nullptr, goto cleanup);

    ret = aclnnAddGetWorkspaceSize(x1T, x2T, alphaScalar, outT, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnAdd(wsAddr, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Execute failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    result.resize(outSize);
    ret = aclrtMemcpy(result.data(), outSize * sizeof(Tout), outDev, outSize * sizeof(Tout), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    failed = 0;
    for (i = 0; i < outSize; i++) {
        expected = static_cast<double>(x1Data[i % x1Data.size()]) +
                  static_cast<double>(alpha) * static_cast<double>(x2Data[i % x2Data.size()]);
        actual = static_cast<double>(result[i]);

        if (!AlmostEqual(expected, actual, atol, rtol)) {
            if (verbose && failed < 5) {
                abs_err = std::fabs(actual - expected);
                rel_err = (expected != 0.0) ? abs_err / std::fabs(expected) : abs_err;
                LOG_PRINT("  Expected: [%.15e]\n", expected);
                LOG_PRINT("  Actual:   [%.15e]\n", actual);
                LOG_PRINT("  Error:    abs=%.6e, rel=%.6e\n", abs_err, rel_err);
                LOG_PRINT("  [FAIL] Precision loss detected\n");
            }
            failed++;
        }
    }

    LOG_PRINT("%s [%s] - %ld/%ld passed, alpha=%.2f\n",
              name, failed == 0 ? "PASS" : "FAIL", outSize - failed, outSize, alpha);

cleanup:
    Cleanup(x1Dev, x2Dev, outDev, wsAddr, x1T, x2T, alphaScalar, outT);
    return failed > 0 ? 1 : 0;
}

template <typename T>
int RunAddsTest(const char* name, const std::vector<T>& x1Data, const std::vector<int64_t>& x1Shape,
                T x2Scalar, float alpha, aclDataType dt, aclrtStream stream,
                double atol, double rtol) {
    void *x1Dev = nullptr, *outDev = nullptr;
    aclTensor *x1T = nullptr, *outT = nullptr;
    aclScalar* x2ScalarPtr = nullptr;
    aclScalar* alphaScalar = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    void* wsAddr = nullptr;
    int ret;
    int64_t outSize;
    std::vector<T> outHost;
    std::vector<T> result;
    int failed;
    int64_t i;
    double expected;

    outSize = GetShapeSize(x1Shape);
    outHost.resize(outSize, 0);

    if (CreateAclTensor(x1Data, x1Shape, &x1Dev, dt, &x1T) != 0) goto cleanup;
    if (CreateAclTensor(outHost, x1Shape, &outDev, dt, &outT) != 0) goto cleanup;

    x2ScalarPtr = aclCreateScalar(&x2Scalar, dt);
    alphaScalar = CreateAclScalar(alpha, dt);
    CHECK_RET(x2ScalarPtr != nullptr && alphaScalar != nullptr, goto cleanup);

    ret = aclnnAddsGetWorkspaceSize(x1T, x2ScalarPtr, alphaScalar, outT, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Adds GetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnAdds(wsAddr, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Adds Execute failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    result.resize(outSize);
    ret = aclrtMemcpy(result.data(), outSize * sizeof(T), outDev, outSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    failed = 0;
    for (i = 0; i < outSize; i++) {
        expected = static_cast<double>(x1Data[i]) +
                  static_cast<double>(alpha) * static_cast<double>(x2Scalar);
        if (!AlmostEqual(expected, static_cast<double>(result[i]), atol, rtol)) {
            failed++;
        }
    }

    LOG_PRINT("%s [%s] - %ld/%ld passed, alpha=%.2f, x2=%.2f\n",
              name, failed == 0 ? "PASS" : "FAIL", outSize - failed, outSize, alpha, x2Scalar);

cleanup:
    if (x1T) aclDestroyTensor(x1T);
    if (outT) aclDestroyTensor(outT);
    if (x2ScalarPtr) aclDestroyScalar(x2ScalarPtr);
    if (alphaScalar) aclDestroyScalar(alphaScalar);
    if (x1Dev) aclrtFree(x1Dev);
    if (outDev) aclrtFree(outDev);
    if (wsAddr) aclrtFree(wsAddr);
    return failed > 0 ? 1 : 0;
}

template <typename T>
int RunInplaceAddTest(const char* name, std::vector<T>& x1Data, const std::vector<int64_t>& x1Shape,
                      const std::vector<T>& x2Data, const std::vector<int64_t>& x2Shape,
                      float alpha, aclDataType dt, aclrtStream stream,
                      double atol, double rtol) {
    void *x1Dev = nullptr, *x2Dev = nullptr;
    aclTensor *x1T = nullptr, *x2T = nullptr;
    aclScalar* alphaScalar = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    void* wsAddr = nullptr;
    int ret;
    std::vector<T> result;
    int failed;
    size_t i;
    double expected;

    if (CreateAclTensor(x1Data, x1Shape, &x1Dev, dt, &x1T) != 0) goto cleanup;
    if (CreateAclTensor(x2Data, x2Shape, &x2Dev, dt, &x2T) != 0) goto cleanup;

    alphaScalar = CreateAclScalar(alpha, dt);
    CHECK_RET(alphaScalar != nullptr, goto cleanup);

    ret = aclnnInplaceAddGetWorkspaceSize(x1T, x2T, alphaScalar, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAdd GetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnInplaceAdd(wsAddr, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAdd Execute failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    result.resize(x1Data.size());
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(T), x1Dev, result.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    failed = 0;
    for (i = 0; i < result.size(); i++) {
        expected = static_cast<double>(x1Data[i]) +
                  static_cast<double>(alpha) * static_cast<double>(x2Data[i]);
        if (!AlmostEqual(expected, static_cast<double>(result[i]), atol, rtol)) {
            failed++;
        }
    }

    LOG_PRINT("%s [%s] - %zu/%zu passed, alpha=%.2f\n",
              name, failed == 0 ? "PASS" : "FAIL", result.size() - failed, result.size(), alpha);

cleanup:
    if (x1T) aclDestroyTensor(x1T);
    if (x2T) aclDestroyTensor(x2T);
    if (alphaScalar) aclDestroyScalar(alphaScalar);
    if (x1Dev) aclrtFree(x1Dev);
    if (x2Dev) aclrtFree(x2Dev);
    if (wsAddr) aclrtFree(wsAddr);
    return failed > 0 ? 1 : 0;
}

template <typename T>
int RunInplaceAddsTest(const char* name, std::vector<T>& x1Data, const std::vector<int64_t>& x1Shape,
                       T x2Scalar, float alpha, aclDataType dt, aclrtStream stream,
                       double atol, double rtol) {
    void *x1Dev = nullptr;
    aclTensor *x1T = nullptr;
    aclScalar* x2ScalarPtr = nullptr;
    aclScalar* alphaScalar = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    void* wsAddr = nullptr;
    int ret;
    std::vector<T> result;
    int failed;
    size_t i;
    double expected;

    if (CreateAclTensor(x1Data, x1Shape, &x1Dev, dt, &x1T) != 0) goto cleanup;

    x2ScalarPtr = aclCreateScalar(&x2Scalar, dt);
    alphaScalar = CreateAclScalar(alpha, dt);
    CHECK_RET(x2ScalarPtr != nullptr && alphaScalar != nullptr, goto cleanup);

    ret = aclnnInplaceAddsGetWorkspaceSize(x1T, x2ScalarPtr, alphaScalar, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAdds GetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnInplaceAdds(wsAddr, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAdds Execute failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    result.resize(x1Data.size());
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(T), x1Dev, result.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    failed = 0;
    for (i = 0; i < result.size(); i++) {
        expected = static_cast<double>(x1Data[i]) +
                  static_cast<double>(alpha) * static_cast<double>(x2Scalar);
        if (!AlmostEqual(expected, static_cast<double>(result[i]), atol, rtol)) {
            failed++;
        }
    }

    LOG_PRINT("%s [%s] - %zu/%zu passed, alpha=%.2f, x2=%.2f\n",
              name, failed == 0 ? "PASS" : "FAIL", result.size() - failed, result.size(), alpha, x2Scalar);

cleanup:
    if (x1T) aclDestroyTensor(x1T);
    if (x2ScalarPtr) aclDestroyScalar(x2ScalarPtr);
    if (alphaScalar) aclDestroyScalar(alphaScalar);
    if (x1Dev) aclrtFree(x1Dev);
    if (wsAddr) aclrtFree(wsAddr);
    return failed > 0 ? 1 : 0;
}

template <typename T2, typename Tout>
int RunAddV3Test(const char* name, float selfScalar, const std::vector<T2>& x2Data,
                 const std::vector<int64_t>& x2Shape, float alpha, aclDataType dt2,
                 aclDataType dtOut, aclrtStream stream, double atol, double rtol) {
    void *x2Dev = nullptr, *outDev = nullptr;
    aclTensor *x2T = nullptr, *outT = nullptr;
    aclScalar* selfScalarPtr = nullptr;
    aclScalar* alphaScalar = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    void* wsAddr = nullptr;
    int ret;
    int64_t outSize;
    std::vector<Tout> outHost;
    std::vector<Tout> result;
    int failed;
    int64_t i;
    double expected, actual;

    outSize = GetShapeSize(x2Shape);
    outHost.resize(outSize, 0);

    selfScalarPtr = aclCreateScalar(&selfScalar, ACL_FLOAT);
    CHECK_RET(selfScalarPtr != nullptr, return -1);

    if (CreateAclTensor(x2Data, x2Shape, &x2Dev, dt2, &x2T) != 0) goto cleanup;
    if (CreateAclTensor(outHost, x2Shape, &outDev, dtOut, &outT) != 0) goto cleanup;

    alphaScalar = CreateAclScalar(alpha, dt2);
    CHECK_RET(alphaScalar != nullptr, goto cleanup);

    ret = aclnnAddV3GetWorkspaceSize(selfScalarPtr, x2T, alphaScalar, outT, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AddV3 GetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnAddV3(wsAddr, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AddV3 Execute failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    result.resize(outSize);
    ret = aclrtMemcpy(result.data(), outSize * sizeof(Tout), outDev, outSize * sizeof(Tout),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    failed = 0;
    for (i = 0; i < outSize; i++) {
        expected = static_cast<double>(selfScalar) +
                  static_cast<double>(alpha) * static_cast<double>(x2Data[i]);
        actual = static_cast<double>(result[i]);

        if (!AlmostEqual(expected, actual, atol, rtol)) {
            failed++;
        }
    }

    LOG_PRINT("%s [%s] - %ld/%ld passed, self=%.2f, alpha=%.2f\n",
              name, failed == 0 ? "PASS" : "FAIL", outSize - failed, outSize, selfScalar, alpha);

cleanup:
    if (selfScalarPtr) aclDestroyScalar(selfScalarPtr);
    if (x2T) aclDestroyTensor(x2T);
    if (outT) aclDestroyTensor(outT);
    if (alphaScalar) aclDestroyScalar(alphaScalar);
    if (x2Dev) aclrtFree(x2Dev);
    if (outDev) aclrtFree(outDev);
    if (wsAddr) aclrtFree(wsAddr);
    return failed > 0 ? 1 : 0;
}

template <typename T>
int RunInplaceAddV3Test(const char* name, float selfScalar, const std::vector<T>& x2Data,
                        const std::vector<int64_t>& x2Shape, float alpha, aclDataType dt,
                        aclrtStream stream, double atol, double rtol) {
    void *x2Dev = nullptr;
    aclTensor *x2T = nullptr;
    aclScalar* selfScalarPtr = nullptr;
    aclScalar* alphaScalar = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    void* wsAddr = nullptr;
    int ret;
    std::vector<T> result;
    std::vector<T> initialX2;
    int failed;
    size_t i;
    double expected;
    size_t copySize;

    initialX2 = x2Data;

    selfScalarPtr = aclCreateScalar(&selfScalar, ACL_FLOAT);
    CHECK_RET(selfScalarPtr != nullptr, return -1);

    if (CreateAclTensor(x2Data, x2Shape, &x2Dev, dt, &x2T) != 0) goto cleanup;

    alphaScalar = CreateAclScalar(alpha, dt);
    CHECK_RET(alphaScalar != nullptr, goto cleanup);

    ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalarPtr, x2T, alphaScalar, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAddV3 GetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);

    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnInplaceAddV3(wsAddr, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAddV3 Execute failed. ERROR: %d\n", ret); goto cleanup);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAddV3 Sync failed. ERROR: %d\n", ret); goto cleanup);

    result.resize(x2Data.size());
    copySize = result.size() * sizeof(T);
    ret = aclrtMemcpy(result.data(), copySize, x2Dev, copySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplaceAddV3 Memcpy failed. ERROR: %d\n", ret); goto cleanup);

    failed = 0;
    for (i = 0; i < result.size(); i++) {
        expected = static_cast<double>(selfScalar) +
                  static_cast<double>(alpha) * static_cast<double>(initialX2[i]);
        if (!AlmostEqual(expected, static_cast<double>(result[i]), atol, rtol)) {
            failed++;
        }
    }

    LOG_PRINT("%s [%s] - %zu/%zu passed, self=%.2f, alpha=%.2f\n",
              name, failed == 0 ? "PASS" : "FAIL", result.size() - failed, result.size(), selfScalar, alpha);

cleanup:
    if (selfScalarPtr) aclDestroyScalar(selfScalarPtr);
    if (x2T) aclDestroyTensor(x2T);
    if (alphaScalar) aclDestroyScalar(alphaScalar);
    if (x2Dev) aclrtFree(x2Dev);
    if (wsAddr) aclrtFree(wsAddr);
    return failed > 0 ? 1 : 0;
}

// ========== 批次1: 基础FLOAT32测试 (5个测试) ==========
int RunBatch1() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int totalFailed = 0;
    LOG_PRINT("========== Batch 1: FLOAT32 Basic Tests ==========\n");

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_alpha_1.0", {0, 1, 2, 3, 4, 5, 6, 7}, {4, 2},
        {1, 1, 1, 2, 2, 2, 3, 3}, {4, 2},
        1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_alpha_0.0", {1, 2, 3, 4}, {2, 2},
        {5, 6, 7, 8}, {2, 2},
        0.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_alpha_negative", {10, 20, 30, 40}, {2, 2},
        {5, 10, 15, 20}, {2, 2},
        -1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_alpha_2.5", {1, 2, 3, 4}, {2, 2},
        {2, 4, 6, 8}, {2, 2},
        2.5f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_boundary", {-1e6f, -1.0f, 0.0f, 1e6f}, {2, 2},
        {-1e6f, 0.0f, 1.0f, 1e6f}, {2, 2},
        1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    LOG_PRINT("Batch 1 Complete. Failed: %d\n", totalFailed);
    CleanupStream(stream, deviceId);
    return totalFailed;
}

// ========== 批次2: API变体测试 (6个测试) ==========
int RunBatch2() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int totalFailed = 0;
    LOG_PRINT("========== Batch 2: API Variant Tests ==========\n");

    totalFailed += RunAddsTest<float>(
        "FLOAT32_Adds", {1, 2, 3, 4}, {2, 2},
        5.0f, 2.0f, ACL_FLOAT, stream, 1e-6, 1e-6);

    {
        std::vector<float> x1Data1 = {1, 2, 3, 4};
        totalFailed += RunInplaceAddTest<float>(
            "FLOAT32_InplaceAdd", x1Data1, {2, 2},
            {5, 6, 7, 8}, {2, 2},
            1.0f, ACL_FLOAT, stream, 1e-6, 1e-6);
    }

    {
        std::vector<float> x1Data2 = {1, 2, 3, 4};
        totalFailed += RunInplaceAddsTest<float>(
            "FLOAT32_InplaceAdds", x1Data2, {2, 2},
            10.0f, 0.5f, ACL_FLOAT, stream, 1e-6, 1e-6);
    }

    totalFailed += RunAddV3Test<float, float>(
        "V3_scalar_add_tensor", 10.0f,
        {1, 2, 3, 4}, {2, 2},
        2.0f,
        ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6);

    totalFailed += RunAddV3Test<float, float>(
        "V3_alpha_zero", 5.0f,
        {1, 2, 3}, {3},
        0.0f,
        ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6);

    totalFailed += RunInplaceAddV3Test<float>(
        "V3_inplace_scalar", 10.0f,
        {1, 2, 3}, {3},
        1.5f,
        ACL_FLOAT, stream, 1e-6, 1e-6);

    LOG_PRINT("Batch 2 Complete. Failed: %d\n", totalFailed);
    CleanupStream(stream, deviceId);
    return totalFailed;
}

// ========== 批次3: 数据类型测试 (6个测试) ==========
int RunBatch3() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int totalFailed = 0;
    LOG_PRINT("========== Batch 3: Data Type Tests ==========\n");

    totalFailed += RunAddTest<uint16_t, uint16_t, uint16_t>(
        "FLOAT16_basic", {0x3c00, 0x4000, 0x4200, 0x4400}, {2, 2},
        {0x3c00, 0x3c00, 0x4000, 0x4200}, {2, 2},
        1.0f, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, stream, 1e-4, 1e-4, false);

    totalFailed += RunAddTest<int32_t, int32_t, int32_t>(
        "INT32_basic", {1, 2, 3, 4, 5, 6, 7, 8}, {4, 2},
        {10, 20, 30, 40, 50, 60, 70, 80}, {4, 2},
        2.0f, ACL_INT32, ACL_INT32, ACL_INT32, stream, 0.0, 0.0, false);

    totalFailed += RunAddTest<int32_t, int32_t, int32_t>(
        "INT32_negative", {-10, -20, 0, 10}, {2, 2},
        {-5, -10, 5, 10}, {2, 2},
        -1.0f, ACL_INT32, ACL_INT32, ACL_INT32, stream, 0.0, 0.0, false);

    totalFailed += RunAddTest<int8_t, int8_t, int8_t>(
        "INT8_overflow", {100, 50, 20, 10}, {2, 2},
        {50, 80, 110, 120}, {2, 2},
        1.0f, ACL_INT8, ACL_INT8, ACL_INT8, stream, 1.0, 0.0, false);

    totalFailed += RunAddTest<uint8_t, uint8_t, uint8_t>(
        "UINT8_basic", {1, 2, 3, 4}, {2, 2},
        {10, 20, 30, 40}, {2, 2},
        1.5f, ACL_UINT8, ACL_UINT8, ACL_UINT8, stream, 1.0, 0.0, false);

    totalFailed += RunAddTest<double, double, double>(
        "DOUBLE_basic", {1.0, 2.0, 3.1415926535, 1e100}, {2, 2},
        {10.0, 20.0, 2.7182818284, -1e100}, {2, 2},
        1.0f, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, stream, 1e-9, 1e-9, false);

    LOG_PRINT("Batch 3 Complete. Failed: %d\n", totalFailed);
    CleanupStream(stream, deviceId);
    return totalFailed;
}

// ========== 批次4: 广播测试 (3个测试) ==========
int RunBatch4() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int totalFailed = 0;
    LOG_PRINT("========== Batch 4: Broadcast Tests ==========\n");

    totalFailed += RunAddTest<float, float, float>(
        "broadcast_right", {1, 2, 3, 4, 5, 6}, {2, 3},
        {10, 20, 30}, {3},
        1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    totalFailed += RunAddTest<float, float, float>(
        "broadcast_both", {1, 2, 3, 4}, {4, 1},
        {10, 20, 30}, {1, 3},
        2.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    totalFailed += RunAddTest<float, float, float>(
        "broadcast_multi", {1, 2}, {2, 1},
        {10, 20, 30}, {1, 3},
        1.5f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    LOG_PRINT("Batch 4 Complete. Failed: %d\n", totalFailed);
    CleanupStream(stream, deviceId);
    return totalFailed;
}

// ========== 批次5: 边界值测试 (7个测试) ==========
int RunBatch5() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int totalFailed = 0;
    LOG_PRINT("========== Batch 5: Boundary Value Tests ==========\n");

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_zero", {0, 1, 2, 3}, {2, 2},
        {0, 5, 10, 15}, {2, 2},
        1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_inf", {std::numeric_limits<float>::infinity(), 1.0f}, {2, 1},
        {1.0f, -std::numeric_limits<float>::infinity()}, {2, 1},
        1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 0.0, 0.0, false);

    totalFailed += RunAddTest<float, float, float>(
        "FLOAT32_nan", {std::numeric_limits<float>::quiet_NaN(), 1.0f}, {2, 1},
        {1.0f, 2.0f}, {2, 1},
        1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 0.0, 0.0, false);

    totalFailed += RunAddTest<double, double, double>(
        "DOUBLE_max", {std::numeric_limits<double>::max(), 1e100}, {2, 1},
        {1e100, -1e100}, {2, 1},
        1.0f, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, stream, 1e-9, 1e-9, false);

    totalFailed += RunAddTest<int32_t, int32_t, int32_t>(
        "INT32_max_min", {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min()}, {2, 1},
        {0, 0}, {2, 1},
        1.0f, ACL_INT32, ACL_INT32, ACL_INT32, stream, 0.0, 0.0, false);

    totalFailed += RunAddTest<int8_t, int8_t, int8_t>(
        "INT8_max_min", {std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min()}, {2, 1},
        {0, 0}, {2, 1},
        1.0f, ACL_INT8, ACL_INT8, ACL_INT8, stream, 0.0, 0.0, false);

    totalFailed += RunAddTest<float, float, float>(
        "single_element", {5.0f}, {1},
        {3.0f}, {1},
        2.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-6, 1e-6, false);

    LOG_PRINT("Batch 5 Complete. Failed: %d\n", totalFailed);
    CleanupStream(stream, deviceId);
    return totalFailed;
}

int main() {
    int totalFailed = 0;
    LOG_PRINT("========== Add Operator Test Suite (Batched) ==========\n\n");

    totalFailed += RunBatch1();
    totalFailed += RunBatch2();
    totalFailed += RunBatch3();
    totalFailed += RunBatch4();
    totalFailed += RunBatch5();

    LOG_PRINT("\n========== FINAL Summary ==========\n");
    LOG_PRINT("Total Failed: %d\n", totalFailed);
    LOG_PRINT("===================================\n");

    return totalFailed > 0 ? 1 : 0;
}
