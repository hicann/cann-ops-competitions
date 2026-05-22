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
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>
#include <functional>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif
#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif
#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
#endif
#ifndef ACLNN_ERR_INNER_NULLPTR
#define ACLNN_ERR_INNER_NULLPTR 161003
#endif

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

static int g_passCount = 0;
static int g_failCount = 0;
static int32_t g_deviceId = 0;
static aclrtStream g_stream = nullptr;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size == 0) {
        *deviceAddr = nullptr;
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), 0);
        return 0;
    }
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// =============== CPU Reference Implementation ===============

template <typename T>
void CpuCumsum(const std::vector<T>& input, const std::vector<int64_t>& shape,
               int64_t dim, bool exclusive, bool reverse, std::vector<T>& output)
{
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim == 0) {
        ndim = 1;
    }
    int64_t adjDim = dim;
    if (adjDim < 0) {
        adjDim += ndim;
    }

    int64_t outerStride = 1;
    for (int64_t i = 0; i < adjDim; i++) {
        outerStride *= shape[i];
    }
    int64_t dimLen = shape[adjDim];
    int64_t innerStride = 1;
    for (int64_t i = adjDim + 1; i < static_cast<int64_t>(shape.size()); i++) {
        innerStride *= shape[i];
    }

    output.resize(input.size());
    for (int64_t outer = 0; outer < outerStride; outer++) {
        for (int64_t inner = 0; inner < innerStride; inner++) {
            std::vector<double> slice(dimLen);
            for (int64_t d = 0; d < dimLen; d++) {
                int64_t idx;
                if (static_cast<int64_t>(shape.size()) == 0) {
                    idx = 0;
                } else {
                    idx = outer * dimLen * innerStride + d * innerStride + inner;
                }
                slice[d] = static_cast<double>(input[idx]);
            }

            if (reverse) {
                std::reverse(slice.begin(), slice.end());
            }

            std::vector<double> cumsum(dimLen);
            double sum = 0.0;
            for (int64_t d = 0; d < dimLen; d++) {
                sum += slice[d];
                cumsum[d] = exclusive ? (d == 0 ? 0.0 : cumsum[d - 1] + slice[d - 1]) : sum;
            }
            if (exclusive && dimLen > 0) {
                double runSum = 0.0;
                for (int64_t d = 0; d < dimLen; d++) {
                    cumsum[d] = runSum;
                    runSum += slice[d];
                }
            }

            if (reverse) {
                std::reverse(cumsum.begin(), cumsum.end());
            }

            for (int64_t d = 0; d < dimLen; d++) {
                int64_t idx;
                if (static_cast<int64_t>(shape.size()) == 0) {
                    idx = 0;
                } else {
                    idx = outer * dimLen * innerStride + d * innerStride + inner;
                }
                output[idx] = static_cast<T>(cumsum[d]);
            }
        }
    }
}

// =============== Tolerance Comparison ===============

template <typename T>
bool CompareResult(const std::vector<T>& actual, const std::vector<T>& expected,
                   int64_t count, const char* dtype_name, double rtol, double atol)
{
    int failCount = 0;
    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    int64_t maxErrIdx = -1;
    for (int64_t i = 0; i < count; i++) {
        double a = static_cast<double>(actual[i]);
        double e = static_cast<double>(expected[i]);
        double absErr = std::fabs(a - e);
        double relErr = (std::fabs(e) > 1e-30) ? absErr / std::fabs(e) : absErr;
        bool pass = (absErr <= atol + rtol * std::fabs(e));
        if (!pass) {
            failCount++;
            if (absErr > maxAbsErr) {
                maxAbsErr = absErr;
                maxRelErr = relErr;
                maxErrIdx = i;
            }
        }
    }
    if (failCount > 0) {
        LOG_PRINT("  Mismatch: %d/%ld elements, max_abs_err=%.10e at idx=%ld (actual=%.10e, expected=%.10e)\n",
                  failCount, count, maxAbsErr, maxErrIdx,
                  maxErrIdx >= 0 ? static_cast<double>(actual[maxErrIdx]) : 0.0,
                  maxErrIdx >= 0 ? static_cast<double>(expected[maxErrIdx]) : 0.0);
        return false;
    }
    return true;
}

// =============== V1 API Test Runner ===============

template <typename T>
bool RunCumsumV1Test(const char* testName, const std::vector<T>& hostData,
                     const std::vector<int64_t>& shape, int64_t dim, aclDataType dtype,
                     double rtol, double atol)
{
    auto size = GetShapeSize(shape);
    std::vector<T> outHostData(size, static_cast<T>(0));

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = CreateAclTensor(hostData, shape, &selfDeviceAddr, dtype, &self);
    if (ret != 0) { LOG_PRINT("[FAIL] %s: CreateAclTensor(self) failed\n", testName); g_failCount++; return false; }
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
    if (ret != 0) {
        LOG_PRINT("[FAIL] %s: CreateAclTensor(out) failed\n", testName);
        if (self) aclDestroyTensor(self);
        if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
        g_failCount++; return false;
    }

    ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    if (ret != ACLNN_SUCCESS) {
        LOG_PRINT("[FAIL] %s: GetWorkspaceSize returned %d\n", testName, ret);
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
        g_failCount++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s: malloc workspace failed\n", testName);
            aclDestroyTensor(self); aclDestroyTensor(out);
            aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
            g_failCount++; return false;
        }
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, g_stream);
    if (ret != ACLNN_SUCCESS) {
        LOG_PRINT("[FAIL] %s: aclnnCumsum returned %d\n", testName, ret);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
        g_failCount++; return false;
    }

    ret = aclrtSynchronizeStream(g_stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s: SynchronizeStream failed %d\n", testName, ret);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
        g_failCount++; return false;
    }

    std::vector<T> resultData(size, static_cast<T>(0));
    if (size > 0 && outDeviceAddr != nullptr) {
        ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outDeviceAddr, size * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s: D2H copy failed\n", testName);
            if (workspaceAddr) aclrtFree(workspaceAddr);
            aclDestroyTensor(self); aclDestroyTensor(out);
            aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
            g_failCount++; return false;
        }
    }

    std::vector<T> expected;
    CpuCumsum(hostData, shape, dim, false, false, expected);

    const char* dtypeName = "";
    switch (dtype) {
        case ACL_FLOAT: dtypeName = "FLOAT"; break;
        case ACL_FLOAT16: dtypeName = "FLOAT16"; break;
        case ACL_BF16: dtypeName = "BF16"; break;
        case ACL_INT32: dtypeName = "INT32"; break;
        case ACL_INT64: dtypeName = "INT64"; break;
        case ACL_INT8: dtypeName = "INT8"; break;
        case ACL_UINT8: dtypeName = "UINT8"; break;
        case ACL_DOUBLE: dtypeName = "DOUBLE"; break;
        default: dtypeName = "OTHER"; break;
    }

    bool passed = CompareResult(resultData, expected, size, dtypeName, rtol, atol);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);

    LOG_PRINT("[%s] %s (%s, shape=?, dim=%ld)\n", passed ? "PASS" : "FAIL", testName, dtypeName, dim);
    if (passed) g_passCount++; else g_failCount++;
    return passed;
}

// =============== V2 API Test Runner ===============

template <typename T>
bool RunCumsumV2Test(const char* testName, const std::vector<T>& hostData,
                     const std::vector<int64_t>& shape, int64_t dim,
                     bool exclusive, bool reverse, aclDataType dtype,
                     double rtol, double atol)
{
    auto size = GetShapeSize(shape);
    std::vector<T> outHostData(size, static_cast<T>(0));

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = CreateAclTensor(hostData, shape, &selfDeviceAddr, dtype, &self);
    if (ret != 0) { LOG_PRINT("[FAIL] %s: CreateAclTensor(self) failed\n", testName); g_failCount++; return false; }
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
    if (ret != 0) {
        LOG_PRINT("[FAIL] %s: CreateAclTensor(out) failed\n", testName);
        if (self) aclDestroyTensor(self);
        if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
        g_failCount++; return false;
    }

    ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    if (ret != ACLNN_SUCCESS) {
        LOG_PRINT("[FAIL] %s: V2 GetWorkspaceSize returned %d\n", testName, ret);
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
        g_failCount++; return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s: malloc workspace failed\n", testName);
            aclDestroyTensor(self); aclDestroyTensor(out);
            aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
            g_failCount++; return false;
        }
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
    if (ret != ACLNN_SUCCESS) {
        LOG_PRINT("[FAIL] %s: aclnnCumsumV2 returned %d\n", testName, ret);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
        g_failCount++; return false;
    }

    ret = aclrtSynchronizeStream(g_stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s: SynchronizeStream failed %d\n", testName, ret);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
        g_failCount++; return false;
    }

    std::vector<T> resultData(size, static_cast<T>(0));
    if (size > 0 && outDeviceAddr != nullptr) {
        ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outDeviceAddr, size * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s: D2H copy failed\n", testName);
            if (workspaceAddr) aclrtFree(workspaceAddr);
            aclDestroyTensor(self); aclDestroyTensor(out);
            aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);
            g_failCount++; return false;
        }
    }

    std::vector<T> expected;
    CpuCumsum(hostData, shape, dim, exclusive, reverse, expected);

    bool passed = CompareResult(resultData, expected, size, "V2", rtol, atol);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr); aclrtFree(outDeviceAddr);

    LOG_PRINT("[%s] %s (excl=%d, rev=%d, dim=%ld)\n",
              passed ? "PASS" : "FAIL", testName, exclusive, reverse, dim);
    if (passed) g_passCount++; else g_failCount++;
    return passed;
}

// =============== Negative Test Runner ===============

bool RunNegativeTestV1(const char* testName, aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out,
                       aclnnStatus expectedErr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    bool passed = (ret == expectedErr);
    LOG_PRINT("[%s] %s: expected %d, got %d\n", passed ? "PASS" : "FAIL", testName, expectedErr, ret);
    if (passed) g_passCount++; else g_failCount++;
    return passed;
}

bool RunNegativeTestV2(const char* testName, aclTensor* self, int64_t dim, bool exclusive, bool reverse,
                       aclTensor* out, aclnnStatus expectedErr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    bool passed = (ret == expectedErr);
    LOG_PRINT("[%s] %s: expected %d, got %d\n", passed ? "PASS" : "FAIL", testName, expectedErr, ret);
    if (passed) g_passCount++; else g_failCount++;
    return passed;
}

// =============== Test Group 1: Negative Parameter Validation ===============

void TestNegativeParamValidation()
{
    LOG_PRINT("\n=== Group 1: Negative Parameter Validation ===\n");

    // N1: null self (V1)
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 0.0f);
        void* devAddr = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(data, shape, &devAddr, ACL_FLOAT, &out);
        RunNegativeTestV1("V1_NullSelf", nullptr, 0, ACL_FLOAT, out, ACLNN_ERR_PARAM_NULLPTR);
        aclDestroyTensor(out);
        if (devAddr) aclrtFree(devAddr);
    }

    // N2: null out (V1)
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 0.0f);
        void* devAddr = nullptr;
        aclTensor* self = nullptr;
        CreateAclTensor(data, shape, &devAddr, ACL_FLOAT, &self);
        RunNegativeTestV1("V1_NullOut", self, 0, ACL_FLOAT, nullptr, ACLNN_ERR_PARAM_NULLPTR);
        aclDestroyTensor(self);
        if (devAddr) aclrtFree(devAddr);
    }

    // N3: null self (V2)
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 0.0f);
        void* devAddr = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(data, shape, &devAddr, ACL_FLOAT, &out);
        RunNegativeTestV2("V2_NullSelf", nullptr, 0, false, false, out, ACLNN_ERR_PARAM_NULLPTR);
        aclDestroyTensor(out);
        if (devAddr) aclrtFree(devAddr);
    }

    // N4: null out (V2)
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 0.0f);
        void* devAddr = nullptr;
        aclTensor* self = nullptr;
        CreateAclTensor(data, shape, &devAddr, ACL_FLOAT, &self);
        RunNegativeTestV2("V2_NullOut", self, 0, false, false, nullptr, ACLNN_ERR_PARAM_NULLPTR);
        aclDestroyTensor(self);
        if (devAddr) aclrtFree(devAddr);
    }

    // D2: V1 dtype/out mismatch (ACL_FLOAT vs ACL_FLOAT16)
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT, &self);
        CreateAclTensor(data, shape, &outDevAddr, ACL_FLOAT16, &out);
        RunNegativeTestV1("V1_DtypeMismatch", self, 0, ACL_FLOAT, out, ACLNN_ERR_PARAM_INVALID);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // D6: V2 self/out dtype mismatch
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT, &self);
        CreateAclTensor(data, shape, &outDevAddr, ACL_FLOAT16, &out);
        RunNegativeTestV2("V2_DtypeMismatch", self, 0, false, false, out, ACLNN_ERR_PARAM_INVALID);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // DM4: dim out of range (positive)
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT, &self);
        CreateAclTensor(data, shape, &outDevAddr, ACL_FLOAT, &out);
        RunNegativeTestV1("V1_DimOutOfRangePos", self, 5, ACL_FLOAT, out, ACLNN_ERR_PARAM_INVALID);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // DM5: dim out of range (negative)
    {
        std::vector<int64_t> shape = {2, 3};
        std::vector<float> data(6, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT, &self);
        CreateAclTensor(data, shape, &outDevAddr, ACL_FLOAT, &out);
        RunNegativeTestV1("V1_DimOutOfRangeNeg", self, -3, ACL_FLOAT, out, ACLNN_ERR_PARAM_INVALID);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // S2: shape mismatch
    {
        std::vector<int64_t> selfShape = {2, 3};
        std::vector<int64_t> outShape = {3, 2};
        std::vector<float> selfData(6, 1.0f);
        std::vector<float> outData(6, 0.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(selfData, selfShape, &selfDevAddr, ACL_FLOAT, &self);
        CreateAclTensor(outData, outShape, &outDevAddr, ACL_FLOAT, &out);
        RunNegativeTestV1("V1_ShapeMismatch", self, 0, ACL_FLOAT, out, ACLNN_ERR_PARAM_INVALID);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // S4: exceeds max dims (9-dim)
    {
        std::vector<int64_t> shape9d = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        int64_t total = GetShapeSize(shape9d);
        std::vector<float> data(total, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        CreateAclTensor(data, shape9d, &selfDevAddr, ACL_FLOAT, &self);
        CreateAclTensor(data, shape9d, &outDevAddr, ACL_FLOAT, &out);
        RunNegativeTestV1("V1_ExceedsMaxDim", self, 0, ACL_FLOAT, out, ACLNN_ERR_PARAM_INVALID);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }
}

// =============== Test Group 2: Empty Tensor ===============

void TestEmptyTensor()
{
    LOG_PRINT("\n=== Group 2: Empty Tensor ===\n");

    // V1 empty tensor: shape={0}
    {
        std::vector<int64_t> shape = {0};
        std::vector<float> data;
        RunCumsumV1Test("V1_EmptyTensor_1d", data, shape, 0, ACL_FLOAT, 1e-5, 1e-6);
    }

    // V2 empty tensor: shape={2, 0, 3}
    {
        std::vector<int64_t> shape = {2, 0, 3};
        std::vector<float> data;
        RunCumsumV2Test("V2_EmptyTensor_3d", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }
}

// =============== Test Group 3: V1 Basic FLOAT ===============

void TestV1BasicFloat()
{
    LOG_PRINT("\n=== Group 3: V1 Basic FLOAT ===\n");

    // dim=0 triggers DT_INT64 dimTensor path
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        RunCumsumV1Test("V1_Float_3x4_dim0", data, shape, 0, ACL_FLOAT, 1e-5, 1e-6);
    }

    // dim=1 triggers DT_INT32 dimTensor path
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        RunCumsumV1Test("V1_Float_3x4_dim1", data, shape, 1, ACL_FLOAT, 1e-5, 1e-6);
    }

    // dim=-1 (negative dim)
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        RunCumsumV1Test("V1_Float_3x4_dimNeg1", data, shape, -1, ACL_FLOAT, 1e-5, 1e-6);
    }

    // dim=-2
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        RunCumsumV1Test("V1_Float_3x4_dimNeg2", data, shape, -2, ACL_FLOAT, 1e-5, 1e-6);
    }

    // 1D shape
    {
        std::vector<int64_t> shape = {8};
        std::vector<float> data = {1,2,3,4,5,6,7,8};
        RunCumsumV1Test("V1_Float_1d_8", data, shape, 0, ACL_FLOAT, 1e-5, 1e-6);
    }

    // 8-dim shape (max dims boundary)
    {
        std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 2};
        std::vector<float> data(2, 1.0f);
        RunCumsumV1Test("V1_Float_8dim", data, shape, 7, ACL_FLOAT, 1e-5, 1e-6);
    }

    // Simple 2x2
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<float> data = {1, 2, 3, 4};
        RunCumsumV1Test("V1_Float_2x2_dim0", data, shape, 0, ACL_FLOAT, 1e-5, 1e-6);
    }
}

// =============== Test Group 4: V1 dtype Variants ===============

void TestV1DtypeVariants()
{
    LOG_PRINT("\n=== Group 4: V1 dtype Variants ===\n");

    // FLOAT16
    {
        std::vector<int64_t> shape = {4, 8};
        std::vector<float> dataF(32);
        for (int i = 0; i < 32; i++) dataF[i] = static_cast<float>(i + 1);
        // FP16 stored as float in host, dtype=ACL_FLOAT16
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;

        CreateAclTensor(dataF, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(32, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);

        auto ret = aclnnCumsumGetWorkspaceSize(self, 1, ACL_FLOAT16, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            std::vector<float> result(32, 0.0f);
            aclrtMemcpy(result.data(), 32 * sizeof(float), outDevAddr, 32 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(dataF, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, 32, "FLOAT16", 1e-3, 1e-3);
            LOG_PRINT("[%s] V1_Float16_4x8_dim1\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] V1_Float16_4x8_dim1: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // BF16
    {
        std::vector<int64_t> shape = {4, 8};
        std::vector<float> dataF(32);
        for (int i = 0; i < 32; i++) dataF[i] = static_cast<float>(i + 1);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;

        CreateAclTensor(dataF, shape, &selfDevAddr, ACL_BF16, &self);
        std::vector<float> outData(32, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_BF16, &out);

        auto ret = aclnnCumsumGetWorkspaceSize(self, 1, ACL_BF16, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            std::vector<float> result(32, 0.0f);
            aclrtMemcpy(result.data(), 32 * sizeof(float), outDevAddr, 32 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(dataF, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, 32, "BF16", 1e-2, 1e-2);
            LOG_PRINT("[%s] V1_BF16_4x8_dim1\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] V1_BF16_4x8_dim1: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // INT32
    {
        std::vector<int64_t> shape = {4, 8};
        std::vector<int32_t> data(32);
        for (int i = 0; i < 32; i++) data[i] = i + 1;
        RunCumsumV1Test("V1_Int32_4x8_dim1", data, shape, 1, ACL_INT32, 0, 0);
    }

    // INT64
    {
        std::vector<int64_t> shape = {4, 8};
        std::vector<int64_t> data(32);
        for (int i = 0; i < 32; i++) data[i] = i + 1;
        RunCumsumV1Test("V1_Int64_4x8_dim1", data, shape, 1, ACL_INT64, 0, 0);
    }

    // INT8
    {
        std::vector<int64_t> shape = {4, 8};
        std::vector<int8_t> data(32);
        for (int i = 0; i < 32; i++) data[i] = static_cast<int8_t>((i % 10) + 1);
        RunCumsumV1Test("V1_Int8_4x8_dim1", data, shape, 1, ACL_INT8, 0, 0);
    }

    // UINT8
    {
        std::vector<int64_t> shape = {4, 8};
        std::vector<uint8_t> data(32);
        for (int i = 0; i < 32; i++) data[i] = static_cast<uint8_t>((i % 10) + 1);
        RunCumsumV1Test("V1_Uint8_4x8_dim1", data, shape, 1, ACL_UINT8, 0, 0);
    }

    // DOUBLE (AiCpu path)
    {
        std::vector<int64_t> shape = {4};
        std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
        RunCumsumV1Test("V1_Double_1d_4", data, shape, 0, ACL_DOUBLE, 1e-12, 1e-12);
    }
}

// =============== Test Group 5: V2 exclusive + reverse ===============

void TestV2Attributes()
{
    LOG_PRINT("\n=== Group 5: V2 exclusive + reverse ===\n");

    // 4 combinations of exclusive × reverse for FLOAT
    {
        std::vector<int64_t> shape = {2, 5};
        std::vector<float> data = {1,2,3,4,5, 6,7,8,9,10};
        RunCumsumV2Test("V2_Float_exF_revF", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
        RunCumsumV2Test("V2_Float_exT_revF", data, shape, 1, true, false, ACL_FLOAT, 1e-5, 1e-6);
        RunCumsumV2Test("V2_Float_exF_revT", data, shape, 1, false, true, ACL_FLOAT, 1e-5, 1e-6);
        RunCumsumV2Test("V2_Float_exT_revT", data, shape, 1, true, true, ACL_FLOAT, 1e-5, 1e-6);
    }

    // V2 with dim=0
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        RunCumsumV2Test("V2_Float_dim0_exF_revF", data, shape, 0, false, false, ACL_FLOAT, 1e-5, 1e-6);
        RunCumsumV2Test("V2_Float_dim0_exT_revF", data, shape, 0, true, false, ACL_FLOAT, 1e-5, 1e-6);
        RunCumsumV2Test("V2_Float_dim0_exF_revT", data, shape, 0, false, true, ACL_FLOAT, 1e-5, 1e-6);
        RunCumsumV2Test("V2_Float_dim0_exT_revT", data, shape, 0, true, true, ACL_FLOAT, 1e-5, 1e-6);
    }

    // V2 with INT32
    {
        std::vector<int64_t> shape = {4, 6};
        std::vector<int32_t> data(24);
        for (int i = 0; i < 24; i++) data[i] = (i % 5) + 1;
        RunCumsumV2Test("V2_Int32_exF_revF", data, shape, 1, false, false, ACL_INT32, 0, 0);
        RunCumsumV2Test("V2_Int32_exT_revF", data, shape, 1, true, false, ACL_INT32, 0, 0);
        RunCumsumV2Test("V2_Int32_exF_revT", data, shape, 1, false, true, ACL_INT32, 0, 0);
        RunCumsumV2Test("V2_Int32_exT_revT", data, shape, 1, true, true, ACL_INT32, 0, 0);
    }

    // V2 with INT64
    {
        std::vector<int64_t> shape = {2, 4};
        std::vector<int64_t> data = {1,2,3,4, 5,6,7,8};
        RunCumsumV2Test("V2_Int64_exF_revF", data, shape, 1, false, false, ACL_INT64, 0, 0);
        RunCumsumV2Test("V2_Int64_exT_revT", data, shape, 1, true, true, ACL_INT64, 0, 0);
    }

    // V2 with FLOAT16
    {
        std::vector<int64_t> shape = {2, 4};
        std::vector<float> data = {1,2,3,4, 5,6,7,8};
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;

        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(8, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);

        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, true, true, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            std::vector<float> result(8, 0.0f);
            aclrtMemcpy(result.data(), 8 * sizeof(float), outDevAddr, 8 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, true, true, expected);
            bool passed = CompareResult(result, expected, 8, "FLOAT16_V2", 1e-3, 1e-3);
            LOG_PRINT("[%s] V2_Float16_exT_revT\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] V2_Float16_exT_revT: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }
}

// =============== Test Group 6: Float Tiling Keys ===============

void TestFloatTilingKeys()
{
    LOG_PRINT("\n=== Group 6: Float Tiling Keys ===\n");

    // ONEWAY (1001): N>=cacheLine, R full load, M>=coreNum
    {
        std::vector<int64_t> shape = {100, 1024};
        std::vector<float> data(100 * 1024, 1.0f);
        RunCumsumV2Test("Tiling_ONEWAY_100x1024", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // ONEWAY: M<coreNum, borrowN
    {
        std::vector<int64_t> shape = {2, 1024};
        std::vector<float> data(2 * 1024, 1.0f);
        RunCumsumV2Test("Tiling_ONEWAY_borrowN_2x1024", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // ONEWAY with FP16 (dtCast_=true)
    {
        std::vector<int64_t> shape = {2, 1024};
        std::vector<float> data(2 * 1024, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(2 * 1024, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            std::vector<float> result(2 * 1024, 0.0f);
            aclrtMemcpy(result.data(), 2 * 1024 * sizeof(float), outDevAddr, 2 * 1024 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, 2 * 1024, "FP16", 1e-3, 1e-3);
            LOG_PRINT("[%s] Tiling_ONEWAY_FP16_2x1024\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Tiling_ONEWAY_FP16_2x1024: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // ONEWAY with BF16 (dtCast_=true, BF16 path)
    {
        std::vector<int64_t> shape = {2, 1024};
        std::vector<float> data(2 * 1024, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_BF16, &self);
        std::vector<float> outData(2 * 1024, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_BF16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            std::vector<float> result(2 * 1024, 0.0f);
            aclrtMemcpy(result.data(), 2 * 1024 * sizeof(float), outDevAddr, 2 * 1024 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, 2 * 1024, "BF16", 1e-2, 1e-2);
            LOG_PRINT("[%s] Tiling_ONEWAY_BF16_2x1024\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Tiling_ONEWAY_BF16_2x1024: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // UB_SS_ONEWAY (1011): large R, M >= coreNum/2
    {
        std::vector<int64_t> shape = {50, 200000};
        std::vector<float> data(50 * 200000, 1.0f);
        RunCumsumV2Test("Tiling_UB_SS_ONEWAY_50x200k", data, shape, 1, false, false, ACL_FLOAT, 5e-3, 1e-4);
    }

    // CORE_SS_ONEWAY (1101): small M, large R, borrowR
    {
        std::vector<int64_t> shape = {3, 200000, 128};
        std::vector<float> data(3 * 200000 * 128, 1.0f);
        RunCumsumV2Test("Tiling_CORE_SS_ONEWAY_3x200kx128", data, shape, 1, false, false, ACL_FLOAT, 5e-3, 1e-4);
    }

    // CORE_SS_UB_SS_ONEWAY (1111): small M, very large R, borrowR, R not full in UB
    {
        std::vector<int64_t> shape = {3, 1000000, 128};
        std::vector<float> data(3LL * 1000000 * 128, 1.0f);
        RunCumsumV2Test("Tiling_CORE_SS_UB_SS_ONEWAY", data, shape, 1, false, false, ACL_FLOAT, 1e-2, 1e-3);
    }

    // TWOWAY (1002): small N, large R, FP16
    {
        std::vector<int64_t> shape = {100, 65536, 2};
        std::vector<float> data(100 * 65536 * 2, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(100 * 65536 * 2, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            int64_t total = 100 * 65536 * 2;
            std::vector<float> result(total, 0.0f);
            aclrtMemcpy(result.data(), total * sizeof(float), outDevAddr, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, total, "FP16_TWOWAY", 1e-3, 1e-2);
            LOG_PRINT("[%s] Tiling_TWOWAY_FP16_100x65kx2\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Tiling_TWOWAY_FP16_100x65kx2: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // UB_SS_TWOWAY (1012): small N, large R, not full load, FP16
    {
        std::vector<int64_t> shape = {50, 200000, 2};
        std::vector<float> data(50 * 200000 * 2, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(50 * 200000 * 2, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            int64_t total = 50 * 200000 * 2;
            std::vector<float> result(total, 0.0f);
            aclrtMemcpy(result.data(), total * sizeof(float), outDevAddr, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, total, "FP16_UB_SS_TWOWAY", 1e-3, 1e-2);
            LOG_PRINT("[%s] Tiling_UB_SS_TWOWAY_FP16_50x200kx2\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Tiling_UB_SS_TWOWAY_FP16_50x200kx2: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // CORE_SS_TWOWAY (1102)
    {
        std::vector<int64_t> shape = {3, 200000, 2};
        std::vector<float> data(3 * 200000 * 2, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(3 * 200000 * 2, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            int64_t total = 3 * 200000 * 2;
            std::vector<float> result(total, 0.0f);
            aclrtMemcpy(result.data(), total * sizeof(float), outDevAddr, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, total, "FP16_CORE_SS_TWOWAY", 1e-3, 1e-2);
            LOG_PRINT("[%s] Tiling_CORE_SS_TWOWAY_FP16_3x200kx2\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Tiling_CORE_SS_TWOWAY_FP16_3x200kx2: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // CORE_SS_UB_SS_TWOWAY (1112)
    {
        std::vector<int64_t> shape = {3, 1000000, 2};
        std::vector<float> data(3LL * 1000000 * 2, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(3LL * 1000000 * 2, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            int64_t total = 3LL * 1000000 * 2;
            std::vector<float> result(total, 0.0f);
            aclrtMemcpy(result.data(), total * sizeof(float), outDevAddr, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, total, "FP16_CORE_SS_UB_SS_TWOWAY", 1e-2, 1e-2);
            LOG_PRINT("[%s] Tiling_CORE_SS_UB_SS_TWOWAY_FP16_3x1Mx2\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Tiling_CORE_SS_UB_SS_TWOWAY_FP16_3x1Mx2: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // MRNGreaterCl path: N<cl, R*N<cl, M*R*N>=cl
    {
        std::vector<int64_t> shape = {1000, 4, 8};
        std::vector<float> data(1000 * 4 * 8, 1.0f);
        RunCumsumV2Test("Tiling_MRNGreaterCl_1000x4x8", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // MRNLesserCl path: very small total
    {
        std::vector<int64_t> shape = {4, 4, 1};
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        RunCumsumV2Test("Tiling_MRNLesserCl_4x4x1", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // Middle axis (axis != 0 and != lastDim)
    {
        std::vector<int64_t> shape = {10, 1024, 32};
        std::vector<float> data(10 * 1024 * 32, 1.0f);
        RunCumsumV2Test("Tiling_MidAxis_10x1024x32", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // dim=0 for 2D (axis=0, lenM=1, lenR=first_dim)
    {
        std::vector<int64_t> shape = {1024, 100};
        std::vector<float> data(1024 * 100, 1.0f);
        RunCumsumV2Test("Tiling_dim0_1024x100", data, shape, 0, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // borrowM path: TWOWAY with M>=coreNum → CalcBorrowM
    {
        std::vector<int64_t> shape = {1000, 65536, 1};
        std::vector<float> data(1000 * 65536, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(1000 * 65536, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            int64_t total = 1000 * 65536;
            std::vector<float> result(total, 0.0f);
            aclrtMemcpy(result.data(), total * sizeof(float), outDevAddr, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, total, "FP16_BORROWM", 1e-3, 1e-2);
            LOG_PRINT("[%s] Tiling_BorrowM_FP16_1000x65kx1\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Tiling_BorrowM_FP16_1000x65kx1: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }
}

// =============== Test Group 7: Int Tiling Keys ===============

void TestIntTilingKeys()
{
    LOG_PRINT("\n=== Group 7: Int Tiling Keys ===\n");

    // INT8 (dtypeSize=1 → vlSize_/=2 branch)
    {
        std::vector<int64_t> shape = {100, 1024};
        std::vector<int8_t> data(100 * 1024);
        for (int i = 0; i < 100 * 1024; i++) data[i] = static_cast<int8_t>((i % 10) + 1);
        RunCumsumV2Test("IntTiling_INT8_100x1024", data, shape, 1, false, false, ACL_INT8, 0, 0);
    }

    // UINT8 (dtypeSize=1)
    {
        std::vector<int64_t> shape = {100, 1024};
        std::vector<uint8_t> data(100 * 1024);
        for (int i = 0; i < 100 * 1024; i++) data[i] = static_cast<uint8_t>((i % 10) + 1);
        RunCumsumV2Test("IntTiling_UINT8_100x1024", data, shape, 1, false, false, ACL_UINT8, 0, 0);
    }

    // INT32 axis=0
    {
        std::vector<int64_t> shape = {1024, 100};
        std::vector<int32_t> data(1024 * 100);
        for (int i = 0; i < 1024 * 100; i++) data[i] = (i % 5) + 1;
        RunCumsumV2Test("IntTiling_INT32_axis0_1024x100", data, shape, 0, false, false, ACL_INT32, 0, 0);
    }

    // INT32 axis=last (middle axis position)
    {
        std::vector<int64_t> shape = {10, 1024, 100};
        std::vector<int32_t> data(10 * 1024 * 100);
        for (int i = 0; i < 10 * 1024 * 100; i++) data[i] = (i % 5) + 1;
        RunCumsumV2Test("IntTiling_INT32_midAxis_10x1024x100", data, shape, 1, false, false, ACL_INT32, 0, 0);
    }

    // INT64
    {
        std::vector<int64_t> shape = {100, 1024};
        std::vector<int64_t> data(100 * 1024);
        for (int i = 0; i < 100 * 1024; i++) data[i] = (i % 5) + 1;
        RunCumsumV2Test("IntTiling_INT64_100x1024", data, shape, 1, false, false, ACL_INT64, 0, 0);
    }

    // Large R for int (CUM_WITH_GROUP tiling key)
    {
        std::vector<int64_t> shape = {1, 100000};
        std::vector<int32_t> data(100000, 1);
        RunCumsumV2Test("IntTiling_INT32_1x100k_CUM_WITH_GROUP", data, shape, 1, false, false, ACL_INT32, 0, 0);
    }

    // INT32 with exclusive/reverse
    {
        std::vector<int64_t> shape = {4, 64};
        std::vector<int32_t> data(256);
        for (int i = 0; i < 256; i++) data[i] = (i % 10) + 1;
        RunCumsumV2Test("IntTiling_INT32_exT_revF", data, shape, 1, true, false, ACL_INT32, 0, 0);
        RunCumsumV2Test("IntTiling_INT32_exF_revT", data, shape, 1, false, true, ACL_INT32, 0, 0);
        RunCumsumV2Test("IntTiling_INT32_exT_revT", data, shape, 1, true, true, ACL_INT32, 0, 0);
    }
}

// =============== Test Group 8: Precision Analysis ===============

void TestPrecisionAnalysis()
{
    LOG_PRINT("\n=== Group 8: Precision Analysis ===\n");

    // P1: Error accumulation - long sequence of small values (FLOAT)
    {
        std::vector<int64_t> shape = {1, 100000};
        std::vector<float> data(100000, 1e-5f);
        RunCumsumV2Test("Precision_FloatAccum_100k_small", data, shape, 1, false, false, ACL_FLOAT, 1e-4, 1e-4);
    }

    // P2: FP16 error accumulation
    {
        std::vector<int64_t> shape = {1, 10000};
        std::vector<float> data(10000, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(10000, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsumV2(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            std::vector<float> result(10000, 0.0f);
            aclrtMemcpy(result.data(), 10000 * sizeof(float), outDevAddr, 10000 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);

            double maxAbsErr = 0.0;
            int64_t maxErrIdx = 0;
            for (int64_t i = 0; i < 10000; i++) {
                double absErr = std::fabs(static_cast<double>(result[i]) - static_cast<double>(expected[i]));
                if (absErr > maxAbsErr) { maxAbsErr = absErr; maxErrIdx = i; }
            }
            LOG_PRINT("  FP16 accumulation: max_abs_err=%.6e at idx=%ld (actual=%.4f, expected=%.4f)\n",
                      maxAbsErr, maxErrIdx, static_cast<double>(result[maxErrIdx]), static_cast<double>(expected[maxErrIdx]));

            bool passed = CompareResult(result, expected, 10000, "FP16_ACCUM", 1e-2, 1e-1);
            LOG_PRINT("[%s] Precision_FP16_Accumulation_10k\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Precision_FP16_Accumulation_10k: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }

    // P4: Alternating positive/negative cancellation
    {
        std::vector<int64_t> shape = {1, 1000};
        std::vector<float> data(1000);
        for (int i = 0; i < 1000; i++) data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        RunCumsumV2Test("Precision_Cancellation_alternating", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // P5: Large then small (magnitude mixing)
    {
        std::vector<int64_t> shape = {1, 100};
        std::vector<float> data(100);
        for (int i = 0; i < 100; i++) {
            if (i % 4 == 0) data[i] = 1e8f;
            else if (i % 4 == 1) data[i] = -1e8f;
            else if (i % 4 == 2) data[i] = 1.0f;
            else data[i] = -1.0f;
        }
        RunCumsumV2Test("Precision_MagnitudeMix", data, shape, 1, false, false, ACL_FLOAT, 1e-4, 1e-4);
    }

    // P6-P8: Exclusive/reverse boundary verification
    {
        std::vector<int64_t> shape = {1, 5};
        std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
        // exclusive=false, reverse=false: [10, 30, 60, 100, 150]
        RunCumsumV2Test("Precision_Boundary_exF_revF", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
        // exclusive=true, reverse=false: [0, 10, 30, 60, 100]
        RunCumsumV2Test("Precision_Boundary_exT_revF", data, shape, 1, true, false, ACL_FLOAT, 1e-5, 1e-6);
        // exclusive=false, reverse=true: [150, 140, 120, 90, 50]
        RunCumsumV2Test("Precision_Boundary_exF_revT", data, shape, 1, false, true, ACL_FLOAT, 1e-5, 1e-6);
        // exclusive=true, reverse=true: [140, 120, 90, 50, 0]
        RunCumsumV2Test("Precision_Boundary_exT_revT", data, shape, 1, true, true, ACL_FLOAT, 1e-5, 1e-6);
    }

    // P9: Single element with exclusive
    {
        std::vector<int64_t> shape = {1, 1};
        std::vector<float> data = {42.0f};
        RunCumsumV2Test("Precision_SingleElem_exT", data, shape, 1, true, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // P14: negative dim
    {
        std::vector<int64_t> shape = {3, 4};
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        RunCumsumV1Test("Precision_NegDim_minus1", data, shape, -1, ACL_FLOAT, 1e-5, 1e-6);
        RunCumsumV1Test("Precision_NegDim_minus2", data, shape, -2, ACL_FLOAT, 1e-5, 1e-6);
    }

    // Decimal representation: 0.1 cannot be exactly represented
    {
        std::vector<int64_t> shape = {1, 1000};
        std::vector<float> data(1000, 0.1f);
        RunCumsumV2Test("Precision_Decimal_0.1_x1000", data, shape, 1, false, false, ACL_FLOAT, 1e-4, 1e-4);
    }
}

// =============== Test Group 9: Cube Support (V1 only) ===============

void TestCubeSupport()
{
    LOG_PRINT("\n=== Group 9: Cube Support (V1 only) ===\n");

    // C1: Cube path triggered - shape meets criteria on 910_93
    // Requires: dim==lastDim, batchNum>=12800, channelNum>=512, FLOAT/FLOAT16/BF16
    {
        std::vector<int64_t> shape = {12800, 512};
        std::vector<float> data(12800 * 512, 1.0f);
        RunCumsumV1Test("Cube_12800x512_Float", data, shape, 1, ACL_FLOAT, 5e-3, 1e-4);
    }

    // C3: Cube not triggered (batchNum too small)
    {
        std::vector<int64_t> shape = {100, 512};
        std::vector<float> data(100 * 512, 1.0f);
        RunCumsumV1Test("Cube_NotMet_batchSmall", data, shape, 1, ACL_FLOAT, 1e-5, 1e-6);
    }

    // C4: Cube not triggered (channelNum too small)
    {
        std::vector<int64_t> shape = {12800, 100};
        std::vector<float> data(12800 * 100, 1.0f);
        RunCumsumV1Test("Cube_NotMet_channelSmall", data, shape, 1, ACL_FLOAT, 1e-5, 1e-6);
    }

    // Cube with FP16
    {
        std::vector<int64_t> shape = {12800, 512};
        std::vector<float> data(12800 * 512, 1.0f);
        void* selfDevAddr = nullptr;
        void* outDevAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;
        CreateAclTensor(data, shape, &selfDevAddr, ACL_FLOAT16, &self);
        std::vector<float> outData(12800 * 512, 0.0f);
        CreateAclTensor(outData, shape, &outDevAddr, ACL_FLOAT16, &out);
        auto ret = aclnnCumsumGetWorkspaceSize(self, 1, ACL_FLOAT16, out, &workspaceSize, &executor);
        if (ret == ACLNN_SUCCESS) {
            if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCumsum(workspaceAddr, workspaceSize, executor, g_stream);
            aclrtSynchronizeStream(g_stream);
            int64_t total = 12800 * 512;
            std::vector<float> result(total, 0.0f);
            aclrtMemcpy(result.data(), total * sizeof(float), outDevAddr, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            std::vector<float> expected;
            CpuCumsum(data, shape, 1, false, false, expected);
            bool passed = CompareResult(result, expected, total, "FP16_CUBE", 1e-3, 1e-2);
            LOG_PRINT("[%s] Cube_12800x512_FP16\n", passed ? "PASS" : "FAIL");
            if (passed) g_passCount++; else g_failCount++;
        } else {
            LOG_PRINT("[FAIL] Cube_12800x512_FP16: API returned %d\n", ret);
            g_failCount++;
        }
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self); aclDestroyTensor(out);
        if (selfDevAddr) aclrtFree(selfDevAddr);
        if (outDevAddr) aclrtFree(outDevAddr);
    }
}

// =============== Test Group 10: Multi-dimensional ===============

void TestMultiDimensional()
{
    LOG_PRINT("\n=== Group 10: Multi-dimensional ===\n");

    // 3D dim=0
    {
        std::vector<int64_t> shape = {3, 4, 5};
        std::vector<float> data(60);
        for (int i = 0; i < 60; i++) data[i] = static_cast<float>(i + 1);
        RunCumsumV2Test("MD_3d_dim0", data, shape, 0, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // 3D dim=1 (middle)
    {
        std::vector<int64_t> shape = {3, 4, 5};
        std::vector<float> data(60);
        for (int i = 0; i < 60; i++) data[i] = static_cast<float>(i + 1);
        RunCumsumV2Test("MD_3d_dim1", data, shape, 1, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // 3D dim=2 (last)
    {
        std::vector<int64_t> shape = {3, 4, 5};
        std::vector<float> data(60);
        for (int i = 0; i < 60; i++) data[i] = static_cast<float>(i + 1);
        RunCumsumV2Test("MD_3d_dim2", data, shape, 2, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // 4D
    {
        std::vector<int64_t> shape = {2, 3, 4, 5};
        std::vector<float> data(120);
        for (int i = 0; i < 120; i++) data[i] = static_cast<float>(i + 1);
        RunCumsumV2Test("MD_4d_dim2", data, shape, 2, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }

    // 8D (max dims boundary)
    {
        std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 4};
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        RunCumsumV2Test("MD_8dim_dim7", data, shape, 7, false, false, ACL_FLOAT, 1e-5, 1e-6);
    }
}

// =============== Main ===============

int main()
{
    auto ret = Init(g_deviceId, &g_stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    TestNegativeParamValidation();
    TestEmptyTensor();
    TestV1BasicFloat();
    TestV1DtypeVariants();
    TestV2Attributes();
    TestFloatTilingKeys();
    TestIntTilingKeys();
    TestPrecisionAnalysis();
    TestCubeSupport();
    TestMultiDimensional();

    LOG_PRINT("\n========================================\n");
    LOG_PRINT("Summary: %d passed, %d failed, %d total\n", g_passCount, g_failCount, g_passCount + g_failCount);
    LOG_PRINT("========================================\n");

    aclrtDestroyStream(g_stream);
    aclrtResetDevice(g_deviceId);
    aclFinalize();

    return (g_failCount > 0) ? 1 : 0;
}
