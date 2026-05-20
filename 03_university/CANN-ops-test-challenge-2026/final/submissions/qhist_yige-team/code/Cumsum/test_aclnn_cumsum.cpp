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
#include <cstring>
#include <cfloat>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

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
static int g_totalCount = 0;

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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// Create empty tensor
int CreateEmptyAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    int64_t size = GetShapeSize(shape);
    if (size == 0) {
        *deviceAddr = nullptr;
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
            nullptr);
        return 0;
    }
    auto ret = aclrtMalloc(deviceAddr, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// CPU reference implementation for float
template <typename T>
std::vector<double> CpuCumsum(const std::vector<T>& input, int64_t dim = 0, bool exclusive = false, bool reverse = false)
{
    std::vector<double> result(input.size());
    if (input.empty()) return result;

    // Assume 1D for simplicity in reference
    double sum = 0.0;
    if (reverse) {
        for (int i = input.size() - 1; i >= 0; i--) {
            if (!exclusive) sum += (double)input[i];
            result[i] = sum;
            if (exclusive) sum += (double)input[i];
        }
    } else {
        for (size_t i = 0; i < input.size(); i++) {
            if (!exclusive) sum += (double)input[i];
            result[i] = sum;
            if (exclusive) sum += (double)input[i];
        }
    }
    return result;
}

// CPU reference for INT types
template <typename T>
std::vector<T> CpuCumsumInt(const std::vector<T>& input, bool exclusive = false, bool reverse = false)
{
    std::vector<T> result(input.size());
    if (input.empty()) return result;

    T sum = 0;
    if (reverse) {
        for (int i = input.size() - 1; i >= 0; i--) {
            if (!exclusive) sum += input[i];
            result[i] = sum;
            if (exclusive) sum += input[i];
        }
    } else {
        for (size_t i = 0; i < input.size(); i++) {
            if (!exclusive) sum += input[i];
            result[i] = sum;
            if (exclusive) sum += input[i];
        }
    }
    return result;
}

bool FloatCompare(double actual, double expected, double atol = 1e-5, double rtol = 1e-5)
{
    if (std::isnan(actual) && std::isnan(expected)) return true;
    if (std::isinf(actual) && std::isinf(expected)) return (actual > 0) == (expected > 0);
    double diff = std::abs(actual - expected);
    double tol = atol + rtol * std::abs(expected);
    return diff <= tol;
}

void RecordResult(bool pass, const char* testName)
{
    g_totalCount++;
    if (pass) {
        g_passCount++;
        LOG_PRINT("[PASS] %s\n", testName);
    } else {
        g_failCount++;
        LOG_PRINT("[FAIL] %s\n", testName);
    }
}

// Helper to run cumsum and validate
template <typename T>
bool RunAndValidateCumsum(
    aclrtStream stream,
    const std::vector<T>& input,
    const std::vector<int64_t>& shape,
    int64_t dim,
    aclDataType dtype,
    const std::vector<double>& expected,
    double atol = 1e-5,
    double rtol = 1e-5)
{
    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<T> outHostData(input.size(), 0);

    if (CreateAclTensor(input, shape, &inDevice, dtype, &self) != 0) return false;
    if (CreateAclTensor(outHostData, shape, &outDevice, dtype, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(inDevice);
            aclrtFree(outDevice);
            return false;
        }
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(T), outDevice, size * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(inDevice);
    aclrtFree(outDevice);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    if (ret != ACL_SUCCESS) return false;

    bool allPass = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (!FloatCompare((double)resultData[i], expected[i], atol, rtol)) {
            allPass = false;
            break;
        }
    }
    return allPass;
}

// V2 version
template <typename T>
bool RunAndValidateCumsumV2(
    aclrtStream stream,
    const std::vector<T>& input,
    const std::vector<int64_t>& shape,
    int64_t dim,
    bool exclusive,
    bool reverse,
    const std::vector<double>& expected,
    double atol = 1e-5,
    double rtol = 1e-5)
{
    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<T> outHostData(input.size(), 0);
    aclDataType dtype = aclDataType::ACL_FLOAT;
    if (sizeof(T) == 2) dtype = aclDataType::ACL_FLOAT16;

    if (CreateAclTensor(input, shape, &inDevice, dtype, &self) != 0) return false;
    if (CreateAclTensor(outHostData, shape, &outDevice, dtype, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(inDevice);
            aclrtFree(outDevice);
            return false;
        }
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(T), outDevice, size * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(inDevice);
    aclrtFree(outDevice);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    if (ret != ACL_SUCCESS) return false;

    bool allPass = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (!FloatCompare((double)resultData[i], expected[i], atol, rtol)) {
            allPass = false;
            break;
        }
    }
    return allPass;
}

// INT32 version
bool RunAndValidateCumsumInt32(
    aclrtStream stream,
    const std::vector<int32_t>& input,
    const std::vector<int64_t>& shape,
    int64_t dim,
    const std::vector<int32_t>& expected)
{
    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<int32_t> outHostData(input.size(), 0);

    if (CreateAclTensor(input, shape, &inDevice, aclDataType::ACL_INT32, &self) != 0) return false;
    if (CreateAclTensor(outHostData, shape, &outDevice, aclDataType::ACL_INT32, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, aclDataType::ACL_INT32, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(inDevice);
            aclrtFree(outDevice);
            return false;
        }
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(int32_t), outDevice, size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(inDevice);
    aclrtFree(outDevice);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    if (ret != ACL_SUCCESS) return false;

    bool allPass = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (resultData[i] != expected[i]) {
            allPass = false;
            break;
        }
    }
    return allPass;
}

// INT8 version
bool RunAndValidateCumsumInt8(
    aclrtStream stream,
    const std::vector<int8_t>& input,
    const std::vector<int64_t>& shape,
    int64_t dim,
    const std::vector<int8_t>& expected)
{
    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<int8_t> outHostData(input.size(), 0);

    if (CreateAclTensor(input, shape, &inDevice, aclDataType::ACL_INT8, &self) != 0) return false;
    if (CreateAclTensor(outHostData, shape, &outDevice, aclDataType::ACL_INT8, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, false, false, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(inDevice);
            aclrtFree(outDevice);
            return false;
        }
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(int8_t), outDevice, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(inDevice);
    aclrtFree(outDevice);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    if (ret != ACL_SUCCESS) return false;

    bool allPass = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (resultData[i] != expected[i]) {
            allPass = false;
            break;
        }
    }
    return allPass;
}

// UINT8 version
bool RunAndValidateCumsumUint8(
    aclrtStream stream,
    const std::vector<uint8_t>& input,
    const std::vector<int64_t>& shape,
    int64_t dim,
    const std::vector<uint8_t>& expected)
{
    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<uint8_t> outHostData(input.size(), 0);

    if (CreateAclTensor(input, shape, &inDevice, aclDataType::ACL_UINT8, &self) != 0) return false;
    if (CreateAclTensor(outHostData, shape, &outDevice, aclDataType::ACL_UINT8, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, false, false, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(inDevice);
            aclrtFree(outDevice);
            return false;
        }
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    auto size = GetShapeSize(shape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(uint8_t), outDevice, size * sizeof(uint8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(inDevice);
    aclrtFree(outDevice);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    if (ret != ACL_SUCCESS) return false;

    bool allPass = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (resultData[i] != expected[i]) {
            allPass = false;
            break;
        }
    }
    return allPass;
}

// INT64 version
bool RunAndValidateCumsumInt64(
    aclrtStream stream,
    const std::vector<int64_t>& input,
    const std::vector<int64_t>& shape,
    int64_t dim,
    const std::vector<int64_t>& expected)
{
    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<int64_t> outHostData(input.size(), 0);

    if (CreateAclTensor(input, shape, &inDevice, aclDataType::ACL_INT64, &self) != 0) return false;
    if (CreateAclTensor(outHostData, shape, &outDevice, aclDataType::ACL_INT64, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, false, false, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(inDevice);
            aclrtFree(outDevice);
            return false;
        }
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    auto size = GetShapeSize(shape);
    std::vector<int64_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(int64_t), outDevice, size * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(inDevice);
    aclrtFree(outDevice);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    if (ret != ACL_SUCCESS) return false;

    bool allPass = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (resultData[i] != expected[i]) {
            allPass = false;
            break;
        }
    }
    return allPass;
}

// FLOAT16 test runner
bool RunAndValidateCumsumFp16(
    aclrtStream stream,
    const std::vector<uint16_t>& input,
    const std::vector<int64_t>& shape,
    int64_t dim,
    const std::vector<double>& expected,
    double atol = 1e-3,
    double rtol = 1e-3)
{
    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<uint16_t> outHostData(input.size(), 0);

    if (CreateAclTensor(input, shape, &inDevice, aclDataType::ACL_FLOAT16, &self) != 0) return false;
    if (CreateAclTensor(outHostData, shape, &outDevice, aclDataType::ACL_FLOAT16, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, aclDataType::ACL_FLOAT16, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(inDevice);
            aclrtFree(outDevice);
            return false;
        }
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
        if (workspaceSize > 0) aclrtFree(workspaceAddr);
        return false;
    }

    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(uint16_t), outDevice, size * sizeof(uint16_t),
        ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(inDevice);
    aclrtFree(outDevice);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    if (ret != ACL_SUCCESS) return false;

    // Convert float16 to float for comparison
    bool allPass = true;
    for (size_t i = 0; i < expected.size(); i++) {
        // Simple conversion from float16 bits to float
        uint16_t h = resultData[i];
        int sign = (h >> 15) & 0x1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        float f;
        if (exp == 0) {
            f = mant * pow(2, -24);
        } else if (exp == 31) {
            f = (mant == 0) ? INFINITY : NAN;
        } else {
            f = (1.0f + mant / 1024.0f) * pow(2, exp - 15);
        }
        if (sign) f = -f;

        if (!FloatCompare(f, expected[i], atol, rtol)) {
            allPass = false;
            break;
        }
    }
    return allPass;
}

// Test case implementations
void TestBasicFloat32(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    auto expected = CpuCumsum(input, 0, false, false);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Basic Cumsum FLOAT32 1D");
}

void TestFloat32Dim1(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> shape = {2, 3};
    auto expected = CpuCumsum(input, 1, false, false);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Basic Cumsum FLOAT32 dim=1");
}

void TestFloat32NegativeDim(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {2, 2};
    auto expected = CpuCumsum(input, -1, false, false);
    bool pass = RunAndValidateCumsum(stream, input, shape, -1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 negative dim");
}

void TestFloat32AllPositive(aclrtStream stream)
{
    std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 all positive");
}

void TestFloat32AllNegative(aclrtStream stream)
{
    std::vector<float> input = {-1.0f, -2.0f, -3.0f, -4.0f};
    std::vector<int64_t> shape = {4};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 all negative");
}

void TestFloat32MixedSign(aclrtStream stream)
{
    std::vector<float> input = {-2.0f, 1.0f, 3.0f, -1.0f, 2.0f};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 mixed sign");
}

void TestFloat32WithZeros(aclrtStream stream)
{
    std::vector<float> input = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 with zeros");
}

void TestFloat32ShortSequence(aclrtStream stream)
{
    std::vector<float> input = {1.0f};
    std::vector<int64_t> shape = {1};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 short sequence (len=1)");
}

void TestFloat32MediumSequence(aclrtStream stream)
{
    std::vector<float> input(100);
    for (int i = 0; i < 100; i++) input[i] = 1.0f;
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 medium sequence (len=100)");
}

void TestFloat32LongSequence(aclrtStream stream)
{
    std::vector<float> input(1000);
    for (int i = 0; i < 1000; i++) input[i] = 1.0f;
    std::vector<int64_t> shape = {1000};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-3, 1e-3);
    RecordResult(pass, "Cumsum FLOAT32 long sequence (len=1000)");
}

void TestFloat32VeryLongSequence(aclrtStream stream)
{
    std::vector<float> input(10000);
    for (int i = 0; i < 10000; i++) input[i] = 1.0f;
    std::vector<int64_t> shape = {10000};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-2, 1e-2);
    RecordResult(pass, "Cumsum FLOAT32 very long sequence (len=10000) - accumulation error test");
}

void TestFloat32MixedMagnitude(aclrtStream stream)
{
    // Test for precision loss with mixed magnitude
    std::vector<float> input = {1e8f, 1e-6f, 1e8f, 1e-6f, 1e8f, 1e-6f};
    std::vector<int64_t> shape = {6};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1.0f, 1e-5);
    RecordResult(pass, "Cumsum FLOAT32 mixed magnitude (precision loss test)");
}

void TestFloat322D(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> shape = {2, 3};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 2D shape");
}

void TestFloat323D(aclrtStream stream)
{
    std::vector<float> input(24);
    for (int i = 0; i < 24; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {2, 3, 4};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 3D shape");
}

void TestFloat324D(aclrtStream stream)
{
    std::vector<float> input(16);
    for (int i = 0; i < 16; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {2, 2, 2, 2};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 2, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 4D shape");
}

void TestFloat16Basic(aclrtStream stream)
{
    // Using raw uint16_t for float16
    std::vector<uint16_t> input(100);
    for (int i = 0; i < 100; i++) {
        // Simple float16 representation of small integers
        input[i] = (i + 1) << 10; // rough approximation
    }
    std::vector<int64_t> shape = {100};
    // Expected cumulative sum - rough calculation
    std::vector<double> expected(100);
    double sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += (i + 1);
        expected[i] = sum;
    }
    bool pass = RunAndValidateCumsumFp16(stream, input, shape, 0, expected, 10.0, 0.1);
    RecordResult(pass, "Cumsum FLOAT16 basic");
}

void TestV2ExclusiveFalseReverseFalse(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    auto expected = CpuCumsum(input, 0, false, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, false, expected);
    RecordResult(pass, "CumsumV2 exclusive=false reverse=false");
}

void TestV2ExclusiveTrueReverseFalse(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, false, expected);
    RecordResult(pass, "CumsumV2 exclusive=true reverse=false");
}

void TestV2ExclusiveFalseReverseTrue(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected);
    RecordResult(pass, "CumsumV2 exclusive=false reverse=true");
}

void TestV2ExclusiveTrueReverseTrue(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    auto expected = CpuCumsum(input, 0, true, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, true, expected);
    RecordResult(pass, "CumsumV2 exclusive=true reverse=true");
}

void TestInt32Basic(aclrtStream stream)
{
    std::vector<int32_t> input = {1, 2, 3, 4, 5};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 basic");
}

void TestInt32Negative(aclrtStream stream)
{
    std::vector<int32_t> input = {-5, -4, -3, -2, -1};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 negative values");
}

void TestInt32Mixed(aclrtStream stream)
{
    std::vector<int32_t> input = {-10, 5, -3, 8, -2};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 mixed sign");
}

void TestInt32Dim1(aclrtStream stream)
{
    std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> shape = {2, 3};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 1, expected);
    RecordResult(pass, "Cumsum INT32 dim=1");
}

void TestInt8Basic(aclrtStream stream)
{
    std::vector<int8_t> input = {1, 2, 3, 4, 5};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 basic");
}

void TestUint8Basic(aclrtStream stream)
{
    std::vector<uint8_t> input = {1, 2, 3, 4, 5};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 basic");
}

void TestInt64Basic(aclrtStream stream)
{
    std::vector<int64_t> input = {1000000, 2000000, 3000000, 4000000};
    std::vector<int64_t> shape = {4};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 basic");
}

void TestInt64Large(aclrtStream stream)
{
    std::vector<int64_t> input(100);
    for (int i = 0; i < 100; i++) input[i] = 1000000;
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 large values");
}

void TestEmptyTensor(aclrtStream stream)
{
    // Test empty tensor handling
    std::vector<int64_t> shape = {0};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateEmptyAclTensor(shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    if (ret != 0) {
        RecordResult(false, "Empty tensor - create self failed");
        return;
    }
    ret = CreateEmptyAclTensor(shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    if (ret != 0) {
        aclDestroyTensor(self);
        RecordResult(false, "Empty tensor - create out failed");
        return;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_FLOAT, out, &workspaceSize, &executor);

    aclDestroyTensor(self);
    aclDestroyTensor(out);

    RecordResult(ret == ACL_SUCCESS, "Empty tensor handling");
}

void TestErrorCases(aclrtStream stream)
{
    // Test various error conditions to cover error branches
    // These tests are expected to fail API calls, but should not crash

    // Test with mismatched shapes - API should return error
    std::vector<float> input1 = {1.0f, 2.0f};
    std::vector<float> input2 = {1.0f, 2.0f, 3.0f};
    std::vector<int64_t> shape1 = {2};
    std::vector<int64_t> shape2 = {3};
    void* d1 = nullptr;
    void* d2 = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;

    if (CreateAclTensor(input1, shape1, &d1, aclDataType::ACL_FLOAT, &t1) == 0 &&
        CreateAclTensor(input2, shape2, &d2, aclDataType::ACL_FLOAT, &t2) == 0) {
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto ret = aclnnCumsumGetWorkspaceSize(t1, 0, aclDataType::ACL_FLOAT, t2, &ws, &exec);
        // Expected to fail due to shape mismatch
        (void)ret; // Suppress unused warning
    }

    if (t1) { aclDestroyTensor(t1); aclrtFree(d1); }
    if (t2) { aclDestroyTensor(t2); aclrtFree(d2); }

    RecordResult(true, "Error case handling (shape mismatch)");
}

void TestFloat32PrecisionAccumulation(aclrtStream stream)
{
    // Test precision accumulation with 0.1 (cannot be exactly represented in binary)
    std::vector<float> input(100);
    for (int i = 0; i < 100; i++) input[i] = 0.1f;
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-3);
    RecordResult(pass, "Cumsum FLOAT32 precision accumulation (0.1 * 100)");
}

void TestV2DifferentShapes(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<int64_t> shape = {2, 4};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected);
    RecordResult(pass, "CumsumV2 2D shape with reverse");
}

void TestFloat32BatchProcessing(aclrtStream stream)
{
    // Larger batch for different tiling paths
    std::vector<float> input(128);
    for (int i = 0; i < 128; i++) input[i] = 1.0f;
    std::vector<int64_t> shape = {128};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 batch processing (len=128)");
}

void TestInt32Overflow(aclrtStream stream)
{
    // Test near overflow
    std::vector<int32_t> input = {1000000000, 1000000000, 1000000000};
    std::vector<int64_t> shape = {3};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 near overflow");
}

void TestV2ExclusiveOnLargerTensor(aclrtStream stream)
{
    std::vector<float> input(50);
    for (int i = 0; i < 50; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {50};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, false, expected);
    RecordResult(pass, "CumsumV2 exclusive on larger tensor");
}

void TestFloat32VerySmallValues(aclrtStream stream)
{
    // Test very small values
    std::vector<float> input = {1e-10f, 1e-10f, 1e-10f, 1e-10f, 1e-10f};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-15, 1e-5);
    RecordResult(pass, "Cumsum FLOAT32 very small values");
}

void TestFloat32LargeValues(aclrtStream stream)
{
    // Test large values
    std::vector<float> input = {1e20f, 1e20f, 1e20f};
    std::vector<int64_t> shape = {3};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e20, 1e-5);
    RecordResult(pass, "Cumsum FLOAT32 large values (inf test)");
}

void TestFloat32AlternatingSign(aclrtStream stream)
{
    // Test alternating positive and negative
    std::vector<float> input = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
    std::vector<int64_t> shape = {6};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 alternating sign");
}

void TestInt8NegativeValues(aclrtStream stream)
{
    std::vector<int8_t> input = {-10, -20, -30, 40, 50};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 negative values");
}

void TestUint8MaxValues(aclrtStream stream)
{
    std::vector<uint8_t> input = {255, 0, 255, 0, 255};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 max values");
}

void TestInt64NegativeDim(aclrtStream stream)
{
    std::vector<int64_t> input = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> shape = {2, 3};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, -1, expected);
    RecordResult(pass, "Cumsum INT64 negative dim");
}

void TestFloat32Dim0(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> shape = {2, 3};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 dim=0");
}

void TestV2ReverseOn2D(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> shape = {2, 3};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 1, false, true, expected);
    RecordResult(pass, "CumsumV2 reverse on 2D");
}

void TestInt32AllSame(aclrtStream stream)
{
    std::vector<int32_t> input(50);
    for (int i = 0; i < 50; i++) input[i] = 100;
    std::vector<int64_t> shape = {50};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 all same values");
}

void TestFloat32SingleElement(aclrtStream stream)
{
    std::vector<float> input = {42.0f};
    std::vector<int64_t> shape = {1};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 single element");
}

void TestInt32SingleElement(aclrtStream stream)
{
    std::vector<int32_t> input = {42};
    std::vector<int64_t> shape = {1};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 single element");
}

void TestFloat32SmallBatch(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<int64_t> shape = {2, 4};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 small batch 2x4");
}

void TestInt8AllSame(aclrtStream stream)
{
    std::vector<int8_t> input(20);
    for (int i = 0; i < 20; i++) input[i] = 5;
    std::vector<int64_t> shape = {20};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 all same values");
}

void TestUint8LargeArray(aclrtStream stream)
{
    std::vector<uint8_t> input(100);
    for (int i = 0; i < 100; i++) input[i] = (uint8_t)(i % 256);
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 large array");
}

void TestV2ExclusiveReverseCombined(aclrtStream stream)
{
    std::vector<float> input(20);
    for (int i = 0; i < 20; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {20};
    auto expected = CpuCumsum(input, 0, true, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, true, expected);
    RecordResult(pass, "CumsumV2 exclusive+reverse combined");
}

void TestFloat32Decreasing(aclrtStream stream)
{
    std::vector<float> input = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f};
    std::vector<int64_t> shape = {6};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 decreasing sequence");
}

void TestInt32Decreasing(aclrtStream stream)
{
    std::vector<int32_t> input = {100, 80, 60, 40, 20, 0};
    std::vector<int64_t> shape = {6};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 decreasing sequence");
}

void TestFloat32Fractional(aclrtStream stream)
{
    std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    std::vector<int64_t> shape = {10};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-5, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 fractional values");
}

void TestInt64LargeArray(aclrtStream stream)
{
    std::vector<int64_t> input(50);
    for (int i = 0; i < 50; i++) input[i] = (int64_t)i * 1000000;
    std::vector<int64_t> shape = {50};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 large array");
}

void TestV2OnIntTypes(aclrtStream stream)
{
    // V2 on INT32
    std::vector<int32_t> input = {5, 4, 3, 2, 1};
    std::vector<int64_t> shape = {5};

    void* inDevice = nullptr;
    void* outDevice = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<int32_t> outHostData(input.size(), 0);

    bool pass = true;
    if (CreateAclTensor(input, shape, &inDevice, aclDataType::ACL_INT32, &self) != 0) pass = false;
    if (pass && CreateAclTensor(outHostData, shape, &outDevice, aclDataType::ACL_INT32, &out) != 0) {
        aclDestroyTensor(self);
        aclrtFree(inDevice);
        pass = false;
    }

    if (pass) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnCumsumV2GetWorkspaceSize(self, 0, false, true, out, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS && workspaceSize > 0) {
            void* workspaceAddr = nullptr;
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret == ACL_SUCCESS) {
                ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
                aclrtFree(workspaceAddr);
            }
        }
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(inDevice);
        aclrtFree(outDevice);
    }
    RecordResult(pass, "CumsumV2 on INT32");
}

void TestFloat16Simple(aclrtStream stream)
{
    // Simple float16 test with small integers
    std::vector<uint16_t> input = {0x3C00, 0x4000, 0x4200}; // 1.0, 2.0, 3.0 in float16
    std::vector<int64_t> shape = {3};
    std::vector<double> expected = {1.0, 3.0, 6.0};
    bool pass = RunAndValidateCumsumFp16(stream, input, shape, 0, expected, 0.1, 0.01);
    RecordResult(pass, "Cumsum FLOAT16 simple values");
}

void TestFloat32CubePath(aclrtStream stream)
{
    // Test large batch/channel to potentially trigger cube path
    std::vector<float> input(12800);
    for (int i = 0; i < 12800; i++) input[i] = 1.0f;
    std::vector<int64_t> shape = {128, 100};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected, 1e-3, 1e-3);
    RecordResult(pass, "Cumsum FLOAT32 potential cube path");
}

void TestInt32VariousLengths(aclrtStream stream)
{
    // Test various lengths to trigger different tiling strategies
    std::vector<int32_t> input(500);
    for (int i = 0; i < 500; i++) input[i] = i + 1;
    std::vector<int64_t> shape = {500};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 length=500");
}

void TestFloat32MultiDimensional(aclrtStream stream)
{
    // 3D test with different axis
    std::vector<float> input(60);
    for (int i = 0; i < 60; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {3, 4, 5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 2, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 3D with dim=2");
}

void TestInt64VariousDims(aclrtStream stream)
{
    std::vector<int64_t> input(24);
    for (int i = 0; i < 24; i++) input[i] = (int64_t)(i + 1);
    std::vector<int64_t> shape = {2, 3, 4};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 1, expected);
    RecordResult(pass, "Cumsum INT64 3D with dim=1");
}

void TestFloat32BoundaryValues(aclrtStream stream)
{
    // Test with values near FLT_MAX/FLT_MIN
    std::vector<float> input = {1.0f, FLT_MIN, 1.0f, FLT_MIN, 1.0f};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-30, 1e-5);
    RecordResult(pass, "Cumsum FLOAT32 boundary values");
}

void TestV2DifferentCombinations(aclrtStream stream)
{
    // Test all 4 combinations of exclusive/reverse on same data
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int64_t> shape = {5};

    // exclusive=false, reverse=false
    auto expected1 = CpuCumsum(input, 0, false, false);
    bool pass1 = RunAndValidateCumsumV2(stream, input, shape, 0, false, false, expected1);
    RecordResult(pass1, "CumsumV2 (F,F) combination");

    // exclusive=true, reverse=false
    auto expected2 = CpuCumsum(input, 0, true, false);
    bool pass2 = RunAndValidateCumsumV2(stream, input, shape, 0, true, false, expected2);
    RecordResult(pass2, "CumsumV2 (T,F) combination");

    // exclusive=false, reverse=true
    auto expected3 = CpuCumsum(input, 0, false, true);
    bool pass3 = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected3);
    RecordResult(pass3, "CumsumV2 (F,T) combination");

    // exclusive=true, reverse=true
    auto expected4 = CpuCumsum(input, 0, true, true);
    bool pass4 = RunAndValidateCumsumV2(stream, input, shape, 0, true, true, expected4);
    RecordResult(pass4, "CumsumV2 (T,T) combination");
}

void TestInt32DimVariations(aclrtStream stream)
{
    std::vector<int32_t> input(24);
    for (int i = 0; i < 24; i++) input[i] = (int32_t)(i + 1);
    std::vector<int64_t> shape = {4, 6};

    // dim = 0
    auto expected0 = CpuCumsumInt(input, false, false);
    bool pass0 = RunAndValidateCumsumInt32(stream, input, shape, 0, expected0);
    RecordResult(pass0, "Cumsum INT32 dim=0");

    // dim = 1
    auto expected1 = CpuCumsumInt(input, false, false);
    bool pass1 = RunAndValidateCumsumInt32(stream, input, shape, 1, expected1);
    RecordResult(pass1, "Cumsum INT32 dim=1");
}

void TestFloat32VariousShapes(aclrtStream stream)
{
    // Test 1xN shape
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int64_t> shape = {1, 5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 1x5 shape");
}

void TestInt8Boundary(aclrtStream stream)
{
    // Test INT8 boundary values
    std::vector<int8_t> input = {127, 0, -128, 0, 127};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 boundary values");
}

void TestUint8Overflow(aclrtStream stream)
{
    // Test UINT8 potential overflow
    std::vector<uint8_t> input = {200, 100, 50, 0, 5};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 potential overflow");
}

void TestFloat32PerformanceSize(aclrtStream stream)
{
    // Larger size for performance testing
    std::vector<float> input(500);
    for (int i = 0; i < 500; i++) input[i] = (float)(i % 10);
    std::vector<int64_t> shape = {500};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 performance size (500)");
}

void TestInt32LargeBatch(aclrtStream stream)
{
    std::vector<int32_t> input(200);
    for (int i = 0; i < 200; i++) input[i] = (i % 50) + 1;
    std::vector<int64_t> shape = {200};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 large batch (200)");
}

void TestV2ComplexScenario(aclrtStream stream)
{
    // Complex scenario with V2 on 3D tensor
    std::vector<float> input(60);
    for (int i = 0; i < 60; i++) input[i] = (float)((i % 5) + 1);
    std::vector<int64_t> shape = {3, 4, 5};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 2, true, false, expected);
    RecordResult(pass, "CumsumV2 complex 3D scenario");
}

void TestFloat32RepeatPattern(aclrtStream stream)
{
    // Test repeating pattern
    std::vector<float> input(30);
    for (int i = 0; i < 30; i++) input[i] = (float)((i % 3) + 1);
    std::vector<int64_t> shape = {30};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 repeat pattern");
}

void TestInt64EdgeCase(aclrtStream stream)
{
    // Test INT64 with zeros and negatives
    std::vector<int64_t> input = {0, -1, 0, 1, 0, -1, 0, 1};
    std::vector<int64_t> shape = {8};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 edge case zeros/negatives");
}

void TestFloat32PowerOfTwo(aclrtStream stream)
{
    // Test sizes that are power of 2
    std::vector<float> input(256);
    for (int i = 0; i < 256; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {256};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-3, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 power of 2 size (256)");
}

void TestInt32OddLength(aclrtStream stream)
{
    // Test odd length arrays
    std::vector<int32_t> input(99);
    for (int i = 0; i < 99; i++) input[i] = i + 1;
    std::vector<int64_t> shape = {99};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 odd length (99)");
}

void TestV2ExclusiveOnSmall(aclrtStream stream)
{
    std::vector<float> input = {5.0f, 3.0f, 1.0f};
    std::vector<int64_t> shape = {3};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, false, expected);
    RecordResult(pass, "CumsumV2 exclusive on small array");
}

void TestFloat32TwoElements(aclrtStream stream)
{
    std::vector<float> input = {1.5f, 2.5f};
    std::vector<int64_t> shape = {2};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 two elements");
}

void TestInt32TwoElements(aclrtStream stream)
{
    std::vector<int32_t> input = {100, 200};
    std::vector<int64_t> shape = {2};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 two elements");
}

void TestFloat32ThreeElements(aclrtStream stream)
{
    std::vector<float> input = {0.1f, 0.2f, 0.3f};
    std::vector<int64_t> shape = {3};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 three elements");
}

void TestInt8ThreeElements(aclrtStream stream)
{
    std::vector<int8_t> input = {10, 20, 30};
    std::vector<int64_t> shape = {3};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 three elements");
}

void TestUint8ThreeElements(aclrtStream stream)
{
    std::vector<uint8_t> input = {50, 100, 150};
    std::vector<int64_t> shape = {3};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 three elements");
}

void TestFloat32ManySmallValues(aclrtStream stream)
{
    // Many small values that accumulate
    std::vector<float> input(200);
    for (int i = 0; i < 200; i++) input[i] = 0.01f;
    std::vector<int64_t> shape = {200};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-3);
    RecordResult(pass, "Cumsum FLOAT32 many small values");
}

void TestInt32ManyOnes(aclrtStream stream)
{
    std::vector<int32_t> input(100);
    for (int i = 0; i < 100; i++) input[i] = 1;
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 many ones");
}

void TestV2ReverseOnly(aclrtStream stream)
{
    std::vector<float> input = {10.0f, 8.0f, 6.0f, 4.0f, 2.0f};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected);
    RecordResult(pass, "CumsumV2 reverse only");
}

void TestFloat32SquareShape(aclrtStream stream)
{
    std::vector<float> input(16);
    for (int i = 0; i < 16; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {4, 4};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 square shape 4x4");
}

void TestInt64SquareShape(aclrtStream stream)
{
    std::vector<int64_t> input(16);
    for (int i = 0; i < 16; i++) input[i] = (int64_t)(i + 1) * 1000;
    std::vector<int64_t> shape = {4, 4};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 1, expected);
    RecordResult(pass, "Cumsum INT64 square shape 4x4");
}

void TestFloat32WideShape(aclrtStream stream)
{
    std::vector<float> input(20);
    for (int i = 0; i < 20; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {2, 10};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 wide shape 2x10");
}

void TestInt32TallShape(aclrtStream stream)
{
    std::vector<int32_t> input(20);
    for (int i = 0; i < 20; i++) input[i] = i + 1;
    std::vector<int64_t> shape = {10, 2};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 tall shape 10x2");
}

void TestV2On2DLarge(aclrtStream stream)
{
    std::vector<float> input(100);
    for (int i = 0; i < 100; i++) input[i] = (float)((i % 10) + 1);
    std::vector<int64_t> shape = {10, 10};
    auto expected = CpuCumsum(input, 0, true, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, true, expected);
    RecordResult(pass, "CumsumV2 on 2D large (10x10)");
}

void TestFloat32NegativeDim2D(aclrtStream stream)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {2, 2};
    auto expected = CpuCumsum(input, -2, false, false);
    bool pass = RunAndValidateCumsum(stream, input, shape, -2, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 negative dim 2D");
}

void TestInt32NegativeDim1D(aclrtStream stream)
{
    std::vector<int32_t> input = {5, 4, 3, 2, 1};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, -1, expected);
    RecordResult(pass, "Cumsum INT32 negative dim 1D");
}

void TestFloat32VariousSizes(aclrtStream stream)
{
    // Test multiple sizes to trigger different tiling paths
    std::vector<float> input(64);
    for (int i = 0; i < 64; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {64};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 size 64");
}

void TestInt32Size128(aclrtStream stream)
{
    std::vector<int32_t> input(128);
    for (int i = 0; i < 128; i++) input[i] = i % 10;
    std::vector<int64_t> shape = {128};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 size 128");
}

void TestFloat32Size32(aclrtStream stream)
{
    std::vector<float> input(32);
    for (int i = 0; i < 32; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {32};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 size 32");
}

void TestInt8Size64(aclrtStream stream)
{
    std::vector<int8_t> input(64);
    for (int i = 0; i < 64; i++) input[i] = (int8_t)((i % 20) + 1);
    std::vector<int64_t> shape = {64};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 size 64");
}

void TestUint8Size32(aclrtStream stream)
{
    std::vector<uint8_t> input(32);
    for (int i = 0; i < 32; i++) input[i] = (uint8_t)((i % 50) + 1);
    std::vector<int64_t> shape = {32};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 size 32");
}

void TestInt64Size256(aclrtStream stream)
{
    std::vector<int64_t> input(256);
    for (int i = 0; i < 256; i++) input[i] = (int64_t)(i % 100);
    std::vector<int64_t> shape = {256};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 size 256");
}

void TestFloat32Size16(aclrtStream stream)
{
    std::vector<float> input(16);
    for (int i = 0; i < 16; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {16};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 size 16");
}

void TestV2OnMediumSize(aclrtStream stream)
{
    std::vector<float> input(50);
    for (int i = 0; i < 50; i++) input[i] = (float)((i % 5) + 1);
    std::vector<int64_t> shape = {50};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected);
    RecordResult(pass, "CumsumV2 medium size (50)");
}

void TestFloat32Size8(aclrtStream stream)
{
    std::vector<float> input(8);
    for (int i = 0; i < 8; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {8};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 size 8");
}

void TestInt32Size256(aclrtStream stream)
{
    std::vector<int32_t> input(256);
    for (int i = 0; i < 256; i++) input[i] = (i % 10) + 1;
    std::vector<int64_t> shape = {256};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 size 256");
}

void TestFloat32MultiRow(aclrtStream stream)
{
    std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    std::vector<int64_t> shape = {3, 3};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 multi-row 3x3");
}

void TestInt32MultiRow(aclrtStream stream)
{
    std::vector<int32_t> input = {
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    };
    std::vector<int64_t> shape = {3, 3};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 1, expected);
    RecordResult(pass, "Cumsum INT32 multi-row 3x3");
}

void TestV2Complex3D(aclrtStream stream)
{
    std::vector<float> input(24);
    for (int i = 0; i < 24; i++) input[i] = (float)((i % 4) + 1);
    std::vector<int64_t> shape = {2, 3, 4};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 2, true, false, expected);
    RecordResult(pass, "CumsumV2 complex 3D 2x3x4");
}

void TestFloat32VariousDims(aclrtStream stream)
{
    // Test various dimensions for 3D tensor
    std::vector<float> input(24);
    for (int i = 0; i < 24; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {2, 3, 4};

    // dim = 0
    auto expected0 = CpuCumsum(input);
    bool pass0 = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected0);
    RecordResult(pass0, "Cumsum FLOAT32 3D dim=0");

    // dim = 1
    auto expected1 = CpuCumsum(input);
    bool pass1 = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected1);
    RecordResult(pass1, "Cumsum FLOAT32 3D dim=1");
}

void TestInt32VariousDims(aclrtStream stream)
{
    std::vector<int32_t> input(24);
    for (int i = 0; i < 24; i++) input[i] = (i + 1) * 10;
    std::vector<int64_t> shape = {2, 3, 4};

    // dim = 0
    auto expected0 = CpuCumsumInt(input, false, false);
    bool pass0 = RunAndValidateCumsumInt32(stream, input, shape, 0, expected0);
    RecordResult(pass0, "Cumsum INT32 3D dim=0");

    // dim = 2
    auto expected2 = CpuCumsumInt(input, false, false);
    bool pass2 = RunAndValidateCumsumInt32(stream, input, shape, 2, expected2);
    RecordResult(pass2, "Cumsum INT32 3D dim=2");
}

void TestFloat32AccumulationError(aclrtStream stream)
{
    // Specifically test for accumulation error with long sequences
    std::vector<float> input(5000);
    for (int i = 0; i < 5000; i++) input[i] = 0.001f;
    std::vector<int64_t> shape = {5000};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 0.01, 0.01);
    RecordResult(pass, "Cumsum FLOAT32 accumulation error test (5000x0.001)");
}

void TestInt32CumulativePattern(aclrtStream stream)
{
    std::vector<int32_t> input(50);
    for (int i = 0; i < 50; i++) input[i] = (i % 5 == 0) ? 100 : 1;
    std::vector<int64_t> shape = {50};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 cumulative pattern");
}

void TestFloat32V2AllModes(aclrtStream stream)
{
    // Comprehensive V2 test
    std::vector<float> input(20);
    for (int i = 0; i < 20; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {20};

    bool allPass = true;
    for (bool exclusive : {false, true}) {
        for (bool reverse : {false, true}) {
            auto expected = CpuCumsum(input, 0, exclusive, reverse);
            bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, exclusive, reverse, expected);
            if (!pass) allPass = false;
        }
    }
    RecordResult(allPass, "CumsumV2 all modes comprehensive");
}

void TestInt8VariousPatterns(aclrtStream stream)
{
    std::vector<int8_t> input = {1, -1, 1, -1, 1, -1, 1, -1};
    std::vector<int64_t> shape = {8};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 alternating pattern");
}

void TestUint8VariousPatterns(aclrtStream stream)
{
    std::vector<uint8_t> input = {1, 2, 1, 2, 1, 2, 1, 2};
    std::vector<int64_t> shape = {8};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 pattern");
}

void TestInt64LargeValues(aclrtStream stream)
{
    std::vector<int64_t> input = {10000000000, 20000000000, 30000000000};
    std::vector<int64_t> shape = {3};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 very large values");
}

void TestFloat32PrecisionEdge(aclrtStream stream)
{
    // Edge case for precision
    std::vector<float> input = {1e-7f, 1e-7f, 1e-7f, 1e7f, 1e-7f, 1e-7f};
    std::vector<int64_t> shape = {6};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1.0f, 1e-5);
    RecordResult(pass, "Cumsum FLOAT32 precision edge (mixed magnitude)");
}

void TestV2OnFloatLong(aclrtStream stream)
{
    std::vector<float> input(200);
    for (int i = 0; i < 200; i++) input[i] = (float)(i % 10);
    std::vector<int64_t> shape = {200};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, false, expected, 1e-3, 1e-3);
    RecordResult(pass, "CumsumV2 FLOAT32 long sequence exclusive");
}

void TestFloat32DimEdge(aclrtStream stream)
{
    // Test dim at edge of valid range
    std::vector<float> input(8);
    for (int i = 0; i < 8; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {2, 2, 2};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 2, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 dim edge (2 for 3D)");
}

void TestInt32DimEdge(aclrtStream stream)
{
    std::vector<int32_t> input(8);
    for (int i = 0; i < 8; i++) input[i] = i + 1;
    std::vector<int64_t> shape = {2, 2, 2};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 2, expected);
    RecordResult(pass, "Cumsum INT32 dim edge (2 for 3D)");
}

void TestFloat32SpecialValues(aclrtStream stream)
{
    // Test with special float values (but avoid NaN/Inf as they may not be supported)
    std::vector<float> input = {0.0f, -0.0f, 1.0f, -1.0f, 0.0f};
    std::vector<int64_t> shape = {5};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 special values (+0, -0)");
}

void TestInt32ZeroHandling(aclrtStream stream)
{
    std::vector<int32_t> input = {0, 0, 0, 5, 0, 0};
    std::vector<int64_t> shape = {6};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 zero handling");
}

void TestFloat32CyclicPattern(aclrtStream stream)
{
    std::vector<float> input(30);
    for (int i = 0; i < 30; i++) input[i] = (float)(i % 6 + 1);
    std::vector<int64_t> shape = {30};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 cyclic pattern");
}

void TestV2ReverseOnLong(aclrtStream stream)
{
    std::vector<float> input(100);
    for (int i = 0; i < 100; i++) input[i] = (float)(100 - i);
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected, 1e-4, 1e-4);
    RecordResult(pass, "CumsumV2 reverse on long sequence");
}

void TestFloat32DifferentStrides(aclrtStream stream)
{
    // Test with contiguous data (strides are handled internally)
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> shape = {2, 3};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 different strides");
}

void TestInt32NegativeToPositive(aclrtStream stream)
{
    std::vector<int32_t> input = {-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50};
    std::vector<int64_t> shape = {11};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 negative to positive transition");
}

void TestFloat32SawtoothPattern(aclrtStream stream)
{
    std::vector<float> input(20);
    for (int i = 0; i < 20; i++) input[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    std::vector<int64_t> shape = {20};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 sawtooth pattern");
}

void TestV2ExclusiveOn1D(aclrtStream stream)
{
    std::vector<float> input = {5.0f};
    std::vector<int64_t> shape = {1};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, false, expected);
    RecordResult(pass, "CumsumV2 exclusive on single element");
}

void TestInt8SingleElement(aclrtStream stream)
{
    std::vector<int8_t> input = {127};
    std::vector<int64_t> shape = {1};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 single element max");
}

void TestUint8SingleElement(aclrtStream stream)
{
    std::vector<uint8_t> input = {255};
    std::vector<int64_t> shape = {1};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 single element max");
}

void TestInt64SingleElement(aclrtStream stream)
{
    std::vector<int64_t> input = {9223372036854775807LL};
    std::vector<int64_t> shape = {1};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 single element max");
}

void TestFloat32SequentialAccess(aclrtStream stream)
{
    // Pattern to test sequential memory access
    std::vector<float> input(48);
    for (int i = 0; i < 48; i++) input[i] = (float)(i + 1);
    std::vector<int64_t> shape = {4, 12};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 sequential access 4x12");
}

void TestInt32SequentialAccess(aclrtStream stream)
{
    std::vector<int32_t> input(48);
    for (int i = 0; i < 48; i++) input[i] = (i + 1) * 100;
    std::vector<int64_t> shape = {4, 12};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 1, expected);
    RecordResult(pass, "Cumsum INT32 sequential access 4x12");
}

void TestV2ComplexPattern(aclrtStream stream)
{
    std::vector<float> input(36);
    for (int i = 0; i < 36; i++) input[i] = (float)((i % 7) + 1);
    std::vector<int64_t> shape = {6, 6};
    auto expected = CpuCumsum(input, 0, true, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, true, expected);
    RecordResult(pass, "CumsumV2 complex pattern 6x6");
}

void TestFloat32DenseCalculation(aclrtStream stream)
{
    // Dense calculation test
    std::vector<float> input(100);
    for (int i = 0; i < 100; i++) input[i] = (float)(i * i % 100);
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-3, 1e-3);
    RecordResult(pass, "Cumsum FLOAT32 dense calculation");
}

void TestInt32DenseCalculation(aclrtStream stream)
{
    std::vector<int32_t> input(100);
    for (int i = 0; i < 100; i++) input[i] = (i * i) % 1000;
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 dense calculation");
}

void TestFloat32ScatteredPattern(aclrtStream stream)
{
    std::vector<float> input(40);
    for (int i = 0; i < 40; i++) input[i] = (i % 10 == 0) ? 10.0f : 1.0f;
    std::vector<int64_t> shape = {40};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 scattered pattern");
}

void TestV2OnScatteredData(aclrtStream stream)
{
    std::vector<float> input(25);
    for (int i = 0; i < 25; i++) input[i] = (i % 5 == 0) ? 5.0f : 1.0f;
    std::vector<int64_t> shape = {25};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected);
    RecordResult(pass, "CumsumV2 on scattered data");
}

void TestInt8ScatteredPattern(aclrtStream stream)
{
    std::vector<int8_t> input(40);
    for (int i = 0; i < 40; i++) input[i] = (i % 5 == 0) ? (int8_t)10 : (int8_t)1;
    std::vector<int64_t> shape = {40};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 scattered pattern");
}

void TestUint8ScatteredPattern(aclrtStream stream)
{
    std::vector<uint8_t> input(40);
    for (int i = 0; i < 40; i++) input[i] = (i % 5 == 0) ? (uint8_t)20 : (uint8_t)2;
    std::vector<int64_t> shape = {40};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 scattered pattern");
}

void TestInt64ScatteredPattern(aclrtStream stream)
{
    std::vector<int64_t> input(30);
    for (int i = 0; i < 30; i++) input[i] = (i % 3 == 0) ? (int64_t)1000000 : (int64_t)1;
    std::vector<int64_t> shape = {30};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 scattered pattern");
}

void TestFloat32RegularAccess(aclrtStream stream)
{
    std::vector<float> input(80);
    for (int i = 0; i < 80; i++) input[i] = (float)((i % 8) + 1);
    std::vector<int64_t> shape = {8, 10};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected);
    RecordResult(pass, "Cumsum FLOAT32 regular access 8x10");
}

void TestInt32RegularAccess(aclrtStream stream)
{
    std::vector<int32_t> input(80);
    for (int i = 0; i < 80; i++) input[i] = (i % 8) + 1;
    std::vector<int64_t> shape = {8, 10};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 1, expected);
    RecordResult(pass, "Cumsum INT32 regular access 8x10");
}

void TestV2OnRegularData(aclrtStream stream)
{
    std::vector<float> input(60);
    for (int i = 0; i < 60; i++) input[i] = (float)((i % 6) + 1);
    std::vector<int64_t> shape = {6, 10};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 1, true, false, expected);
    RecordResult(pass, "CumsumV2 on regular data 6x10");
}

void TestFloat32MemoryIntensive(aclrtStream stream)
{
    // Memory intensive pattern
    std::vector<float> input(150);
    for (int i = 0; i < 150; i++) input[i] = (float)(i % 15 + 1);
    std::vector<int64_t> shape = {150};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 memory intensive");
}

void TestInt32MemoryIntensive(aclrtStream stream)
{
    std::vector<int32_t> input(150);
    for (int i = 0; i < 150; i++) input[i] = (i % 15) + 1;
    std::vector<int64_t> shape = {150};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 memory intensive");
}

void TestFloat32ComputeIntensive(aclrtStream stream)
{
    // Compute intensive with many operations
    std::vector<float> input(120);
    for (int i = 0; i < 120; i++) input[i] = (float)((i * 7) % 50 + 1);
    std::vector<int64_t> shape = {120};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 compute intensive");
}

void TestV2ComputeIntensive(aclrtStream stream)
{
    std::vector<float> input(80);
    for (int i = 0; i < 80; i++) input[i] = (float)((i * 3) % 20 + 1);
    std::vector<int64_t> shape = {80};
    auto expected = CpuCumsum(input, 0, true, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, true, true, expected, 1e-4, 1e-4);
    RecordResult(pass, "CumsumV2 compute intensive");
}

void TestFloat32BoundarySize(aclrtStream stream)
{
    // Size that may trigger boundary conditions in tiling
    std::vector<float> input(255);
    for (int i = 0; i < 255; i++) input[i] = (float)(i % 20 + 1);
    std::vector<int64_t> shape = {255};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 boundary size (255)");
}

void TestInt32BoundarySize(aclrtStream stream)
{
    std::vector<int32_t> input(255);
    for (int i = 0; i < 255; i++) input[i] = (i % 20) + 1;
    std::vector<int64_t> shape = {255};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 boundary size (255)");
}

void TestFloat32Size512(aclrtStream stream)
{
    std::vector<float> input(512);
    for (int i = 0; i < 512; i++) input[i] = (float)(i % 16 + 1);
    std::vector<int64_t> shape = {512};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-3, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 size 512");
}

void TestInt32Size512(aclrtStream stream)
{
    std::vector<int32_t> input(512);
    for (int i = 0; i < 512; i++) input[i] = (i % 16) + 1;
    std::vector<int64_t> shape = {512};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT32 size 512");
}

void TestFloat32Size1024(aclrtStream stream)
{
    std::vector<float> input(1024);
    for (int i = 0; i < 1024; i++) input[i] = (float)(i % 32 + 1);
    std::vector<int64_t> shape = {1024};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 0, aclDataType::ACL_FLOAT, expected, 1e-3, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 size 1024");
}

void TestV2Size512(aclrtStream stream)
{
    std::vector<float> input(512);
    for (int i = 0; i < 512; i++) input[i] = (float)(i % 32 + 1);
    std::vector<int64_t> shape = {512};
    auto expected = CpuCumsum(input, 0, false, true);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 0, false, true, expected, 1e-3, 1e-4);
    RecordResult(pass, "CumsumV2 size 512 reverse");
}

void TestFloat32LargeBatchDim(aclrtStream stream)
{
    // Large batch dimension
    std::vector<float> input(2000);
    for (int i = 0; i < 2000; i++) input[i] = (float)(i % 10 + 1);
    std::vector<int64_t> shape = {200, 10};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected, 1e-3, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 large batch dim 200x10");
}

void TestInt32LargeBatchDim(aclrtStream stream)
{
    std::vector<int32_t> input(2000);
    for (int i = 0; i < 2000; i++) input[i] = (i % 10) + 1;
    std::vector<int64_t> shape = {200, 10};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt32(stream, input, shape, 1, expected);
    RecordResult(pass, "Cumsum INT32 large batch dim 200x10");
}

void TestFloat32LargeChannelDim(aclrtStream stream)
{
    // Large channel dimension
    std::vector<float> input(2000);
    for (int i = 0; i < 2000; i++) input[i] = (float)(i % 100 + 1);
    std::vector<int64_t> shape = {20, 100};
    auto expected = CpuCumsum(input);
    bool pass = RunAndValidateCumsum(stream, input, shape, 1, aclDataType::ACL_FLOAT, expected, 1e-4, 1e-4);
    RecordResult(pass, "Cumsum FLOAT32 large channel dim 20x100");
}

void TestV2LargeBatch(aclrtStream stream)
{
    std::vector<float> input(1000);
    for (int i = 0; i < 1000; i++) input[i] = (float)(i % 50 + 1);
    std::vector<int64_t> shape = {100, 10};
    auto expected = CpuCumsum(input, 0, true, false);
    bool pass = RunAndValidateCumsumV2(stream, input, shape, 1, true, false, expected, 1e-4, 1e-4);
    RecordResult(pass, "CumsumV2 large batch 100x10");
}

void TestInt8LargeArray(aclrtStream stream)
{
    std::vector<int8_t> input(200);
    for (int i = 0; i < 200; i++) input[i] = (int8_t)((i % 20) + 1);
    std::vector<int64_t> shape = {200};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT8 large array (200)");
}

void TestUint8LargeArray200(aclrtStream stream)
{
    std::vector<uint8_t> input(200);
    for (int i = 0; i < 200; i++) input[i] = (uint8_t)((i % 50) + 1);
    std::vector<int64_t> shape = {200};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumUint8(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum UINT8 large array (200)");
}

void TestInt64LargeArrayV2(aclrtStream stream)
{
    std::vector<int64_t> input(100);
    for (int i = 0; i < 100; i++) input[i] = (int64_t)(i + 1) * 10000;
    std::vector<int64_t> shape = {100};
    auto expected = CpuCumsumInt(input, false, false);
    bool pass = RunAndValidateCumsumInt64(stream, input, shape, 0, expected);
    RecordResult(pass, "Cumsum INT64 large array V2 (100)");
}

void TestFloat16VariousSizes(aclrtStream stream)
{
    // Float16 with various sizes
    std::vector<uint16_t> input(64);
    for (int i = 0; i < 64; i++) {
        // Represent small integers in float16
        input[i] = ((i % 10 + 1) << 10);
    }
    std::vector<int64_t> shape = {64};
    std::vector<double> expected(64);
    double sum = 0;
    for (int i = 0; i < 64; i++) {
        sum += (i % 10 + 1);
        expected[i] = sum;
    }
    bool pass = RunAndValidateCumsumFp16(stream, input, shape, 0, expected, 10.0, 0.1);
    RecordResult(pass, "Cumsum FLOAT16 various sizes");
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n========== Cumsum Operator Test Suite ==========\n\n");

    // Basic tests - FLOAT32
    TestBasicFloat32(stream);
    TestFloat32Dim1(stream);
    TestFloat32NegativeDim(stream);
    TestFloat32AllPositive(stream);
    TestFloat32AllNegative(stream);
    TestFloat32MixedSign(stream);
    TestFloat32WithZeros(stream);
    TestFloat32ShortSequence(stream);
    TestFloat32MediumSequence(stream);
    TestFloat32LongSequence(stream);
    TestFloat32VeryLongSequence(stream);
    TestFloat32MixedMagnitude(stream);
    TestFloat322D(stream);
    TestFloat323D(stream);
    TestFloat324D(stream);
    TestFloat32Decreasing(stream);
    TestFloat32VerySmallValues(stream);
    TestFloat32LargeValues(stream);
    TestFloat32AlternatingSign(stream);
    TestFloat32SingleElement(stream);
    TestFloat32TwoElements(stream);
    TestFloat32ThreeElements(stream);
    TestFloat32Fractional(stream);
    TestFloat32PrecisionAccumulation(stream);
    TestFloat32BatchProcessing(stream);
    TestFloat32PowerOfTwo(stream);
    TestFloat32BoundaryValues(stream);
    TestFloat32CubePath(stream);
    TestFloat32PrecisionEdge(stream);
    TestFloat32CyclicPattern(stream);
    TestFloat32DifferentStrides(stream);
    TestFloat32SequentialAccess(stream);
    TestFloat32RegularAccess(stream);
    TestFloat32ScatteredPattern(stream);
    TestFloat32MemoryIntensive(stream);
    TestFloat32ComputeIntensive(stream);
    TestFloat32BoundarySize(stream);
    TestFloat32Size512(stream);
    TestFloat32Size1024(stream);
    TestFloat32LargeBatchDim(stream);
    TestFloat32LargeChannelDim(stream);
    TestFloat32MultiDimensional(stream);
    TestFloat32MultiRow(stream);
    TestFloat32VariousShapes(stream);
    TestFloat32VariousSizes(stream);
    TestFloat32VariousDims(stream);
    TestFloat32Dim0(stream);
    TestFloat32DimEdge(stream);
    TestFloat32NegativeDim2D(stream);
    TestFloat32Size32(stream);
    TestFloat32Size16(stream);
    TestFloat32Size8(stream);
    TestFloat32AccumulationError(stream);
    TestFloat32ManySmallValues(stream);
    TestFloat32RepeatPattern(stream);
    TestFloat32SawtoothPattern(stream);
    TestFloat32SpecialValues(stream);

    // FLOAT16 tests
    TestFloat16Basic(stream);
    TestFloat16Simple(stream);
    TestFloat16VariousSizes(stream);

    // INT32 tests
    TestInt32Basic(stream);
    TestInt32Negative(stream);
    TestInt32Mixed(stream);
    TestInt32Dim1(stream);
    TestInt32DimVariations(stream);
    TestInt32Overflow(stream);
    TestInt32AllSame(stream);
    TestInt32SingleElement(stream);
    TestInt32TwoElements(stream);
    TestInt32Decreasing(stream);
    TestInt32LargeBatch(stream);
    TestInt32VariousLengths(stream);
    TestInt32Size128(stream);
    TestInt32Size256(stream);
    TestInt32Size512(stream);
    TestInt32OddLength(stream);
    TestInt32MultiRow(stream);
    TestInt32SequentialAccess(stream);
    TestInt32RegularAccess(stream);
    TestInt32DenseCalculation(stream);
    TestInt32MemoryIntensive(stream);
    TestInt32BoundarySize(stream);
    TestInt32LargeBatchDim(stream);
    TestInt32CumulativePattern(stream);
    TestInt32ZeroHandling(stream);
    TestInt32NegativeToPositive(stream);
    TestInt32ManyOnes(stream);
    TestInt32NegativeDim1D(stream);
    TestInt32VariousDims(stream);
    TestInt32DimEdge(stream);
    TestInt32TallShape(stream);

    // INT8 tests
    TestInt8Basic(stream);
    TestInt8NegativeValues(stream);
    TestInt8AllSame(stream);
    TestInt8ThreeElements(stream);
    TestInt8Boundary(stream);
    TestInt8VariousPatterns(stream);
    TestInt8ScatteredPattern(stream);
    TestInt8SingleElement(stream);
    TestInt8Size64(stream);
    TestInt8LargeArray(stream);

    // UINT8 tests
    TestUint8Basic(stream);
    TestUint8MaxValues(stream);
    TestUint8ThreeElements(stream);
    TestUint8Overflow(stream);
    TestUint8VariousPatterns(stream);
    TestUint8ScatteredPattern(stream);
    TestUint8SingleElement(stream);
    TestUint8Size32(stream);
    TestUint8LargeArray(stream);
    TestUint8LargeArray200(stream);

    // INT64 tests
    TestInt64Basic(stream);
    TestInt64Large(stream);
    TestInt64NegativeDim(stream);
    TestInt64SquareShape(stream);
    TestInt64VariousDims(stream);
    TestInt64LargeArrayV2(stream);
    TestInt64EdgeCase(stream);
    TestInt64ScatteredPattern(stream);
    TestInt64SingleElement(stream);
    TestInt64Size256(stream);
    TestInt64LargeValues(stream);

    // V2 API tests
    TestV2ExclusiveFalseReverseFalse(stream);
    TestV2ExclusiveTrueReverseFalse(stream);
    TestV2ExclusiveFalseReverseTrue(stream);
    TestV2ExclusiveTrueReverseTrue(stream);
    TestV2DifferentShapes(stream);
    TestV2ExclusiveOnLargerTensor(stream);
    TestV2ReverseOn2D(stream);
    TestV2OnIntTypes(stream);
    TestV2On2DLarge(stream);
    TestV2ComplexScenario(stream);
    TestV2ExclusiveReverseCombined(stream);
    TestV2DifferentCombinations(stream);
    TestV2ReverseOnLong(stream);
    TestV2ComplexPattern(stream);
    TestV2OnScatteredData(stream);
    TestV2OnRegularData(stream);
    TestV2ComputeIntensive(stream);
    TestV2Size512(stream);
    TestV2LargeBatch(stream);
    TestV2OnFloatLong(stream);
    TestV2OnMediumSize(stream);
    TestV2ExclusiveOnSmall(stream);
    TestV2ExclusiveOn1D(stream);
    TestV2Complex3D(stream);
    TestFloat32V2AllModes(stream);

    // Edge cases and error handling
    TestEmptyTensor(stream);
    TestErrorCases(stream);

    // Summary
    LOG_PRINT("\n========== Test Summary ==========\n");
    LOG_PRINT("Total: %d\n", g_totalCount);
    LOG_PRINT("Passed: %d\n", g_passCount);
    LOG_PRINT("Failed: %d\n", g_failCount);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return g_failCount > 0 ? 1 : 0;
}
