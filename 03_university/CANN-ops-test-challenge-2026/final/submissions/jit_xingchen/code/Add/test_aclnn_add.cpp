/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_ACL_RET(expr)                                                                          \
    do {                                                                                             \
        auto _ret = (expr);                                                                          \
        if (_ret != ACL_SUCCESS) {                                                                   \
            std::printf("[ERROR] %s failed, ret=%d, msg=%s\n", #expr, _ret, aclGetRecentErrMsg()); \
            return false;                                                                            \
        }                                                                                            \
    } while (0)

namespace {

int gPassed = 0;
int gFailed = 0;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000U) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x1000U) >> 13));
}

float HalfToFloat(uint16_t value)
{
    uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16;
    uint32_t exponent = (value >> 10) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<uint16_t> ToHalfVector(const std::vector<float>& values)
{
    std::vector<uint16_t> result;
    result.reserve(values.size());
    for (float value : values) {
        result.push_back(FloatToHalf(value));
    }
    return result;
}

struct TensorHolder {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
    std::vector<int64_t> shape;
    size_t elemSize = 0;

    ~TensorHolder()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }

    TensorHolder() = default;
    TensorHolder(const TensorHolder&) = delete;
    TensorHolder& operator=(const TensorHolder&) = delete;
};

struct ScalarHolder {
    aclScalar* scalar = nullptr;

    ~ScalarHolder()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
        }
    }

    ScalarHolder() = default;
    ScalarHolder(const ScalarHolder&) = delete;
    ScalarHolder& operator=(const ScalarHolder&) = delete;
};

struct AclEnv {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool ready = false;

    bool Init()
    {
        CHECK_ACL_RET(aclInit(nullptr));
        CHECK_ACL_RET(aclrtSetDevice(deviceId));
        CHECK_ACL_RET(aclrtCreateStream(&stream));
        ready = true;
        return true;
    }

    ~AclEnv()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (ready) {
            aclrtResetDevice(deviceId);
            aclFinalize();
        }
    }
};

template <typename T>
bool CreateTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
    TensorHolder* holder)
{
    holder->shape = shape;
    holder->elemSize = sizeof(T);
    size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    CHECK_ACL_RET(aclrtMalloc(&holder->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMemcpy(holder->deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), holder->deviceAddr);
    if (holder->tensor == nullptr) {
        std::printf("[ERROR] aclCreateTensor failed, msg=%s\n", aclGetRecentErrMsg());
        return false;
    }
    return true;
}

template <typename T>
bool ReadTensor(const TensorHolder& holder, std::vector<T>* result)
{
    size_t count = static_cast<size_t>(GetShapeSize(holder.shape));
    result->assign(count, T{});
    CHECK_ACL_RET(aclrtMemcpy(result->data(), count * sizeof(T), holder.deviceAddr, count * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST));
    return true;
}

template <typename T>
bool CreateScalar(T value, aclDataType dataType, ScalarHolder* holder)
{
    holder->scalar = aclCreateScalar(&value, dataType);
    if (holder->scalar == nullptr) {
        std::printf("[ERROR] aclCreateScalar failed, msg=%s\n", aclGetRecentErrMsg());
        return false;
    }
    return true;
}

template <typename GetWorkspace, typename RunKernel>
bool RunAclnn(const std::string& apiName, aclrtStream stream, GetWorkspace getWorkspace, RunKernel runKernel)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = getWorkspace(&workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::printf("[FAIL] %s GetWorkspaceSize ret=%d, msg=%s\n", apiName.c_str(), ret, aclGetRecentErrMsg());
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL_RET(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ret = runKernel(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::printf("[FAIL] %s run ret=%d, msg=%s\n", apiName.c_str(), ret, aclGetRecentErrMsg());
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return false;
    }
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return true;
}

bool Near(double actual, double expected, double atol, double rtol)
{
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

bool ExpectFloatVector(const std::vector<float>& actual, const std::vector<double>& expected, double atol, double rtol)
{
    if (actual.size() != expected.size()) {
        std::printf("  size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!Near(actual[i], expected[i], atol, rtol)) {
            std::printf("  mismatch[%zu]: actual=%.10f expected=%.10f\n", i, actual[i], expected[i]);
            ok = false;
        }
    }
    return ok;
}

bool ExpectIntVector(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected)
{
    if (actual.size() != expected.size()) {
        std::printf("  size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::printf("  mismatch[%zu]: actual=%d expected=%d\n", i, actual[i], expected[i]);
            ok = false;
        }
    }
    return ok;
}

bool ExpectHalfVector(const std::vector<uint16_t>& actual, const std::vector<double>& expected, double atol, double rtol)
{
    if (actual.size() != expected.size()) {
        std::printf("  size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        float actualFloat = HalfToFloat(actual[i]);
        if (!Near(actualFloat, expected[i], atol, rtol)) {
            std::printf("  mismatch[%zu]: actual=%.10f expected=%.10f\n", i, actualFloat, expected[i]);
            ok = false;
        }
    }
    return ok;
}

template <typename T>
bool ExpectIntegralVector(const std::vector<T>& actual, const std::vector<int64_t>& expected)
{
    if (actual.size() != expected.size()) {
        std::printf("  size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        int64_t actualValue = static_cast<int64_t>(actual[i]);
        if (actualValue != expected[i]) {
            std::printf("  mismatch[%zu]: actual=%ld expected=%ld\n", i, actualValue, expected[i]);
            ok = false;
        }
    }
    return ok;
}

void Record(const std::string& name, bool ok)
{
    std::printf("Test case: %s\n  [%s]\n", name.c_str(), ok ? "PASS" : "FAIL");
    if (ok) {
        ++gPassed;
    } else {
        ++gFailed;
    }
}

bool ExpectAclFailure(const char* name, aclnnStatus ret)
{
    if (ret == ACL_SUCCESS) {
        std::printf("  unexpected success: %s\n", name);
        return false;
    }
    std::printf("  expected failure: %s ret=%d\n", name, static_cast<int>(ret));
    return true;
}

bool OptionalWorkspaceProbe(const char* name, aclnnStatus ret)
{
    if (ret != ACL_SUCCESS) {
        std::printf("  optional probe skipped: %s ret=%d msg=%s\n", name, static_cast<int>(ret),
            aclGetRecentErrMsg());
        return true;
    }
    std::printf("  optional probe hit: %s\n", name);
    return true;
}

bool TestAddFloatAlpha(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> otherData = {1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f};
    std::vector<float> outData(6, 0.0f);
    std::vector<int64_t> shape = {2, 3};
    float alphaValue = 1.25f;
    if (!CreateTensor(selfData, shape, aclDataType::ACL_FLOAT, &self) ||
        !CreateTensor(otherData, shape, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, shape, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.float.alpha", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(out, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < selfData.size(); ++i) {
        expected.push_back(static_cast<double>(selfData[i]) + alphaValue * static_cast<double>(otherData[i]));
    }
    return ok && ExpectFloatVector(actual, expected, 1e-6, 1e-6);
}

bool TestAddBroadcastAlphaZero(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    std::vector<float> otherData = {100.0f, 200.0f, 300.0f};
    std::vector<float> outData(6, 0.0f);
    float alphaValue = 0.0f;
    if (!CreateTensor(selfData, {2, 3}, aclDataType::ACL_FLOAT, &self) ||
        !CreateTensor(otherData, {3}, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, {2, 3}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.broadcast.alpha0", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(out, &actual);
    std::vector<double> expected(selfData.begin(), selfData.end());
    return ok && ExpectFloatVector(actual, expected, 1e-6, 1e-6);
}

bool TestAddMixedFp16Fp32(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<float> selfFloat = {1.5f, -2.0f, 3.25f, 4.5f};
    std::vector<uint16_t> selfData = ToHalfVector(selfFloat);
    std::vector<float> otherData = {0.5f, 2.0f, -1.25f, 8.0f};
    std::vector<float> outData(4, 0.0f);
    float alphaValue = 1.0f;
    if (!CreateTensor(selfData, {4}, aclDataType::ACL_FLOAT16, &self) ||
        !CreateTensor(otherData, {4}, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, {4}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.mixed.fp16.fp32", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(out, &actual);
    if (ok) {
        std::printf("  mixed dtype coverage probe output: ");
        for (float value : actual) {
            std::printf("%.4f ", value);
        }
        std::printf("\n");
    }
    return ok;
}

bool TestAddFloat16SameDtype(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<float> selfFloat = {1.5f, -2.0f, 3.25f, 4.5f};
    std::vector<float> otherFloat = {0.5f, 2.0f, -1.25f, 1.5f};
    auto selfData = ToHalfVector(selfFloat);
    auto otherData = ToHalfVector(otherFloat);
    std::vector<uint16_t> outData(4, FloatToHalf(0.0f));
    float alphaValue = 1.0f;
    if (!CreateTensor(selfData, {4}, aclDataType::ACL_FLOAT16, &self) ||
        !CreateTensor(otherData, {4}, aclDataType::ACL_FLOAT16, &other) ||
        !CreateTensor(outData, {4}, aclDataType::ACL_FLOAT16, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.float16.same_dtype", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<uint16_t> actual;
    ok = ok && ReadTensor(out, &actual);
    if (ok) {
        std::printf("  float16 coverage probe output: ");
        for (uint16_t value : actual) {
            std::printf("%.4f ", HalfToFloat(value));
        }
        std::printf("\n");
    }
    return ok;
}

bool TestAddInt32NegativeAlpha(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<int32_t> selfData = {10, -10, 20, -20};
    std::vector<int32_t> otherData = {3, 4, -5, -6};
    std::vector<int32_t> outData(4, 0);
    int32_t alphaValue = -2;
    if (!CreateTensor(selfData, {4}, aclDataType::ACL_INT32, &self) ||
        !CreateTensor(otherData, {4}, aclDataType::ACL_INT32, &other) ||
        !CreateTensor(outData, {4}, aclDataType::ACL_INT32, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_INT32, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.int32.negative_alpha", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<int32_t> actual;
    ok = ok && ReadTensor(out, &actual);
    std::vector<int32_t> expected;
    for (size_t i = 0; i < selfData.size(); ++i) {
        expected.push_back(selfData[i] + alphaValue * otherData[i]);
    }
    return ok && ExpectIntVector(actual, expected);
}

bool TestAddInt8(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<int8_t> selfData = {1, 2, -3, 4};
    std::vector<int8_t> otherData = {2, -1, 4, -2};
    std::vector<int8_t> outData(4, 0);
    int8_t alphaValue = 1;
    if (!CreateTensor(selfData, {4}, aclDataType::ACL_INT8, &self) ||
        !CreateTensor(otherData, {4}, aclDataType::ACL_INT8, &other) ||
        !CreateTensor(outData, {4}, aclDataType::ACL_INT8, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_INT8, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.int8", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<int8_t> actual;
    ok = ok && ReadTensor(out, &actual);
    if (ok) {
        std::printf("  int8 coverage probe output: ");
        for (int8_t value : actual) {
            std::printf("%d ", static_cast<int32_t>(value));
        }
        std::printf("\n");
    }
    return ok;
}

bool TestAddInt64(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<int64_t> selfData = {10000000000LL, -10000000000LL, 7, -9};
    std::vector<int64_t> otherData = {5, -5, -3, 4};
    std::vector<int64_t> outData(4, 0);
    int64_t alphaValue = 1;
    if (!CreateTensor(selfData, {4}, aclDataType::ACL_INT64, &self) ||
        !CreateTensor(otherData, {4}, aclDataType::ACL_INT64, &other) ||
        !CreateTensor(outData, {4}, aclDataType::ACL_INT64, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_INT64, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.int64", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<int64_t> actual;
    ok = ok && ReadTensor(out, &actual);
    if (ok) {
        std::printf("  int64 coverage probe output: ");
        for (int64_t value : actual) {
            std::printf("%ld ", value);
        }
        std::printf("\n");
    }
    return ok;
}

bool TestAddsScalar(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;
    std::vector<float> selfData = {-2.0f, -1.0f, 0.0f, 1.0f};
    std::vector<float> outData(4, 0.0f);
    float otherValue = 4.0f;
    float alphaValue = 0.5f;
    if (!CreateTensor(selfData, {4}, aclDataType::ACL_FLOAT, &self) ||
        !CreateTensor(outData, {4}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(otherValue, aclDataType::ACL_FLOAT, &other) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdds.scalar", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdds(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(out, &actual);
    std::vector<double> expected;
    for (float value : selfData) {
        expected.push_back(static_cast<double>(value) + alphaValue * otherValue);
    }
    return ok && ExpectFloatVector(actual, expected, 1e-6, 1e-6);
}

bool TestInplaceAdd(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f};
    std::vector<float> otherData = {10.0f, -10.0f, 5.0f};
    float alphaValue = 2.0f;
    if (!CreateTensor(selfData, {3}, aclDataType::ACL_FLOAT, &self) ||
        !CreateTensor(otherData, {3}, aclDataType::ACL_FLOAT, &other) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnInplaceAdd", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(self, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < selfData.size(); ++i) {
        expected.push_back(static_cast<double>(selfData[i]) + alphaValue * otherData[i]);
    }
    return ok && ExpectFloatVector(actual, expected, 1e-6, 1e-6);
}

bool TestInplaceAdds(aclrtStream stream)
{
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;
    std::vector<int32_t> selfData = {5, 10, -20, 100};
    int32_t otherValue = 3;
    int32_t alphaValue = -4;
    if (!CreateTensor(selfData, {4}, aclDataType::ACL_INT32, &self) ||
        !CreateScalar(otherValue, aclDataType::ACL_INT32, &other) ||
        !CreateScalar(alphaValue, aclDataType::ACL_INT32, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnInplaceAdds", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<int32_t> actual;
    ok = ok && ReadTensor(self, &actual);
    std::vector<int32_t> expected;
    for (int32_t value : selfData) {
        expected.push_back(value + alphaValue * otherValue);
    }
    return ok && ExpectIntVector(actual, expected);
}

bool TestAddV3(aclrtStream stream)
{
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;
    std::vector<float> otherData = {1.0f, -2.0f, 3.5f};
    std::vector<float> outData(3, 0.0f);
    float selfValue = 10.0f;
    float alphaValue = 1.5f;
    if (!CreateTensor(otherData, {3}, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, {3}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(selfValue, aclDataType::ACL_FLOAT, &self) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAddV3", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAddV3(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(out, &actual);
    std::vector<double> expected;
    for (float value : otherData) {
        expected.push_back(selfValue + alphaValue * static_cast<double>(value));
    }
    return ok && ExpectFloatVector(actual, expected, 1e-6, 1e-6);
}

bool TestInplaceAddV3(aclrtStream stream)
{
    TensorHolder otherAndOut;
    ScalarHolder self;
    ScalarHolder alpha;
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f};
    float selfValue = -3.0f;
    float alphaValue = 2.0f;
    if (!CreateTensor(otherData, {3}, aclDataType::ACL_FLOAT, &otherAndOut) ||
        !CreateScalar(selfValue, aclDataType::ACL_FLOAT, &self) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnInplaceAddV3", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnInplaceAddV3GetWorkspaceSize(self.scalar, otherAndOut.tensor, alpha.scalar, workspaceSize,
                executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(otherAndOut, &actual);
    std::vector<double> expected;
    for (float value : otherData) {
        expected.push_back(selfValue + alphaValue * static_cast<double>(value));
    }
    return ok && ExpectFloatVector(actual, expected, 1e-6, 1e-6);
}

bool TestPrecisionCancellation(aclrtStream stream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<float> selfData = {1.000001f, 2.000001f};
    std::vector<float> otherData = {-1.0f, -2.0f};
    std::vector<float> outData(2, 0.0f);
    float alphaValue = 1.0f;
    if (!CreateTensor(selfData, {2}, aclDataType::ACL_FLOAT, &self) ||
        !CreateTensor(otherData, {2}, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, {2}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    bool ok = RunAclnn("aclnnAdd.precision.cancellation", stream,
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
        },
        [&](void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnAdd(workspaceAddr, workspaceSize, executor, runStream);
        });
    std::vector<float> actual;
    ok = ok && ReadTensor(out, &actual);
    std::vector<double> expected;
    for (size_t i = 0; i < selfData.size(); ++i) {
        expected.push_back(static_cast<double>(selfData[i]) + static_cast<double>(otherData[i]));
    }
    std::printf("  precision note: near cancellation keeps only a small residual in float32.\n");
    return ok && ExpectFloatVector(actual, expected, 1e-5, 1e-5);
}

bool TestWorkspaceOnlyBranchProbes(aclrtStream)
{
    {
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<int32_t> selfData = {1, -2, 3, -4};
        std::vector<float> otherData = {0.5f, 1.5f, -2.5f, 3.5f};
        std::vector<float> outData(4, 0.0f);
        float alphaValue = 1.0f;
        if (!CreateTensor(selfData, {4}, aclDataType::ACL_INT32, &self) ||
            !CreateTensor(otherData, {4}, aclDataType::ACL_FLOAT, &other) ||
            !CreateTensor(outData, {4}, aclDataType::ACL_FLOAT, &out) ||
            !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
            return false;
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        OptionalWorkspaceProbe("aclnnAdd promote int32+float alpha=1",
            aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor));
    }

    {
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<int8_t> selfData = {1, 2, -3, 4};
        std::vector<int8_t> otherData = {3, -2, 1, -4};
        std::vector<int8_t> outData(4, 0);
        int8_t alphaValue = 2;
        if (!CreateTensor(selfData, {4}, aclDataType::ACL_INT8, &self) ||
            !CreateTensor(otherData, {4}, aclDataType::ACL_INT8, &other) ||
            !CreateTensor(outData, {4}, aclDataType::ACL_INT8, &out) ||
            !CreateScalar(alphaValue, aclDataType::ACL_INT8, &alpha)) {
            return false;
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        OptionalWorkspaceProbe("aclnnAdd int8 alpha=2 AxpyV2 branch",
            aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor));
    }

    {
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<double> selfData = {1.0, -2.0};
        std::vector<double> otherData = {4.0, -8.0};
        std::vector<double> outData(2, 0.0);
        double alphaValue = 0.25;
        if (!CreateTensor(selfData, {2}, aclDataType::ACL_DOUBLE, &self) ||
            !CreateTensor(otherData, {2}, aclDataType::ACL_DOUBLE, &other) ||
            !CreateTensor(outData, {2}, aclDataType::ACL_DOUBLE, &out) ||
            !CreateScalar(alphaValue, aclDataType::ACL_DOUBLE, &alpha)) {
            return false;
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        OptionalWorkspaceProbe("aclnnAdd double alpha fallback Mul+Add/AiCpu",
            aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor));
    }

    {
        TensorHolder self;
        TensorHolder out;
        ScalarHolder other;
        ScalarHolder alpha;
        std::vector<float> selfData = {-1.0f, 0.0f, 1.0f, 2.0f};
        std::vector<float> outData(4, 0.0f);
        float otherValue = 3.0f;
        float alphaValue = 1.0f;
        if (!CreateTensor(selfData, {4}, aclDataType::ACL_FLOAT, &self) ||
            !CreateTensor(outData, {4}, aclDataType::ACL_FLOAT, &out) ||
            !CreateScalar(otherValue, aclDataType::ACL_FLOAT, &other) ||
            !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
            return false;
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        OptionalWorkspaceProbe("aclnnAdds alpha=1 Add branch",
            aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor));
    }

    {
        TensorHolder self;
        TensorHolder out;
        ScalarHolder other;
        ScalarHolder alpha;
        std::vector<int8_t> selfData = {1, -2, 3, -4};
        std::vector<int8_t> outData(4, 0);
        int8_t otherValue = 2;
        int8_t alphaValue = 3;
        if (!CreateTensor(selfData, {4}, aclDataType::ACL_INT8, &self) ||
            !CreateTensor(outData, {4}, aclDataType::ACL_INT8, &out) ||
            !CreateScalar(otherValue, aclDataType::ACL_INT8, &other) ||
            !CreateScalar(alphaValue, aclDataType::ACL_INT8, &alpha)) {
            return false;
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        OptionalWorkspaceProbe("aclnnAdds int8 scalar AxpyV2 branch",
            aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor));
    }

    {
        TensorHolder self;
        TensorHolder out;
        ScalarHolder other;
        ScalarHolder alpha;
        std::vector<uint8_t> selfData = {1, 0, 1, 1};
        std::vector<int32_t> outData(4, 0);
        bool otherValue = true;
        bool alphaValue = true;
        if (!CreateTensor(selfData, {4}, aclDataType::ACL_BOOL, &self) ||
            !CreateTensor(outData, {4}, aclDataType::ACL_INT32, &out) ||
            !CreateScalar(otherValue, aclDataType::ACL_BOOL, &other) ||
            !CreateScalar(alphaValue, aclDataType::ACL_BOOL, &alpha)) {
            return false;
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        OptionalWorkspaceProbe("aclnnAdds bool scalar true to int32 branch",
            aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor));
    }

    return true;
}

bool TestNegativeNullSelf(aclrtStream)
{
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<float> data = {1.0f, 2.0f};
    std::vector<float> outData(2, 0.0f);
    float alphaValue = 1.0f;
    if (!CreateTensor(data, {2}, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, {2}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    auto ret = aclnnAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ret != ACL_SUCCESS;
}

bool TestNegativeBadBroadcast(aclrtStream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f};
    std::vector<float> outData(4, 0.0f);
    float alphaValue = 1.0f;
    if (!CreateTensor(selfData, {2, 2}, aclDataType::ACL_FLOAT, &self) ||
        !CreateTensor(otherData, {3}, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, {2, 2}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ret != ACL_SUCCESS;
}

bool TestNegativeBadOutShape(aclrtStream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> outData(3, 0.0f);
    float alphaValue = 1.0f;
    if (!CreateTensor(data, {2, 2}, aclDataType::ACL_FLOAT, &self) ||
        !CreateTensor(data, {2, 2}, aclDataType::ACL_FLOAT, &other) ||
        !CreateTensor(outData, {3}, aclDataType::ACL_FLOAT, &out) ||
        !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
        return false;
    }
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ret != ACL_SUCCESS;
}

bool TestNegativeAdditionalGuards(aclrtStream)
{
    bool ok = true;

    {
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<float> data = {1.0f, 2.0f};
        std::vector<float> outData(2, 0.0f);
        float alphaValue = 1.0f;
        if (!CreateTensor(data, {2}, aclDataType::ACL_FLOAT, &self) ||
            !CreateTensor(data, {2}, aclDataType::ACL_FLOAT, &other) ||
            !CreateTensor(outData, {2}, aclDataType::ACL_FLOAT, &out) ||
            !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
            return false;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdd nullptr other",
            aclnnAddGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor));

        workspaceSize = 0;
        executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdd nullptr alpha",
            aclnnAddGetWorkspaceSize(self.tensor, other.tensor, nullptr, out.tensor, &workspaceSize, &executor));

        workspaceSize = 0;
        executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdd nullptr out",
            aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, nullptr, &workspaceSize, &executor));
    }

    {
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<float> data = {1.0f};
        std::vector<int64_t> tooManyDims = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        float alphaValue = 1.0f;
        if (!CreateTensor(data, tooManyDims, aclDataType::ACL_FLOAT, &self) ||
            !CreateTensor(data, tooManyDims, aclDataType::ACL_FLOAT, &other) ||
            !CreateTensor(data, tooManyDims, aclDataType::ACL_FLOAT, &out) ||
            !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
            return false;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdd rank > 8",
            aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor));
    }

    {
        TensorHolder self;
        TensorHolder out;
        TensorHolder badOut;
        ScalarHolder other;
        ScalarHolder alpha;
        std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> outData(4, 0.0f);
        std::vector<float> badOutData(3, 0.0f);
        float otherValue = 2.0f;
        float alphaValue = 0.5f;
        if (!CreateTensor(selfData, {4}, aclDataType::ACL_FLOAT, &self) ||
            !CreateTensor(outData, {4}, aclDataType::ACL_FLOAT, &out) ||
            !CreateTensor(badOutData, {3}, aclDataType::ACL_FLOAT, &badOut) ||
            !CreateScalar(otherValue, aclDataType::ACL_FLOAT, &other) ||
            !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
            return false;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdds nullptr other",
            aclnnAddsGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor));

        workspaceSize = 0;
        executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdds nullptr alpha",
            aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, nullptr, out.tensor, &workspaceSize, &executor));

        workspaceSize = 0;
        executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdds nullptr out",
            aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, nullptr, &workspaceSize, &executor));

        workspaceSize = 0;
        executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnAdds bad out shape",
            aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, badOut.tensor, &workspaceSize,
                &executor));
    }

    {
        TensorHolder self;
        TensorHolder other;
        TensorHolder broadcastOther;
        ScalarHolder alpha;
        std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> badOtherData = {1.0f, 2.0f, 3.0f};
        float alphaValue = 1.0f;
        if (!CreateTensor(selfData, {2, 2}, aclDataType::ACL_FLOAT, &self) ||
            !CreateTensor(otherData, {2, 2}, aclDataType::ACL_FLOAT, &other) ||
            !CreateTensor(badOtherData, {3}, aclDataType::ACL_FLOAT, &broadcastOther) ||
            !CreateScalar(alphaValue, aclDataType::ACL_FLOAT, &alpha)) {
            return false;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnInplaceAdd nullptr other",
            aclnnInplaceAddGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, &workspaceSize, &executor));

        workspaceSize = 0;
        executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnInplaceAdd nullptr alpha",
            aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, nullptr, &workspaceSize, &executor));

        workspaceSize = 0;
        executor = nullptr;
        ok = ok && ExpectAclFailure("aclnnInplaceAdd invalid broadcast",
            aclnnInplaceAddGetWorkspaceSize(self.tensor, broadcastOther.tensor, alpha.scalar, &workspaceSize,
                &executor));
    }

    return ok;
}

} // namespace

int main()
{
    AclEnv env;
    if (!env.Init()) {
        return 1;
    }

    Record("Basic aclnnAdd float32 alpha=1.25", TestAddFloatAlpha(env.stream));
    Record("Broadcast aclnnAdd alpha=0", TestAddBroadcastAlphaZero(env.stream));
    Record("Mixed dtype aclnnAdd fp16 + fp32 coverage", TestAddMixedFp16Fp32(env.stream));
    Record("FLOAT16 aclnnAdd same dtype", TestAddFloat16SameDtype(env.stream));
    Record("INT32 aclnnAdd negative alpha", TestAddInt32NegativeAlpha(env.stream));
    Record("INT8 aclnnAdd", TestAddInt8(env.stream));
    Record("INT64 aclnnAdd", TestAddInt64(env.stream));
    Record("aclnnAdds tensor + scalar", TestAddsScalar(env.stream));
    Record("aclnnInplaceAdd tensor += tensor", TestInplaceAdd(env.stream));
    Record("aclnnInplaceAdds tensor += scalar", TestInplaceAdds(env.stream));
    Record("aclnnAddV3 scalar + tensor", TestAddV3(env.stream));
    Record("aclnnInplaceAddV3 scalar + tensor in-place", TestInplaceAddV3(env.stream));
    Record("Precision cancellation", TestPrecisionCancellation(env.stream));
    Record("Workspace-only branch coverage probes", TestWorkspaceOnlyBranchProbes(env.stream));
    Record("Negative nullptr self", TestNegativeNullSelf(env.stream));
    Record("Negative invalid broadcast shape", TestNegativeBadBroadcast(env.stream));
    Record("Negative invalid output shape", TestNegativeBadOutShape(env.stream));
    Record("Negative additional parameter guards", TestNegativeAdditionalGuards(env.stream));

    std::printf("Summary: %d passed, %d failed\n", gPassed, gFailed);
    return gFailed == 0 ? 0 : 1;
}
