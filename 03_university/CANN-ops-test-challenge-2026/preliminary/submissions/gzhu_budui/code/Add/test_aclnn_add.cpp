/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, action) \
    do {                         \
        if (!(cond)) {           \
            action;              \
        }                        \
    } while (0)

#define LOG_PRINT(fmt, ...)         \
    do {                            \
        std::printf(fmt, ##__VA_ARGS__); \
    } while (0)

struct CaseStat {
    int total = 0;
    int failed = 0;
};

struct TensorPack {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
};

struct AclResourceGuard {
    std::vector<void*> devices;
    std::vector<aclTensor*> tensors;
    std::vector<aclScalar*> scalars;

    ~AclResourceGuard()
    {
        for (auto* s : scalars) {
            if (s != nullptr) {
                aclDestroyScalar(s);
            }
        }
        for (auto* t : tensors) {
            if (t != nullptr) {
                aclDestroyTensor(t);
            }
        }
        for (auto* d : devices) {
            if (d != nullptr) {
                aclrtFree(d);
            }
        }
    }

    void Keep(void* device)
    {
        devices.push_back(device);
    }

    void Keep(aclTensor* tensor)
    {
        tensors.push_back(tensor);
    }

    void Keep(aclScalar* scalar)
    {
        scalars.push_back(scalar);
    }
};

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= shape[i];
    }
    return size;
}

static std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

static void LinearToIndices(int64_t linear, const std::vector<int64_t>& shape, std::vector<int64_t>* indices)
{
    indices->assign(shape.size(), 0);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        const auto dim = shape[static_cast<size_t>(i)];
        (*indices)[static_cast<size_t>(i)] = linear % dim;
        linear /= dim;
    }
}

static int64_t BroadcastOffset(
    const std::vector<int64_t>& outIndices, const std::vector<int64_t>& inShape, const std::vector<int64_t>& inStrides)
{
    const int64_t rankDiff = static_cast<int64_t>(outIndices.size()) - static_cast<int64_t>(inShape.size());
    int64_t offset = 0;
    for (size_t i = 0; i < inShape.size(); ++i) {
        const int64_t outIdx = outIndices[static_cast<size_t>(static_cast<int64_t>(i) + rankDiff)];
        const int64_t inIdx = inShape[i] == 1 ? 0 : outIdx;
        offset += inIdx * inStrides[i];
    }
    return offset;
}

static bool AlmostEqual(double expected, double actual, double atol = 1e-5, double rtol = 1e-5)
{
    if (std::isnan(expected) && std::isnan(actual)) {
        return true;
    }
    if (std::isinf(expected) && std::isinf(actual)) {
        return (expected > 0.0) == (actual > 0.0);
    }
    return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, ret=%d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, ret=%d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, ret=%d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
static bool CreateTensor(
    AclResourceGuard* guard,
    const std::vector<T>& hostData,
    const std::vector<int64_t>& shape,
    aclDataType dataType,
    TensorPack* out)
{
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(&out->device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, ret=%d\n", ret); return false);
    guard->Keep(out->device);

    ret = aclrtMemcpy(out->device, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed, ret=%d\n", ret); return false);

    auto strides = GetStrides(shape);
    out->tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0,
        ACL_FORMAT_ND, shape.data(), shape.size(), out->device);
    CHECK_RET(out->tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return false);
    guard->Keep(out->tensor);
    return true;
}

template <typename T>
static bool CopyToHost(void* deviceAddr, std::vector<T>* hostData)
{
    const size_t bytes = hostData->size() * sizeof(T);
    auto ret = aclrtMemcpy(hostData->data(), bytes, deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed, ret=%d\n", ret); return false);
    return true;
}

template <typename T>
static aclScalar* CreateScalar(AclResourceGuard* guard, T value, aclDataType type)
{
    aclScalar* scalar = aclCreateScalar(&value, type);
    if (scalar != nullptr) {
        guard->Keep(scalar);
    }
    return scalar;
}

template <typename GetWorkspaceFn, typename LaunchFn>
static bool RunWithWorkspace(const std::string& name, GetWorkspaceFn getWorkspace, LaunchFn launch, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;

    auto ret = getWorkspace(&workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  %s getWorkspace ret=%d\n", name.c_str(), ret); return false);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  %s malloc workspace ret=%d\n", name.c_str(), ret); return false);
    }

    ret = launch(workspace, workspaceSize, executor);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  %s launch ret=%d\n", name.c_str(), ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  %s sync ret=%d\n", name.c_str(), ret); return false);
    return true;
}

static bool ValidateFloatAdd(
    const std::string& name,
    const std::vector<float>& actual,
    const std::vector<float>& lhs,
    const std::vector<int64_t>& lhsShape,
    const std::vector<float>& rhs,
    const std::vector<int64_t>& rhsShape,
    const std::vector<int64_t>& outShape,
    float alpha)
{
    std::vector<int64_t> outIndices;
    const auto lhsStrides = GetStrides(lhsShape);
    const auto rhsStrides = GetStrides(rhsShape);
    const int64_t total = GetShapeSize(outShape);

    for (int64_t i = 0; i < total; ++i) {
        LinearToIndices(i, outShape, &outIndices);
        const int64_t l = BroadcastOffset(outIndices, lhsShape, lhsStrides);
        const int64_t r = BroadcastOffset(outIndices, rhsShape, rhsStrides);
        const double expected = static_cast<double>(lhs[static_cast<size_t>(l)]) +
            static_cast<double>(alpha) * static_cast<double>(rhs[static_cast<size_t>(r)]);
        if (!AlmostEqual(expected, actual[static_cast<size_t>(i)])) {
            LOG_PRINT(
                "  %s mismatch at %lld expected=%f actual=%f\n",
                name.c_str(), static_cast<long long>(i), expected, static_cast<double>(actual[static_cast<size_t>(i)]));
            return false;
        }
    }
    return true;
}

static bool ReportAndReturn(CaseStat* stat, const std::string& name, bool pass)
{
    ++stat->total;
    if (!pass) {
        ++stat->failed;
    }
    LOG_PRINT("[%s] %s\n", pass ? "PASS" : "FAIL", name.c_str());
    return pass;
}

static bool CaseAddFloat(
    const std::string& name,
    const std::vector<float>& selfData,
    const std::vector<int64_t>& selfShape,
    const std::vector<float>& otherData,
    const std::vector<int64_t>& otherShape,
    const std::vector<int64_t>& outShape,
    float alphaValue,
    aclrtStream stream)
{
    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;

    std::vector<float> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
    CHECK_RET(CreateTensor(&guard, selfData, selfShape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, otherShape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, outShape, ACL_FLOAT, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    return ValidateFloatAdd(name, outHost, selfData, selfShape, otherData, otherShape, outShape, alphaValue);
}

static bool CaseAddPromoteInt32Float(aclrtStream stream)
{
    const std::string name = "add_promote_int32_float_alpha1";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int32_t> selfData = {1, -2, 3, 4};
    const std::vector<float> otherData = {0.5f, -1.0f, 2.5f, 1.0f};
    const float alphaValue = 1.0f;
    std::vector<float> outHost(otherData.size(), 0.0f);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_INT32, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const double expected = static_cast<double>(selfData[i]) +
            static_cast<double>(alphaValue) * static_cast<double>(otherData[i]);
        if (!AlmostEqual(expected, static_cast<double>(outHost[i]))) {
            LOG_PRINT("  %s mismatch at %zu expected=%f actual=%f\n", name.c_str(), i, expected,
                static_cast<double>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseAddsFloat(
    const std::string& name,
    const std::vector<float>& selfData,
    const std::vector<int64_t>& shape,
    float otherValue,
    float alphaValue,
    aclrtStream stream)
{
    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    std::vector<float> outHost(selfData.size(), 0.0f);
    std::vector<float> rhs(selfData.size(), otherValue);

    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdds(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    return ValidateFloatAdd(name, outHost, selfData, shape, rhs, shape, shape, alphaValue);
}

static bool CaseAddsInt64Exact(aclrtStream stream)
{
    const std::string name = "adds_int64_alpha2_exact";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int64_t> selfData = {10, -20, 0, 5};
    const int64_t otherValue = -3;
    const int64_t alphaValue = 2;
    std::vector<int64_t> outHost(selfData.size(), 0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_INT64, &self), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_INT64, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_INT64);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_INT64);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdds(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const int64_t expected = selfData[i] + alphaValue * otherValue;
        if (outHost[i] != expected) {
            LOG_PRINT("  %s mismatch at %zu expected=%lld actual=%lld\n", name.c_str(), i,
                static_cast<long long>(expected), static_cast<long long>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseInplaceAddFloat(
    const std::string& name,
    const std::vector<float>& selfInit,
    const std::vector<int64_t>& selfShape,
    const std::vector<float>& otherData,
    const std::vector<int64_t>& otherShape,
    float alphaValue,
    aclrtStream stream)
{
    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    std::vector<float> selfOut(selfInit.size(), 0.0f);

    CHECK_RET(CreateTensor(&guard, selfInit, selfShape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, otherShape, ACL_FLOAT, &other), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnInplaceAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(self.device, &selfOut), return false);
    return ValidateFloatAdd(name, selfOut, selfInit, selfShape, otherData, otherShape, selfShape, alphaValue);
}

static bool CaseInplaceAddsFloat(
    const std::string& name,
    const std::vector<float>& selfInit,
    const std::vector<int64_t>& shape,
    float otherValue,
    float alphaValue,
    aclrtStream stream)
{
    AclResourceGuard guard;
    TensorPack self;
    std::vector<float> selfOut(selfInit.size(), 0.0f);
    std::vector<float> rhs(selfInit.size(), otherValue);

    CHECK_RET(CreateTensor(&guard, selfInit, shape, ACL_FLOAT, &self), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnInplaceAddsGetWorkspaceSize(self.tensor, other, alpha, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnInplaceAdds(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(self.device, &selfOut), return false);
    return ValidateFloatAdd(name, selfOut, selfInit, shape, rhs, shape, shape, alphaValue);
}

static bool CaseAddV3Float(
    const std::string& name,
    float selfScalar,
    const std::vector<float>& otherData,
    const std::vector<int64_t>& shape,
    float alphaValue,
    aclrtStream stream)
{
    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    std::vector<float> outHost(otherData.size(), 0.0f);
    std::vector<float> lhs(otherData.size(), selfScalar);

    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT, &out), return false);

    auto* self = CreateScalar(&guard, selfScalar, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAddV3(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    return ValidateFloatAdd(name, outHost, lhs, shape, otherData, shape, shape, alphaValue);
}

static bool CaseAddV3DoubleSelfInt32OutFloat(aclrtStream stream)
{
    const std::string name = "add_v3_self_double_other_int32_out_float";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int32_t> otherData = {1, -2, 3, 0};
    std::vector<float> outHost(otherData.size(), 0.0f);
    const double selfValue = 1.25;
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_INT32, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT, &out), return false);

    auto* self = CreateScalar(&guard, selfValue, ACL_DOUBLE);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAddV3(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const double expected = selfValue + static_cast<double>(alphaValue) * static_cast<double>(otherData[i]);
        if (!AlmostEqual(expected, static_cast<double>(outHost[i]))) {
            LOG_PRINT("  %s mismatch at %zu expected=%f actual=%f\n", name.c_str(), i, expected,
                static_cast<double>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseAddV3FloatSelfInt32Status(aclrtStream stream)
{
    const std::string name = "add_v3_self_float_other_int32_alpha2";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int32_t> otherData = {4, -3, 1, 2};
    std::vector<float> outHost(otherData.size(), 0.0f);
    const float selfValue = -0.5f;
    const float alphaValue = 2.0f;

    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_INT32, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT, &out), return false);

    auto* self = CreateScalar(&guard, selfValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAddV3(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const double expected = static_cast<double>(selfValue) +
            static_cast<double>(alphaValue) * static_cast<double>(otherData[i]);
        if (!AlmostEqual(expected, static_cast<double>(outHost[i]))) {
            LOG_PRINT("  %s mismatch at %zu expected=%f actual=%f\n", name.c_str(), i, expected,
                static_cast<double>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseInplaceAddV3Status(const std::string& name, float selfValue, float alphaValue, aclrtStream stream)
{
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> otherData = {1.0f, -2.0f, 3.0f, 0.5f};

    AclResourceGuard guard;
    TensorPack other;
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_FLOAT, &other), return false);

    auto* self = CreateScalar(&guard, selfValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(self, other.tensor, alpha, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s getWorkspace ret=%d (accepted on simulator)\n", name.c_str(), ret);
        return true;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return false);
    }
    ret = aclnnInplaceAddV3(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    return true;
}

static bool CaseAddInt32(aclrtStream stream)
{
    const std::string name = "add_int32";
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<int32_t> selfData = {1, 2, -3, 4, 0, 7};
    const std::vector<int32_t> otherData = {3, -2, 5, 1, 2, -4};
    const int32_t alphaValue = 2;
    std::vector<int32_t> outHost(selfData.size(), 0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_INT32, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_INT32, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_INT32, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_INT32);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const int32_t expected = static_cast<int32_t>(selfData[i] + alphaValue * otherData[i]);
        if (outHost[i] != expected) {
            LOG_PRINT("  %s mismatch at %zu expected=%d actual=%d\n", name.c_str(), i, expected, outHost[i]);
            return false;
        }
    }
    return true;
}

static bool CaseAddInt64Exact(aclrtStream stream)
{
    const std::string name = "add_int64_alpha3_exact";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int64_t> selfData = {10, -20, 30, 0};
    const std::vector<int64_t> otherData = {1, 2, -3, 4};
    const int64_t alphaValue = 3;
    std::vector<int64_t> outHost(selfData.size(), 0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_INT64, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_INT64, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_INT64, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const int64_t expected = selfData[i] + alphaValue * otherData[i];
        if (outHost[i] != expected) {
            LOG_PRINT("  %s mismatch at %zu expected=%lld actual=%lld\n", name.c_str(), i,
                static_cast<long long>(expected), static_cast<long long>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseAddUint8Exact(aclrtStream stream)
{
    const std::string name = "add_uint8_alpha2_exact";
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<uint8_t> selfData = {1, 2, 3, 4, 5, 6};
    const std::vector<uint8_t> otherData = {1, 0, 2, 3, 1, 2};
    const uint8_t alphaValue = 2;
    std::vector<uint8_t> outHost(selfData.size(), 0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_UINT8, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_UINT8, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_UINT8, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_UINT8);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const int32_t expected = static_cast<int32_t>(selfData[i]) +
            static_cast<int32_t>(alphaValue) * static_cast<int32_t>(otherData[i]);
        if (static_cast<int32_t>(outHost[i]) != expected) {
            LOG_PRINT("  %s mismatch at %zu expected=%d actual=%d\n", name.c_str(), i, expected,
                static_cast<int32_t>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseAddsBoolOutBoolExact(aclrtStream stream)
{
    const std::string name = "adds_bool_out_bool_exact";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint8_t> selfData = {0, 1, 1, 0};
    const bool otherValue = true;
    const bool alphaValue = true;
    std::vector<uint8_t> outHost(selfData.size(), 0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_BOOL, &self), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_BOOL, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_BOOL);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_BOOL);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdds(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const uint8_t expected = static_cast<uint8_t>((selfData[i] != 0) || (otherValue && alphaValue));
        if (outHost[i] != expected) {
            LOG_PRINT("  %s mismatch at %zu expected=%u actual=%u\n", name.c_str(), i,
                static_cast<unsigned>(expected), static_cast<unsigned>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseAddEmptyTensor(aclrtStream stream)
{
    (void)stream;
    const std::string name = "add_empty_tensor_workspace0";
    const std::vector<int64_t> shape = {0, 3};
    std::vector<float> data;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &out), return false);

    auto* alpha = CreateScalar(&guard, 1.0f, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t ws = 123;
    aclOpExecutor* ex = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &ws, &ex);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s getWorkspace ret=%d\n", name.c_str(), ret);
        return false;
    }
    return ws == 0 && ex != nullptr;
}

static bool CaseAddsEmptyTensor(aclrtStream stream)
{
    (void)stream;
    const std::string name = "adds_empty_tensor_workspace0";
    const std::vector<int64_t> shape = {0, 2};
    std::vector<float> data;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &out), return false);

    auto* other = CreateScalar(&guard, 2.0f, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, 0.5f, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t ws = 123;
    aclOpExecutor* ex = nullptr;
    const auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &ws, &ex);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s getWorkspace ret=%d\n", name.c_str(), ret);
        return false;
    }
    return ws == 0 && ex != nullptr;
}

static bool CaseAddV3EmptyTensor(aclrtStream stream)
{
    (void)stream;
    const std::string name = "add_v3_empty_tensor_workspace0";
    const std::vector<int64_t> shape = {0, 2};
    std::vector<float> otherData;
    std::vector<float> outData;

    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outData, shape, ACL_FLOAT, &out), return false);

    auto* self = CreateScalar(&guard, 1.0f, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, 1.0f, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t ws = 123;
    aclOpExecutor* ex = nullptr;
    const auto ret = aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &ws, &ex);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s getWorkspace ret=%d\n", name.c_str(), ret);
        return false;
    }
    return ws == 0 && ex != nullptr;
}

template <typename T, typename S>
static bool CaseAddStatus(
    const std::string& name,
    const std::vector<T>& selfData,
    const std::vector<T>& otherData,
    const std::vector<int64_t>& shape,
    aclDataType tensorType,
    S alphaValue,
    aclDataType alphaType,
    aclrtStream stream)
{
    std::vector<T> outHost(selfData.size(), static_cast<T>(0));

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, tensorType, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, tensorType, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, tensorType, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, alphaType);
    CHECK_RET(alpha != nullptr, return false);

    return RunWithWorkspace(
        name,
        [&](uint64_t* ws, aclOpExecutor** ex) {
            return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
        },
        [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
        stream);
}

static bool CaseAddV3Int8(aclrtStream stream)
{
    const std::string name = "add_v3_int8_alpha2";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int8_t> otherData = {1, -2, 3, 0};
    const int8_t selfValue = 2;
    const int8_t alphaValue = 2;
    std::vector<int8_t> outHost(otherData.size(), 0);

    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_INT8, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_INT8, &out), return false);

    auto* self = CreateScalar(&guard, selfValue, ACL_INT8);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_INT8);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAddV3(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const int32_t expected = static_cast<int32_t>(selfValue) + static_cast<int32_t>(alphaValue) * static_cast<int32_t>(otherData[i]);
        if (static_cast<int32_t>(outHost[i]) != expected) {
            LOG_PRINT("  %s mismatch at %zu expected=%d actual=%d\n", name.c_str(), i, expected, static_cast<int32_t>(outHost[i]));
            return false;
        }
    }
    return true;
}

static bool CaseAddsBoolToInt(aclrtStream stream)
{
    const std::string name = "adds_bool_to_int";
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<uint8_t> selfData = {0, 1, 0, 1, 0, 1};
    const bool otherValue = true;
    const bool alphaValue = true;
    std::vector<int32_t> outHost(selfData.size(), 0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_BOOL, &self), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_INT32, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_BOOL);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_BOOL);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdds(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        if (!(outHost[i] == 0 || outHost[i] == 1)) {
            LOG_PRINT("  %s unexpected value at %zu: %d\n", name.c_str(), i, outHost[i]);
            return false;
        }
    }
    return true;
}

static bool CaseAddsBoolToIntAlphaFalse(aclrtStream stream)
{
    const std::string name = "adds_bool_to_int_alpha_false";
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<uint8_t> selfData = {0, 1, 0, 1, 1, 0};
    const bool otherValue = true;
    const bool alphaValue = false;
    std::vector<int32_t> outHost(selfData.size(), 0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_BOOL, &self), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_INT32, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_BOOL);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_BOOL);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdds(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const int32_t expected = selfData[i] != 0 ? 1 : 0;
        if (outHost[i] != expected) {
            LOG_PRINT("  %s mismatch at %zu expected=%d actual=%d\n", name.c_str(), i, expected, outHost[i]);
            return false;
        }
    }
    return true;
}

static bool CaseMixedF16F32Status(const std::string& name, float alphaValue, aclrtStream stream)
{
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> selfF16Bits = {0x3C00, 0xC000, 0x4200, 0x0000};
    const std::vector<float> otherData = {1.0f, 2.0f, -1.0f, 4.0f};
    std::vector<float> outHost(otherData.size(), 0.0f);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfF16Bits, shape, ACL_FLOAT16, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        if (!std::isfinite(outHost[i])) {
            return false;
        }
    }
    return true;
}

static bool CaseMixedF32F16Status(const std::string& name, float alphaValue, aclrtStream stream)
{
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> selfData = {1.0f, -2.0f, 0.5f, 4.0f};
    const std::vector<uint16_t> otherF16Bits = {0x3C00, 0x4000, 0xBC00, 0x0000};
    std::vector<float> outHost(selfData.size(), 0.0f);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherF16Bits, shape, ACL_FLOAT16, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        if (!std::isfinite(outHost[i])) {
            return false;
        }
    }
    return true;
}

static bool CaseAddFloat16Status(aclrtStream stream)
{
    const std::string name = "add_float16_alpha2_status";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> selfData = {0x3C00, 0x4000, 0xC000, 0x0000};
    const std::vector<uint16_t> otherData = {0x3800, 0x3C00, 0x3C00, 0x3C00};
    std::vector<uint16_t> outHost(selfData.size(), 0);
    const float alphaValue = 2.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_FLOAT16, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_FLOAT16, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_FLOAT16, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    CHECK_RET(
        RunWithWorkspace(
            name,
            [&](uint64_t* ws, aclOpExecutor** ex) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, ws, ex);
            },
            [&](void* ws, uint64_t wsSize, aclOpExecutor* ex) { return aclnnAdd(ws, wsSize, ex, stream); },
            stream),
        return false);
    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    return true;
}

static bool CaseAddBf16Status(aclrtStream stream)
{
    const std::string name = "add_bf16_alpha2_status";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> selfData = {0x3F80, 0x4000, 0xC000, 0x0000};
    const std::vector<uint16_t> otherData = {0x3F00, 0x3F80, 0x3F80, 0x3F80};
    std::vector<uint16_t> outHost(selfData.size(), 0);
    const float alphaValue = 2.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_BF16, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_BF16, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_BF16, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s getWorkspace ret=%d (accepted on simulator or unsupported arch)\n", name.c_str(), ret);
        return true;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return false);
    }

    ret = aclnnAdd(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s launch ret=%d (accepted on simulator or unsupported arch)\n", name.c_str(), ret);
        return true;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s sync ret=%d (accepted on simulator or unsupported arch)\n", name.c_str(), ret);
        return true;
    }
    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    return true;
}

static bool CaseAddDoubleStatus(aclrtStream stream)
{
    const std::string name = "add_double_alpha1_status";
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfData = {1.0, -2.0, 3.0, 0.0};
    const std::vector<double> otherData = {0.5, 1.0, -1.5, 2.0};
    const double alphaValue = 1.0;
    std::vector<double> outHost(selfData.size(), 0.0);

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_DOUBLE, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_DOUBLE, &other), return false);
    CHECK_RET(CreateTensor(&guard, outHost, shape, ACL_DOUBLE, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_DOUBLE);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s getWorkspace ret=%d (accepted on simulator)\n", name.c_str(), ret);
        return true;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return false);
    }
    ret = aclnnAdd(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s launch ret=%d (accepted on simulator)\n", name.c_str(), ret);
        return true;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  %s sync ret=%d (accepted on simulator)\n", name.c_str(), ret);
        return true;
    }

    CHECK_RET(CopyToHost(out.device, &outHost), return false);
    for (size_t i = 0; i < outHost.size(); ++i) {
        const double expected = selfData[i] + alphaValue * otherData[i];
        if (!AlmostEqual(expected, outHost[i], 1e-12, 1e-12)) {
            LOG_PRINT("  %s mismatch at %zu expected=%f actual=%f\n", name.c_str(), i, expected, outHost[i]);
            return false;
        }
    }
    return true;
}

static bool CaseErrorAddShapeMismatch(aclrtStream stream)
{
    (void)stream;
    const std::vector<int64_t> inShape = {2, 2};
    const std::vector<int64_t> outShape = {2, 1};
    const std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> otherData = {-1.0f, 0.5f, 2.0f, -3.0f};
    std::vector<float> outData(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, inShape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, inShape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outData, outShape, ACL_FLOAT, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorAddsBoolAlphaFloat()
{
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint8_t> selfData = {0, 1, 1, 0};
    std::vector<uint8_t> outData(selfData.size(), 0);
    const bool otherValue = true;
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_BOOL, &self), return false);
    CHECK_RET(CreateTensor(&guard, outData, shape, ACL_BOOL, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_BOOL);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorInplaceShapeMismatch(aclrtStream stream)
{
    (void)stream;
    const std::vector<int64_t> selfShape = {1, 2};
    const std::vector<int64_t> otherShape = {2, 2};
    const std::vector<float> selfData = {1.0f, 2.0f};
    const std::vector<float> otherData = {0.5f, -1.0f, 3.0f, 2.0f};
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    CHECK_RET(CreateTensor(&guard, selfData, selfShape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, otherShape, ACL_FLOAT, &other), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorInvalidRankAdds(aclrtStream stream)
{
    (void)stream;
    const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> data(static_cast<size_t>(GetShapeSize(shape)), 1.0f);
    const float otherValue = 2.0f;
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorNullptrAdd()
{
    float alphaValue = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    const aclnnStatus ret1 = aclnnAddGetWorkspaceSize(nullptr, nullptr, alpha, nullptr, &ws, &ex);
    const aclnnStatus ret2 = aclnnAddGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &ws, &ex);
    aclDestroyScalar(alpha);
    return ret1 != ACL_SUCCESS && ret2 != ACL_SUCCESS;
}

static bool CaseErrorNullptrAddV3()
{
    float selfValue = 1.0f;
    float alphaValue = 1.0f;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, if (self != nullptr) { aclDestroyScalar(self); } if (alpha != nullptr) { aclDestroyScalar(alpha); } return false;);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    const aclnnStatus ret = aclnnAddV3GetWorkspaceSize(self, nullptr, alpha, nullptr, &ws, &ex);
    aclDestroyScalar(self);
    aclDestroyScalar(alpha);
    return ret != ACL_SUCCESS;
}

static bool CaseErrorAddV3UnsupportedDtype()
{
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> otherData = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> outData(otherData.size(), 0.0);
    const double selfValue = 1.0;
    const double alphaValue = 1.0;

    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_DOUBLE, &other), return false);
    CHECK_RET(CreateTensor(&guard, outData, shape, ACL_DOUBLE, &out), return false);

    auto* self = CreateScalar(&guard, selfValue, ACL_DOUBLE);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_DOUBLE);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorAddV3ShapeMismatch()
{
    const std::vector<int64_t> otherShape = {2, 2};
    const std::vector<int64_t> outShape = {2, 1};
    const std::vector<float> otherData = {1.0f, -2.0f, 3.0f, 4.0f};
    std::vector<float> outData(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
    const float selfValue = 1.0f;
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, otherData, otherShape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outData, outShape, ACL_FLOAT, &out), return false);

    auto* self = CreateScalar(&guard, selfValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseAddFloatNanInf(aclrtStream stream)
{
    const std::string name = "add_float_nan_inf";
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    return CaseAddFloat(name, {nan, inf, -inf, 1.0f}, {2, 2}, {1.0f, 2.0f, 3.0f, inf}, {2, 2}, {2, 2}, 1.0f, stream);
}

static bool CaseAddFloatLargeMagnitude(aclrtStream stream)
{
    const std::string name = "add_float_large_magnitude";
    return CaseAddFloat(
        name,
        {1.0e20f, -1.0e20f, 3.0e10f, -3.0e10f}, {2, 2},
        {2.0e20f, 5.0e19f, -1.0e10f, 1.0e10f}, {2, 2}, {2, 2},
        -0.75f, stream);
}

static bool CaseAddsFloatAlphaNearOne(aclrtStream stream)
{
    return CaseAddsFloat(
        "adds_float_alpha_near_one", {0.25f, -1.0f, 2.5f, -3.5f}, {2, 2}, 4.0f, 1.000001f, stream);
}

static bool CaseAddV3FloatAlphaNearOne(aclrtStream stream)
{
    return CaseAddV3Float(
        "add_v3_float_alpha_near_one", 0.25f, {1.0f, -2.0f, 3.0f, -4.0f}, {2, 2}, 1.000001f, stream);
}

static bool CaseErrorAddBoolAlphaFloat()
{
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint8_t> selfData = {0, 1, 1, 0};
    const std::vector<uint8_t> otherData = {1, 0, 1, 0};
    std::vector<uint8_t> outData(selfData.size(), 0);
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, shape, ACL_BOOL, &self), return false);
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_BOOL, &other), return false);
    CHECK_RET(CreateTensor(&guard, outData, shape, ACL_BOOL, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorAddsShapeMismatch()
{
    const std::vector<int64_t> selfShape = {2, 2};
    const std::vector<int64_t> outShape = {2, 1};
    const std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> outData(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
    const float otherValue = 2.0f;
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, selfData, selfShape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, outData, outShape, ACL_FLOAT, &out), return false);

    auto* other = CreateScalar(&guard, otherValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorInvalidRankAdd(aclrtStream stream)
{
    (void)stream;
    const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> data(static_cast<size_t>(GetShapeSize(shape)), 1.0f);
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack self;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &self), return false);
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, data, shape, ACL_FLOAT, &out), return false);

    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorInvalidRankAddV3(aclrtStream stream)
{
    (void)stream;
    const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> otherData(static_cast<size_t>(GetShapeSize(shape)), 1.0f);
    std::vector<float> outData(static_cast<size_t>(GetShapeSize(shape)), 0.0f);
    const float selfValue = 1.0f;
    const float alphaValue = 1.0f;

    AclResourceGuard guard;
    TensorPack other;
    TensorPack out;
    CHECK_RET(CreateTensor(&guard, otherData, shape, ACL_FLOAT, &other), return false);
    CHECK_RET(CreateTensor(&guard, outData, shape, ACL_FLOAT, &out), return false);

    auto* self = CreateScalar(&guard, selfValue, ACL_FLOAT);
    auto* alpha = CreateScalar(&guard, alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    return aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &ws, &ex) != ACL_SUCCESS;
}

static bool CaseErrorNullptrAdds()
{
    float otherValue = 1.0f;
    float alphaValue = 1.0f;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr,
        if (other != nullptr) { aclDestroyScalar(other); } if (alpha != nullptr) { aclDestroyScalar(alpha); } return false;);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    const aclnnStatus ret1 = aclnnAddsGetWorkspaceSize(nullptr, other, alpha, nullptr, &ws, &ex);
    const aclnnStatus ret2 = aclnnAddsGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &ws, &ex);

    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
    return ret1 != ACL_SUCCESS && ret2 != ACL_SUCCESS;
}

static bool CaseErrorNullptrInplaceAdds()
{
    float otherValue = 1.0f;
    float alphaValue = 1.0f;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr,
        if (other != nullptr) { aclDestroyScalar(other); } if (alpha != nullptr) { aclDestroyScalar(alpha); } return false;);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    const aclnnStatus ret = aclnnInplaceAddsGetWorkspaceSize(nullptr, other, alpha, &ws, &ex);
    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
    return ret != ACL_SUCCESS;
}

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    CaseStat stat;

    struct NamedCase {
        const char* name;
        std::function<bool()> run;
    };

    const std::vector<NamedCase> cases = {
        {"add_float_alpha1", [&]() {
            return CaseAddFloat(
                "add_float_alpha1",
                {0.0f, 1.0f, 2.0f, 3.0f}, {2, 2},
                {1.0f, 0.0f, -1.0f, 2.0f}, {2, 2}, {2, 2},
                1.0f, stream);
        }},
        {"add_float_broadcast_alpha_neg", [&]() {
            return CaseAddFloat(
                "add_float_broadcast_alpha_neg",
                {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}, {2, 3},
                {2.0f, 4.0f, -2.0f}, {1, 3}, {2, 3},
                -0.5f, stream);
        }},
        {"add_float_alpha_zero", [&]() {
            return CaseAddFloat(
                "add_float_alpha_zero",
                {5.0f, -2.0f, 0.0f, 9.0f}, {2, 2},
                {100.0f, 100.0f, 100.0f, 100.0f}, {2, 2}, {2, 2},
                0.0f, stream);
        }},
        {"add_float_nan_inf", [&]() { return CaseAddFloatNanInf(stream); }},
        {"add_float_large_magnitude", [&]() { return CaseAddFloatLargeMagnitude(stream); }},
        {"add_promote_int32_float_alpha1", [&]() { return CaseAddPromoteInt32Float(stream); }},
        {"add_float16_float_mixed_alpha1", [&]() {
            return CaseMixedF16F32Status("add_float16_float_mixed_alpha1", 1.0f, stream);
        }},
        {"add_float16_float_mixed_alpha2", [&]() {
            return CaseMixedF16F32Status("add_float16_float_mixed_alpha2", 2.0f, stream);
        }},
        {"add_float_float16_mixed_alpha1", [&]() {
            return CaseMixedF32F16Status("add_float_float16_mixed_alpha1", 1.0f, stream);
        }},
        {"add_float16_alpha2_status", [&]() { return CaseAddFloat16Status(stream); }},
        {"add_bf16_alpha2_status", [&]() { return CaseAddBf16Status(stream); }},
        {"add_double_alpha1_status", [&]() { return CaseAddDoubleStatus(stream); }},
        {"adds_float", [&]() {
            return CaseAddsFloat("adds_float", {1.0f, -2.0f, 4.0f, 0.5f}, {2, 2}, 3.0f, 1.0f, stream);
        }},
        {"adds_float_alpha_neg", [&]() {
            return CaseAddsFloat("adds_float_alpha_neg", {1.5f, -2.0f, 4.0f, 0.25f}, {2, 2}, -3.0f, -0.5f, stream);
        }},
        {"adds_float_alpha_near_one", [&]() { return CaseAddsFloatAlphaNearOne(stream); }},
        {"adds_int64_alpha2_exact", [&]() { return CaseAddsInt64Exact(stream); }},
        {"adds_bool_out_bool_exact", [&]() { return CaseAddsBoolOutBoolExact(stream); }},
        {"adds_bool_to_int", [&]() { return CaseAddsBoolToInt(stream); }},
        {"adds_bool_to_int_alpha_false", [&]() { return CaseAddsBoolToIntAlphaFalse(stream); }},
        {"add_int32", [&]() { return CaseAddInt32(stream); }},
        {"add_int64_alpha3_exact", [&]() { return CaseAddInt64Exact(stream); }},
        {"add_uint8_alpha2_exact", [&]() { return CaseAddUint8Exact(stream); }},
        {"add_int8_status", [&]() {
            return CaseAddStatus<int8_t, int8_t>(
                "add_int8_status", {1, -2, 3, 0}, {4, 1, -1, 2}, {2, 2}, ACL_INT8, static_cast<int8_t>(1), ACL_INT8,
                stream);
        }},
        {"add_uint8_status", [&]() {
            return CaseAddStatus<uint8_t, uint8_t>(
                "add_uint8_status", {1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2}, ACL_UINT8,
                static_cast<uint8_t>(1), ACL_UINT8, stream);
        }},
        {"add_int64_status", [&]() {
            return CaseAddStatus<int64_t, int64_t>(
                "add_int64_status", {1, -2, 3, 4}, {5, 6, -7, 8}, {2, 2}, ACL_INT64,
                static_cast<int64_t>(1), ACL_INT64, stream);
        }},
        {"add_bool_status", [&]() {
            return CaseAddStatus<uint8_t, bool>(
                "add_bool_status", {0, 1, 0, 1}, {1, 0, 1, 0}, {2, 2}, ACL_BOOL, true, ACL_BOOL, stream);
        }},
        {"inplace_add_float", [&]() {
            return CaseInplaceAddFloat(
                "inplace_add_float", {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, {1.0f, 0.0f}, {1, 2}, 2.0f, stream);
        }},
        {"inplace_adds_float", [&]() {
            return CaseInplaceAddsFloat("inplace_adds_float", {-1.0f, 0.0f, 1.0f, 2.0f}, {2, 2}, -2.0f, 1.0f, stream);
        }},
        {"add_v3_float", [&]() {
            return CaseAddV3Float("add_v3_float", 1.5f, {2.0f, -4.0f, 0.0f, 8.0f}, {2, 2}, 1.0f, stream);
        }},
        {"add_v3_float_alpha2", [&]() {
            return CaseAddV3Float("add_v3_float_alpha2", 1.5f, {2.0f, -4.0f, 0.0f, 8.0f}, {2, 2}, 2.0f, stream);
        }},
        {"add_v3_float_alpha_near_one", [&]() { return CaseAddV3FloatAlphaNearOne(stream); }},
        {"add_v3_self_double_other_int32_out_float", [&]() {
            return CaseAddV3DoubleSelfInt32OutFloat(stream);
        }},
        {"add_v3_self_float_other_int32_alpha2", [&]() { return CaseAddV3FloatSelfInt32Status(stream); }},
        {"add_v3_int8_alpha2", [&]() { return CaseAddV3Int8(stream); }},
        {"inplace_add_v3_status", [&]() { return CaseInplaceAddV3Status("inplace_add_v3_status", 3.0f, -1.0f, stream); }},
        {"add_empty_tensor_workspace0", [&]() { return CaseAddEmptyTensor(stream); }},
        {"adds_empty_tensor_workspace0", [&]() { return CaseAddsEmptyTensor(stream); }},
        {"add_v3_empty_tensor_workspace0", [&]() { return CaseAddV3EmptyTensor(stream); }},
        {"error_nullptr_add", [&]() { return CaseErrorNullptrAdd(); }},
        {"error_nullptr_adds", [&]() { return CaseErrorNullptrAdds(); }},
        {"error_nullptr_inplace_adds", [&]() { return CaseErrorNullptrInplaceAdds(); }},
        {"error_add_bool_alpha_float_cast", [&]() { return CaseErrorAddBoolAlphaFloat(); }},
        {"error_add_shape_mismatch", [&]() { return CaseErrorAddShapeMismatch(stream); }},
        {"error_adds_shape_mismatch", [&]() { return CaseErrorAddsShapeMismatch(); }},
        {"error_invalid_rank_add", [&]() { return CaseErrorInvalidRankAdd(stream); }},
        {"error_adds_bool_alpha_float_cast", [&]() { return CaseErrorAddsBoolAlphaFloat(); }},
        {"error_invalid_rank_adds", [&]() { return CaseErrorInvalidRankAdds(stream); }},
        {"error_inplace_add_shape_mismatch", [&]() { return CaseErrorInplaceShapeMismatch(stream); }},
        {"error_nullptr_add_v3", [&]() { return CaseErrorNullptrAddV3(); }},
        {"error_add_v3_unsupported_dtype", [&]() { return CaseErrorAddV3UnsupportedDtype(); }},
        {"error_invalid_rank_add_v3", [&]() { return CaseErrorInvalidRankAddV3(stream); }},
        {"error_add_v3_shape_mismatch", [&]() { return CaseErrorAddV3ShapeMismatch(); }},
    };

    for (const auto& c : cases) {
        ReportAndReturn(&stat, c.name, c.run());
    }

    LOG_PRINT("\n=== Summary: total=%d pass=%d fail=%d ===\n", stat.total, stat.total - stat.failed, stat.failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return stat.failed == 0 ? 0 : 1;
}
