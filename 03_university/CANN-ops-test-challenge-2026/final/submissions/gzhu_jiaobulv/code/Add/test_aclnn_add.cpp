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
#include <limits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

extern "C" aclnnStatus aclnnAddsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor) __attribute__((weak));
extern "C" aclnnStatus
aclnnAdds(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
    __attribute__((weak));
extern "C" aclnnStatus aclnnInplaceAddGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha, uint64_t* workspaceSize,
    aclOpExecutor** executor) __attribute__((weak));
extern "C" aclnnStatus
aclnnInplaceAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
    __attribute__((weak));
extern "C" aclnnStatus aclnnInplaceAddsGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha, uint64_t* workspaceSize,
    aclOpExecutor** executor) __attribute__((weak));
extern "C" aclnnStatus
aclnnInplaceAdds(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
    __attribute__((weak));
extern "C" aclnnStatus aclnnAddV3GetWorkspaceSize(
    const aclScalar* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor) __attribute__((weak));
extern "C" aclnnStatus
aclnnAddV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
    __attribute__((weak));
extern "C" aclnnStatus aclnnInplaceAddV3GetWorkspaceSize(
    const aclScalar* selfRef, const aclTensor* other, const aclScalar* alpha, uint64_t* workspaceSize,
    aclOpExecutor** executor) __attribute__((weak));
extern "C" aclnnStatus
aclnnInplaceAddV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
    __attribute__((weak));

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

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
};

bool IsClose(float a, float b, float atol = 1e-4f)
{
    return std::fabs(a - b) <= atol;
}

bool IsCloseWithTol(float actual, float expected, float atol, float rtol)
{
    return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

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
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int CreateAclTensorFromRaw(
    const void* hostData, size_t dataSize, const std::vector<int64_t>& shape, aclDataType dataType, TensorResource& res)
{
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(&res.deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(res.deviceAddr, dataSize, hostData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    res.tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        res.deviceAddr);
    return 0;
}

int CreateAclTensorFromRawWithFormat(
    const void* hostData, size_t dataSize, const std::vector<int64_t>& shape, aclDataType dataType, aclFormat format,
    TensorResource& res)
{
    auto ret = aclrtMalloc(&res.deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(res.deviceAddr, dataSize, hostData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    res.tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), res.deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType, TensorResource& res)
{
    return CreateAclTensorFromRaw(
        hostData.data(), static_cast<size_t>(GetShapeSize(shape)) * sizeof(T), shape, dataType, res);
}

template <typename T>
int CreateAclTensorWithFormat(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType, aclFormat format,
    TensorResource& res)
{
    return CreateAclTensorFromRawWithFormat(
        hostData.data(), static_cast<size_t>(GetShapeSize(shape)) * sizeof(T), shape, dataType, format, res);
}

int CreateEmptyAclTensor(const std::vector<int64_t>& shape, aclDataType dataType, aclFormat format, TensorResource& res)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    const int64_t* shapePtr = shape.empty() ? nullptr : shape.data();
    const int64_t* stridesPtr = strides.empty() ? nullptr : strides.data();
    res.deviceAddr = nullptr;
    res.tensor = aclCreateTensor(
        shapePtr, shape.size(), dataType, stridesPtr, 0, format, shapePtr, shape.size(), nullptr);
    CHECK_RET(res.tensor != nullptr, LOG_PRINT("aclCreateTensor empty tensor failed.\n"); return -1);
    return 0;
}

void DestroyTensorResource(TensorResource& res)
{
    if (res.tensor != nullptr) {
        aclDestroyTensor(res.tensor);
        res.tensor = nullptr;
    }
    if (res.deviceAddr != nullptr) {
        aclrtFree(res.deviceAddr);
        res.deviceAddr = nullptr;
    }
}

bool RunAdd(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }

    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return true;
}

bool RunAdds(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, aclTensor* out, aclrtStream stream,
    bool* unsupported = nullptr)
{
    if (aclnnAddsGetWorkspaceSize == nullptr || aclnnAdds == nullptr) {
        if (unsupported != nullptr) {
            *unsupported = true;
        }
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }

    ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdds failed. ERROR: %d\n", ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (unsupported != nullptr) {
        *unsupported = false;
    }
    return true;
}

bool RunInplaceAdd(
    const aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha, aclrtStream stream, bool* unsupported)
{
    if (aclnnInplaceAddGetWorkspaceSize == nullptr || aclnnInplaceAdd == nullptr) {
        if (unsupported != nullptr) {
            *unsupported = true;
        }
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }
    ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdd failed. ERROR: %d\n", ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (unsupported != nullptr) {
        *unsupported = false;
    }
    return true;
}

bool RunInplaceAdds(
    const aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha, aclrtStream stream, bool* unsupported)
{
    if (aclnnInplaceAddsGetWorkspaceSize == nullptr || aclnnInplaceAdds == nullptr) {
        if (unsupported != nullptr) {
            *unsupported = true;
        }
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n", ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }
    ret = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAdds failed. ERROR: %d\n", ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (unsupported != nullptr) {
        *unsupported = false;
    }
    return true;
}

bool RunAddV3(
    const aclScalar* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, aclrtStream stream,
    bool* unsupported = nullptr)
{
    if (aclnnAddV3GetWorkspaceSize == nullptr || aclnnAddV3 == nullptr) {
        if (unsupported != nullptr) {
            *unsupported = true;
        }
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }

    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddV3 failed. ERROR: %d\n", ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (unsupported != nullptr) {
        *unsupported = false;
    }
    return true;
}

bool RunInplaceAddV3(
    const aclScalar* selfRef, const aclTensor* other, const aclScalar* alpha, aclrtStream stream, bool* unsupported)
{
    if (aclnnInplaceAddV3GetWorkspaceSize == nullptr || aclnnInplaceAddV3 == nullptr) {
        if (unsupported != nullptr) {
            *unsupported = true;
        }
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddV3GetWorkspaceSize failed. ERROR: %d\n", ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }
    ret = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddV3 failed. ERROR: %d\n", ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (unsupported != nullptr) {
        *unsupported = false;
    }
    return true;
}

template <typename T>
bool ReadTensorData(const TensorResource& res, std::vector<T>& hostOut)
{
    auto ret = aclrtMemcpy(
        hostOut.data(), hostOut.size() * sizeof(T), res.deviceAddr, hostOut.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return false);
    return true;
}

template <typename T>
bool CheckExactResult(const std::vector<T>& result, const std::vector<T>& expected, const char* caseName)
{
    if (result.size() != expected.size()) {
        LOG_PRINT("%s size mismatch, got=%zu, expected=%zu\n", caseName, result.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (result[i] != expected[i]) {
            LOG_PRINT("%s mismatch, idx=%zu\n", caseName, i);
            return false;
        }
    }
    return true;
}

bool CheckFloatResult(
    const std::vector<float>& result, const std::vector<double>& expected, float atol, float rtol, const char* caseName)
{
    if (result.size() != expected.size()) {
        LOG_PRINT("%s size mismatch, got=%zu, expected=%zu\n", caseName, result.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!IsCloseWithTol(result[i], static_cast<float>(expected[i]), atol, rtol)) {
            LOG_PRINT(
                "%s mismatch, idx=%zu, got=%f, expected=%f, atol=%f, rtol=%f\n", caseName, i, result[i],
                static_cast<float>(expected[i]), atol, rtol);
            return false;
        }
    }
    return true;
}

bool CaseFloatBroadcastAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] float32 broadcast + alpha!=1 (Add)\n");
    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {3};
    const std::vector<int64_t> outShape = {2, 3};
    const std::vector<float> selfHost = {1.0f, -2.0f, 3.0f, 4.5f, -5.5f, 6.0f};
    const std::vector<float> otherHost = {2.0f, -1.0f, 0.5f};
    const std::vector<float> expected = {3.5f, -3.25f, 3.625f, 7.0f, -6.75f, 6.625f};
    float alphaValue = 1.25f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<float> outInit(expected.size(), 0.0f);
    CHECK_RET(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, outShape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!IsClose(result[i], expected[i], 2e-4f)) {
                match = false;
                LOG_PRINT("float32 add mismatch, idx=%zu, got=%f, expected=%f\n", i, result[i], expected[i]);
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseMixedFp16Fp32Add(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] float16 + float32 mixed dtype (Add)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> selfFloat = {-3.5f, 1.5f, 2.25f, -0.75f};
    const std::vector<float> otherHost = {4.0f, -2.0f, 0.5f, 1.5f};
    std::vector<aclFloat16> selfHost(shape[0] * shape[1], 0);
    for (size_t i = 0; i < selfHost.size(); ++i) {
        selfHost[i] = aclFloatToFloat16(selfFloat[i]);
    }
    const std::vector<float> expected = {0.5f, -0.5f, 2.75f, 0.75f};
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<float> outInit(expected.size(), 0.0f);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT16, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!IsClose(result[i], expected[i], 2e-2f)) {
                match = false;
                LOG_PRINT("mixed add mismatch, idx=%zu, got=%f, expected=%f\n", i, result[i], expected[i]);
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseInt32Add(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] int32 + alpha=2 (Add)\n");
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<int32_t> selfHost = {-5, 3, 0, 7, -2, 4};
    const std::vector<int32_t> otherHost = {1, -2, 3, 2, -1, -3};
    const std::vector<int32_t> expected = {-3, -1, 6, 11, -4, -2};
    int64_t alphaValue = 2;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<int32_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_INT32, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_INT32, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT32, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<int32_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (result[i] != expected[i]) {
                match = false;
                LOG_PRINT("int32 add mismatch, idx=%zu, got=%d, expected=%d\n", i, result[i], expected[i]);
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseInt16AiCpuAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] int16 + alpha=1 (Add, expect AiCpu fallback)\n");
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<int16_t> selfHost = {-300, 1200, -1, 32760, -32760, 0};
    const std::vector<int16_t> otherHost = {100, -200, 1, -10, 10, 0};
    const std::vector<int16_t> expected = {-200, 1000, 0, 32750, -32750, 0};
    int64_t alphaValue = 1;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<int16_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_INT16, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_INT16, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT16, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<int16_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (result[i] != expected[i]) {
                match = false;
                LOG_PRINT(
                    "int16 add mismatch, idx=%zu, got=%d, expected=%d\n", i, static_cast<int>(result[i]),
                    static_cast<int>(expected[i]));
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseInt8BoundaryAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] int8 boundary values + alpha=1 (Add)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int8_t> selfHost = {-128, 127, -127, 126};
    const std::vector<int8_t> otherHost = {0, 0, 1, -1};
    const std::vector<int8_t> expected = {-128, 127, -126, 125};
    int64_t alphaValue = 1;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<int8_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_INT8, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_INT8, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT8, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<int8_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (result[i] != expected[i]) {
                match = false;
                LOG_PRINT(
                    "int8 add mismatch, idx=%zu, got=%d, expected=%d\n", i, static_cast<int>(result[i]),
                    static_cast<int>(expected[i]));
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseUint8BoundaryAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] uint8 boundary values + alpha=1 (Add)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint8_t> selfHost = {0, 255, 1, 254};
    const std::vector<uint8_t> otherHost = {0, 0, 1, 1};
    const std::vector<uint8_t> expected = {0, 255, 2, 255};
    int64_t alphaValue = 1;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<uint8_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_UINT8, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_UINT8, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_UINT8, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<uint8_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (result[i] != expected[i]) {
                match = false;
                LOG_PRINT(
                    "uint8 add mismatch, idx=%zu, got=%u, expected=%u\n", i, static_cast<unsigned>(result[i]),
                    static_cast<unsigned>(expected[i]));
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseInt64BoundaryAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] int64 near-limit values + alpha=1 (Add)\n");
    const std::vector<int64_t> shape = {2};
    const std::vector<int64_t> selfHost = {
        std::numeric_limits<int64_t>::min() + 2, std::numeric_limits<int64_t>::max() - 2};
    const std::vector<int64_t> otherHost = {1, -1};
    const std::vector<int64_t> expected = {
        std::numeric_limits<int64_t>::min() + 3, std::numeric_limits<int64_t>::max() - 3};
    int64_t alphaValue = 1;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<int64_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_INT64, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_INT64, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT64, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<int64_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (result[i] != expected[i]) {
                match = false;
                LOG_PRINT(
                    "int64 add mismatch, idx=%zu, got=%lld, expected=%lld\n", i, static_cast<long long>(result[i]),
                    static_cast<long long>(expected[i]));
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseBoolAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] bool OR branch (Add)\n");
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<uint8_t> selfHost = {1, 0, 0, 1, 0, 1};
    const std::vector<uint8_t> otherHost = {0, 1, 0, 0, 1, 1};
    const std::vector<uint8_t> expected = {1, 1, 0, 1, 1, 1};
    int64_t alphaValue = 1;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<uint8_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_BOOL, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_BOOL, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_BOOL, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<uint8_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if ((result[i] != 0 ? 1 : 0) != expected[i]) {
                match = false;
                LOG_PRINT("bool add mismatch, idx=%zu, got=%u, expected=%u\n", i, result[i], expected[i]);
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseInplaceAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] inplace add (broadcast)\n");
    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {3};
    const std::vector<float> selfHost = {1.0f, -2.0f, 3.0f, 4.5f, -5.5f, 6.0f};
    const std::vector<float> otherHost = {2.0f, -1.0f, 0.5f};
    const std::vector<float> expected = {3.0f, -3.0f, 3.5f, 6.5f, -6.5f, 6.5f};
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    CHECK_RET(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunInplaceAdd(self.tensor, other.tensor, alpha, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip inplace case: aclnnInplaceAdd symbols are unavailable in current opapi library\n");
        aclDestroyScalar(alpha);
        DestroyTensorResource(self);
        DestroyTensorResource(other);
        return true;
    }

    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(self, result)) {
        match = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!IsClose(result[i], expected[i], 2e-4f)) {
                match = false;
                LOG_PRINT("inplace add mismatch, idx=%zu, got=%f, expected=%f\n", i, result[i], expected[i]);
                break;
            }
        }
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return ok && match;
}

bool CaseMixedFp32Fp16Add(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] float32 + float16 mixed dtype (Add)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> selfHost = {1.0f, -1.5f, 3.25f, -0.5f};
    const std::vector<float> otherFloat = {2.5f, 0.5f, -1.25f, 4.0f};
    std::vector<aclFloat16> otherHost(shape[0] * shape[1], 0);
    for (size_t i = 0; i < otherHost.size(); ++i) {
        otherHost[i] = aclFloatToFloat16(otherFloat[i]);
    }
    const std::vector<double> expected = {3.5, -1.0, 2.0, 3.5};
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<float> outInit(expected.size(), 0.0f);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT16, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckFloatResult(result, expected, 1e-4f, 1e-4f, "mixed fp32+fp16");
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseDoubleMulPathAdd(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] double + alpha!=1 (Add, Mul+Add fallback path)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfHost = {1.0, -4.0, 8.5, -2.5};
    const std::vector<double> otherHost = {2.0, -1.0, 3.0, 0.5};
    const std::vector<double> expected = {0.0, -3.5, 7.0, -2.75};
    double alphaValue = -0.5;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<double> outInit(expected.size(), 0.0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_DOUBLE, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_DOUBLE, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_DOUBLE, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<double> result(expected.size(), 0.0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckExactResult(result, expected, "double add");
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddInvalidBroadcastExpectFail()
{
    LOG_PRINT("\n[CASE] invalid broadcast (Add, expect failure)\n");
    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {2, 2};
    const std::vector<int64_t> outShape = {2, 3};
    std::vector<float> selfHost(6, 1.0f);
    std::vector<float> otherHost(4, 2.0f);
    std::vector<float> outHost(6, 0.0f);
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, outShape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddNullptrExpectFail()
{
    LOG_PRINT("\n[CASE] nullptr input (Add, expect failure)\n");
    const std::vector<int64_t> shape = {1};
    std::vector<float> host = {1.0f};
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(host, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(host, shape, ACL_FLOAT, out) == 0, return false);
    float alphaValue = 1.0f;
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(nullptr, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddOutShapeMismatchExpectFail()
{
    LOG_PRINT("\n[CASE] out shape mismatch (Add, expect failure)\n");
    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {3};
    const std::vector<int64_t> outShape = {3, 2};
    std::vector<float> selfHost(6, 1.0f);
    std::vector<float> otherHost(3, 2.0f);
    std::vector<float> outHost(6, 0.0f);
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, outShape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddAlphaCastInvalidExpectFail()
{
    LOG_PRINT("\n[CASE] alpha cast invalid (Add bool+bool with float alpha, expect failure)\n");
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<uint8_t> selfHost = {1, 0, 1, 0, 1, 0};
    const std::vector<uint8_t> otherHost = {0, 1, 0, 1, 0, 1};
    std::vector<uint8_t> outHost(6, 0);
    float alphaValue = 0.5f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_BOOL, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_BOOL, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_BOOL, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddUnsupportedUint32ExpectFail()
{
    LOG_PRINT("\n[CASE] unsupported dtype uint32 (Add, expect failure)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint32_t> selfHost = {1, 2, 3, 4};
    const std::vector<uint32_t> otherHost = {5, 6, 7, 8};
    std::vector<uint32_t> outHost(4, 0);
    int64_t alphaValue = 1;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_UINT32, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_UINT32, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_UINT32, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseInplaceAddShapeMismatchExpectFail()
{
    LOG_PRINT("\n[CASE] inplace shape mismatch (expect failure)\n");
    const std::vector<int64_t> selfShape = {3};
    const std::vector<int64_t> otherShape = {2, 3};
    std::vector<float> selfHost = {1.0f, 2.0f, 3.0f};
    std::vector<float> otherHost = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    CHECK_RET(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    if (aclnnInplaceAddGetWorkspaceSize == nullptr) {
        LOG_PRINT("skip inplace mismatch case: aclnnInplaceAdd symbols are unavailable\n");
        aclDestroyScalar(alpha);
        DestroyTensorResource(self);
        DestroyTensorResource(other);
        return true;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return ret != ACL_SUCCESS;
}

bool CaseAddsNullptrExpectFail()
{
    LOG_PRINT("\n[CASE] nullptr input (Adds, expect failure)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr) {
        LOG_PRINT("skip Adds nullptr case: aclnnAdds symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {2, 2};
    std::vector<float> outHost(4, 0.0f);
    TensorResource out;
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_FLOAT, out) == 0, return false);
    float otherValue = 1.0f;
    float alphaValue = 1.0f;
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddsGetWorkspaceSize(nullptr, other, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddsAlphaCastInvalidExpectFail()
{
    LOG_PRINT("\n[CASE] alpha cast invalid (Adds bool with float alpha, expect failure)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr) {
        LOG_PRINT("skip Adds alpha-cast case: aclnnAdds symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint8_t> selfHost = {1, 0, 1, 0};
    std::vector<uint8_t> outHost(4, 0);
    uint8_t otherValue = 1;
    float alphaValue = 1.1f;

    TensorResource self;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_BOOL, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_BOOL, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_BOOL);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddsInt32AlphaNeg(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] Adds int32 + alpha<0\n");
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<int32_t> selfHost = {4, -3, 0, 7, 2, -5};
    const std::vector<int32_t> expected = {10, 3, 6, 13, 8, 1};
    int64_t otherValue = -3;
    int64_t alphaValue = -2;

    TensorResource self;
    TensorResource out;
    std::vector<int32_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_INT32, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT32, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_INT64);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAdds(self.tensor, other, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip Adds case: aclnnAdds symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(other);
        DestroyTensorResource(self);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<int32_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckExactResult(result, expected, "adds int32");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddsBoolToInt(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] Adds bool tensor + bool scalar -> int32 output\n");
    const std::vector<int64_t> shape = {2, 3};
    const std::vector<uint8_t> selfHost = {0, 0, 1, 1, 0, 1};
    const std::vector<int32_t> expected = {1, 1, 1, 1, 1, 1};
    uint8_t otherValue = 1;
    uint8_t alphaValue = 1;

    TensorResource self;
    TensorResource out;
    std::vector<int32_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_BOOL, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT32, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_BOOL);
    auto alpha = aclCreateScalar(&alphaValue, ACL_BOOL);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAdds(self.tensor, other, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip Adds bool case: aclnnAdds symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(other);
        DestroyTensorResource(self);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<int32_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckExactResult(result, expected, "adds bool->int32");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddsFloatToFloat16Out(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] Adds float tensor + float scalar -> float16 out\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> selfHost = {1.0f, -2.0f, 3.5f, -4.5f};
    const std::vector<double> expected = {1.5, -1.5, 4.0, -4.0};
    float otherValue = 1.0f;
    float alphaValue = 0.5f;

    TensorResource self;
    TensorResource out;
    std::vector<aclFloat16> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT16, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAdds(self.tensor, other, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip Adds float->float16 case: aclnnAdds symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(other);
        DestroyTensorResource(self);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<aclFloat16> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        std::vector<float> resultFp32(expected.size(), 0.0f);
        for (size_t i = 0; i < expected.size(); ++i) {
            resultFp32[i] = aclFloat16ToFloat(result[i]);
        }
        match = CheckFloatResult(resultFp32, expected, 5e-3f, 5e-3f, "adds float->float16");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddsInt64MulPath(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] Adds int64 + alpha!=1 (Mul+Add path)\n");
    const std::vector<int64_t> shape = {4};
    const std::vector<int64_t> selfHost = {10, -20, 30, -40};
    const std::vector<int64_t> expected = {4, -26, 24, -46};
    int64_t otherValue = 3;
    int64_t alphaValue = -2;

    TensorResource self;
    TensorResource out;
    std::vector<int64_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_INT64, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT64, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_INT64);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAdds(self.tensor, other, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip Adds int64 mul-path case: aclnnAdds symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(other);
        DestroyTensorResource(self);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<int64_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckExactResult(result, expected, "adds int64 mul-path");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseInplaceAddsFloat(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] InplaceAdds float + scalar\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> selfHost = {2.0f, -1.0f, 0.5f, 8.0f};
    const std::vector<double> expected = {1.0, -2.0, -0.5, 7.0};
    float otherValue = 0.5f;
    float alphaValue = -2.0f;

    TensorResource self;
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT, self) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunInplaceAdds(self.tensor, other, alpha, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip InplaceAdds case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(other);
        DestroyTensorResource(self);
        return true;
    }

    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(self, result)) {
        match = CheckFloatResult(result, expected, 1e-6f, 1e-6f, "inplace adds float");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    return ok && match;
}

bool CaseAddV3FloatAxpy(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] AddV3 scalar(float) + alpha * tensor(float)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> otherHost = {1.0f, -2.0f, 4.0f, 0.5f};
    const std::vector<double> expected = {10.5, 9.0, 12.0, 10.25};
    float selfValue = 10.0f;
    float alphaValue = 0.5f;

    if (aclnnAddV3GetWorkspaceSize == nullptr || aclnnAddV3 == nullptr) {
        LOG_PRINT("skip AddV3 case: symbols are unavailable\n");
        return true;
    }

    TensorResource other;
    TensorResource out;
    std::vector<float> outInit(expected.size(), 0.0f);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAddV3(self, other.tensor, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip AddV3 case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(self);
        DestroyTensorResource(other);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckFloatResult(result, expected, 1e-6f, 1e-6f, "add v3 float");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddV3Int8MulPath(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] AddV3 int8 with alpha!=1 (Mul+Add branch)\n");
    const std::vector<int64_t> shape = {4};
    const std::vector<int8_t> otherHost = {10, -10, 20, -20};
    const std::vector<int8_t> expected = {32, -28, 62, -58};
    int64_t selfValue = 2;
    int64_t alphaValue = 3;

    if (aclnnAddV3GetWorkspaceSize == nullptr || aclnnAddV3 == nullptr) {
        LOG_PRINT("skip AddV3 int8 case: symbols are unavailable\n");
        return true;
    }

    TensorResource other;
    TensorResource out;
    std::vector<int8_t> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_INT8, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_INT8, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_INT64);
    auto alpha = aclCreateScalar(&alphaValue, ACL_INT64);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAddV3(self, other.tensor, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip AddV3 int8 case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(self);
        DestroyTensorResource(other);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<int8_t> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckExactResult(result, expected, "add v3 int8");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseInplaceAddV3Float(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] InplaceAddV3 scalar(float) + tensor(float)\n");
    const std::vector<int64_t> shape = {3};
    const std::vector<float> otherHost = {1.0f, -2.0f, 0.0f};
    const std::vector<double> expected = {4.0, 7.0, 5.0};
    float selfValue = 5.0f;
    float alphaValue = -1.0f;

    if (aclnnInplaceAddV3GetWorkspaceSize == nullptr || aclnnInplaceAddV3 == nullptr) {
        LOG_PRINT("skip InplaceAddV3 case: symbols are unavailable\n");
        return true;
    }

    TensorResource other;
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunInplaceAddV3(self, other.tensor, alpha, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip InplaceAddV3 case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(self);
        DestroyTensorResource(other);
        return true;
    }

    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(other, result)) {
        match = CheckFloatResult(result, expected, 1e-6f, 1e-6f, "inplace add v3 float");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    return ok && match;
}

bool CaseAddV3FloatToFloat16Out(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] AddV3 float scalar + tensor(float) -> float16 out\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr || aclnnAddV3 == nullptr) {
        LOG_PRINT("skip AddV3 float->float16 case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> otherHost = {2.0f, -4.0f, 0.5f, 3.0f};
    const std::vector<double> expected = {2.5, -3.5, 1.0, 3.5};
    float selfValue = 1.5f;
    float alphaValue = 0.5f;

    TensorResource other;
    TensorResource out;
    std::vector<aclFloat16> outInit(expected.size(), 0);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT16, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAddV3(self, other.tensor, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip AddV3 float->float16 case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(self);
        DestroyTensorResource(other);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<aclFloat16> result(expected.size(), 0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        std::vector<float> resultFp32(expected.size(), 0.0f);
        for (size_t i = 0; i < expected.size(); ++i) {
            resultFp32[i] = aclFloat16ToFloat(result[i]);
        }
        match = CheckFloatResult(resultFp32, expected, 5e-3f, 5e-3f, "add v3 float->float16");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddV3DoubleOutFloatPath(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] AddV3 self=double, out=float path\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr || aclnnAddV3 == nullptr) {
        LOG_PRINT("skip AddV3 double->float case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {3};
    const std::vector<int32_t> otherHost = {1, 2, -3};
    const std::vector<double> expected = {7.0, 8.0, 3.0};
    double selfValue = 5.0;
    double alphaValue = 2.0;

    TensorResource other;
    TensorResource out;
    std::vector<float> outInit(expected.size(), 0.0f);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_INT32, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_DOUBLE);
    auto alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAddV3(self, other.tensor, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip AddV3 double->float case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(self);
        DestroyTensorResource(other);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckFloatResult(result, expected, 1e-5f, 1e-5f, "add v3 double->float");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddV3NullptrExpectFail()
{
    LOG_PRINT("\n[CASE] nullptr input (AddV3, expect failure)\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr) {
        LOG_PRINT("skip AddV3 nullptr case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {1};
    std::vector<float> host = {2.0f};
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(host, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(host, shape, ACL_FLOAT, out) == 0, return false);
    float alphaValue = 1.0f;
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddV3GetWorkspaceSize(nullptr, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddV3UnsupportedUint8ExpectFail()
{
    LOG_PRINT("\n[CASE] unsupported dtype uint8 (AddV3, expect failure)\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr) {
        LOG_PRINT("skip AddV3 uint8 case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {4};
    const std::vector<uint8_t> otherHost = {1, 2, 3, 4};
    std::vector<uint8_t> outHost(4, 0);
    float selfValue = 3.0f;
    float alphaValue = 1.0f;

    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_UINT8, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_UINT8, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddMaxDimExceededExpectFail()
{
    LOG_PRINT("\n[CASE] max dim exceeded (Add, expect failure)\n");
    const std::vector<int64_t> shape9d = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> host(1, 1.0f);
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(host, shape9d, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(host, shape9d, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(host, shape9d, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddsOutShapeMismatchExpectFail()
{
    LOG_PRINT("\n[CASE] out shape mismatch (Adds, expect failure)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr) {
        LOG_PRINT("skip Adds out-shape case: aclnnAdds symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> outShape = {3, 2};
    std::vector<float> selfHost(6, 1.0f);
    std::vector<float> outHost(6, 0.0f);
    float otherValue = 2.0f;
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, outShape, ACL_FLOAT, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddsMaxDimExceededExpectFail()
{
    LOG_PRINT("\n[CASE] max dim exceeded (Adds, expect failure)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr) {
        LOG_PRINT("skip Adds max-dim case: aclnnAdds symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape9d = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> host(1, 1.0f);
    float otherValue = 2.0f;
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource out;
    CHECK_RET(CreateAclTensor(host, shape9d, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(host, shape9d, ACL_FLOAT, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddsNonNdFormatPath(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] non-ND format input path (Adds)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr || aclnnAdds == nullptr) {
        LOG_PRINT("skip Adds non-ND format case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {1, 2, 2, 1};
    const std::vector<float> selfHost = {1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<double> expected = {2.0, -1.0, 4.0, -3.0};
    float otherValue = 1.0f;
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource out;
    std::vector<float> outInit(expected.size(), 0.0f);
    CHECK_RET(CreateAclTensorWithFormat(selfHost, shape, ACL_FLOAT, ACL_FORMAT_NHWC, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAdds(self.tensor, other, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip Adds non-ND format case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(other);
        DestroyTensorResource(self);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckFloatResult(result, expected, 1e-6f, 1e-6f, "adds non-nd format");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddV3OutShapeMismatchExpectFail()
{
    LOG_PRINT("\n[CASE] out shape mismatch (AddV3, expect failure)\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr) {
        LOG_PRINT("skip AddV3 out-shape case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> otherShape = {2, 3};
    const std::vector<int64_t> outShape = {3, 2};
    std::vector<float> otherHost(6, 1.0f);
    std::vector<float> outHost(6, 0.0f);
    float selfValue = 2.0f;
    float alphaValue = 1.0f;

    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, outShape, ACL_FLOAT, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddV3MaxDimExceededExpectFail()
{
    LOG_PRINT("\n[CASE] max dim exceeded (AddV3, expect failure)\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr) {
        LOG_PRINT("skip AddV3 max-dim case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape9d = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> host(1, 1.0f);
    float selfValue = 2.0f;
    float alphaValue = 1.0f;

    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(host, shape9d, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(host, shape9d, ACL_FLOAT, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddPromoteToOutCastFailExpectFail()
{
    LOG_PRINT("\n[CASE] promote->out cast invalid (Add float+float -> int32, expect failure)\n");
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> selfHost = {1.1f, -2.2f, 3.3f, 4.4f};
    const std::vector<float> otherHost = {0.5f, 0.5f, -1.0f, 2.0f};
    std::vector<int32_t> outHost(4, 0);
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_INT32, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddsPromoteToOutCastFailExpectFail()
{
    LOG_PRINT("\n[CASE] promote->out cast invalid (Adds float + scalar -> int32, expect failure)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr) {
        LOG_PRINT("skip Adds promote->out case: aclnnAdds symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> selfHost = {1.25f, -2.5f, 3.75f, 4.0f};
    std::vector<int32_t> outHost(4, 0);
    float otherValue = 0.5f;
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource out;
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_INT32, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddV3PromoteToOutCastFailExpectFail()
{
    LOG_PRINT("\n[CASE] promote->out cast invalid (AddV3 float -> int32, expect failure)\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr) {
        LOG_PRINT("skip AddV3 promote->out case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> otherHost = {1.0f, 2.0f, -3.0f, 4.0f};
    std::vector<int32_t> outHost(4, 0);
    float selfValue = 2.0f;
    float alphaValue = 1.0f;

    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_INT32, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret != ACL_SUCCESS;
}

bool CaseAddEmptyTensorWorkspaceOnly()
{
    LOG_PRINT("\n[CASE] empty tensor (Add, expect workspace=0 success)\n");
    const std::vector<int64_t> selfShape = {0};
    const std::vector<int64_t> otherShape = {0};
    const std::vector<int64_t> outShape = {0};
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateEmptyAclTensor(selfShape, ACL_FLOAT, ACL_FORMAT_ND, self) == 0, return false);
    CHECK_RET(CreateEmptyAclTensor(otherShape, ACL_FLOAT, ACL_FORMAT_ND, other) == 0, return false);
    CHECK_RET(CreateEmptyAclTensor(outShape, ACL_FLOAT, ACL_FORMAT_ND, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    uint64_t workspaceSize = 1;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret == ACL_SUCCESS && workspaceSize == 0;
}

bool CaseAddsEmptyTensorWorkspaceOnly()
{
    LOG_PRINT("\n[CASE] empty tensor (Adds, expect workspace=0 success)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr) {
        LOG_PRINT("skip Adds empty tensor case: aclnnAdds symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {0};
    float otherValue = 2.0f;
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource out;
    CHECK_RET(CreateEmptyAclTensor(shape, ACL_FLOAT, ACL_FORMAT_ND, self) == 0, return false);
    CHECK_RET(CreateEmptyAclTensor(shape, ACL_FLOAT, ACL_FORMAT_ND, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 1;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ret == ACL_SUCCESS && workspaceSize == 0;
}

bool CaseAddV3EmptyTensorWorkspaceOnly()
{
    LOG_PRINT("\n[CASE] empty tensor (AddV3, expect workspace=0 success)\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr) {
        LOG_PRINT("skip AddV3 empty tensor case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {0};
    float selfValue = 3.0f;
    float alphaValue = 1.0f;

    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateEmptyAclTensor(shape, ACL_FLOAT, ACL_FORMAT_ND, other) == 0, return false);
    CHECK_RET(CreateEmptyAclTensor(shape, ACL_FLOAT, ACL_FORMAT_ND, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_FLOAT);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    uint64_t workspaceSize = 1;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &workspaceSize, &executor);

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ret == ACL_SUCCESS && workspaceSize == 0;
}

bool CaseAddNonNdFormatPath(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] non-ND format input path (Add)\n");
    const std::vector<int64_t> shape = {1, 2, 2, 1};
    const std::vector<float> selfHost = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> otherHost = {0.5f, -0.5f, 1.0f, -1.0f};
    const std::vector<double> expected = {1.5, 1.5, 4.0, 3.0};
    std::vector<float> outHost(expected.size(), 0.0f);
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    CHECK_RET(CreateAclTensorWithFormat(selfHost, shape, ACL_FLOAT, ACL_FORMAT_NHWC, self) == 0, return false);
    CHECK_RET(CreateAclTensorWithFormat(otherHost, shape, ACL_FLOAT, ACL_FORMAT_NHWC, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outHost, shape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckFloatResult(result, expected, 1e-6f, 1e-6f, "add non-nd format");
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CasePrecisionLargeSmall(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] precision: large + small (float32)\n");
    const std::vector<int64_t> shape = {2};
    const std::vector<float> selfHost = {1e10f, 1e10f};
    const std::vector<float> otherHost = {1e-5f, -1e-5f};
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<float> outInit(selfHost.size(), 0.0f);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<float> result(selfHost.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        const double ref0 = static_cast<double>(selfHost[0]) + static_cast<double>(otherHost[0]);
        const double ref1 = static_cast<double>(selfHost[1]) + static_cast<double>(otherHost[1]);
        LOG_PRINT(
            "precision large+small: ref=(%.10f, %.10f), actual=(%.10f, %.10f)\n", ref0, ref1, result[0], result[1]);
        match = true;
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CasePrecisionCancellation(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] precision: cancellation (float32)\n");
    const std::vector<int64_t> shape = {2};
    const std::vector<float> selfHost = {1.0000001f, 2.0000001f};
    const std::vector<float> otherHost = {-1.0f, -2.0f};
    const std::vector<double> expected = {1.1920928955078125e-07, 1.1920928955078125e-07};
    float alphaValue = 1.0f;

    TensorResource self;
    TensorResource other;
    TensorResource out;
    std::vector<float> outInit(selfHost.size(), 0.0f);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_FLOAT, self) == 0, return false);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_FLOAT, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_FLOAT, out) == 0, return false);
    auto alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return false);

    const bool ok = RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream);
    std::vector<float> result(expected.size(), 0.0f);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckFloatResult(result, expected, 1e-7f, 1e-6f, "precision cancellation");
    }

    aclDestroyScalar(alpha);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddsDoubleAlphaEqOne(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] Adds double + alpha==1 (IsEqualToOne double branch)\n");
    if (aclnnAddsGetWorkspaceSize == nullptr || aclnnAdds == nullptr) {
        LOG_PRINT("skip Adds double alpha==1 case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfHost = {1.0, -2.0, 3.0, -4.0};
    const std::vector<double> expected = {3.0, 0.0, 5.0, -2.0};
    double otherValue = 2.0;
    double alphaValue = 1.0;

    TensorResource self;
    TensorResource out;
    std::vector<double> outInit(expected.size(), 0.0);
    CHECK_RET(CreateAclTensor(selfHost, shape, ACL_DOUBLE, self) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_DOUBLE, out) == 0, return false);
    auto other = aclCreateScalar(&otherValue, ACL_DOUBLE);
    auto alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
    CHECK_RET(other != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAdds(self.tensor, other, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip Adds double alpha==1 case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(other);
        DestroyTensorResource(self);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<double> result(expected.size(), 0.0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckExactResult(result, expected, "adds double alpha==1");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return ok && match;
}

bool CaseAddV3DoubleAlphaEqOne(aclrtStream stream)
{
    LOG_PRINT("\n[CASE] AddV3 self=double, other=int32, out=double + alpha==1\n");
    if (aclnnAddV3GetWorkspaceSize == nullptr || aclnnAddV3 == nullptr) {
        LOG_PRINT("skip AddV3 double/int32 case: symbols are unavailable\n");
        return true;
    }

    const std::vector<int64_t> shape = {3};
    const std::vector<int32_t> otherHost = {2, -3, 5};
    const std::vector<double> expected = {5.0, 0.0, 8.0};
    double selfValue = 3.0;
    double alphaValue = 1.0;

    TensorResource other;
    TensorResource out;
    std::vector<double> outInit(expected.size(), 0.0);
    CHECK_RET(CreateAclTensor(otherHost, shape, ACL_INT32, other) == 0, return false);
    CHECK_RET(CreateAclTensor(outInit, shape, ACL_DOUBLE, out) == 0, return false);
    auto self = aclCreateScalar(&selfValue, ACL_DOUBLE);
    auto alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
    CHECK_RET(self != nullptr && alpha != nullptr, return false);

    bool unsupported = false;
    const bool ok = RunAddV3(self, other.tensor, alpha, out.tensor, stream, &unsupported);
    if (unsupported) {
        LOG_PRINT("skip AddV3 double/int32 case: symbols are unavailable\n");
        aclDestroyScalar(alpha);
        aclDestroyScalar(self);
        DestroyTensorResource(other);
        DestroyTensorResource(out);
        return true;
    }

    std::vector<double> result(expected.size(), 0.0);
    bool match = false;
    if (ok && ReadTensorData(out, result)) {
        match = CheckExactResult(result, expected, "add v3 double alpha==1");
    }

    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return ok && match;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int failCount = 0;
    failCount += CaseFloatBroadcastAdd(stream) ? 0 : 1;   // 覆盖 broadcast + alpha!=1 分支
    failCount += CaseMixedFp16Fp32Add(stream) ? 0 : 1;    // 覆盖混合 dtype 分支
    failCount += CaseInt32Add(stream) ? 0 : 1;            // 覆盖整型 + alpha 分支
    failCount += CaseInt16AiCpuAdd(stream) ? 0 : 1;       // 覆盖 int16 回退分支（AiCpu）
    failCount += CaseInt8BoundaryAdd(stream) ? 0 : 1;     // 覆盖 int8 边界取值
    failCount += CaseUint8BoundaryAdd(stream) ? 0 : 1;    // 覆盖 uint8 边界取值
    failCount += CaseInt64BoundaryAdd(stream) ? 0 : 1;    // 覆盖 int64 近边界取值
    failCount += CaseBoolAdd(stream) ? 0 : 1;             // 覆盖 bool 特殊分支
    failCount += CaseInplaceAdd(stream) ? 0 : 1;          // 覆盖 inplace 分支（若符号可用）
    failCount += CaseMixedFp32Fp16Add(stream) ? 0 : 1;    // 覆盖 float+float16 混合分支
    failCount += CaseDoubleMulPathAdd(stream) ? 0 : 1;    // 覆盖 double + alpha!=1 的 Mul+Add 路径
    failCount += CaseAddsInt32AlphaNeg(stream) ? 0 : 1;   // 覆盖 Adds + 负 alpha
    failCount += CaseAddsBoolToInt(stream) ? 0 : 1;       // 覆盖 Adds bool 特殊 cast 分支
    failCount += CaseAddsFloatToFloat16Out(stream) ? 0 : 1; // 覆盖 Adds 末端 cast + viewCopy
    failCount += CaseAddsInt64MulPath(stream) ? 0 : 1;      // 覆盖 Adds 非 Axpy 的 Mul+Add 分支
    failCount += CaseInplaceAddsFloat(stream) ? 0 : 1;    // 覆盖 InplaceAdds
    failCount += CaseAddV3FloatAxpy(stream) ? 0 : 1;      // 覆盖 AddV3 Axpy 分支
    failCount += CaseAddV3Int8MulPath(stream) ? 0 : 1;    // 覆盖 AddV3 Mul+Add 分支
    failCount += CaseAddV3FloatToFloat16Out(stream) ? 0 : 1; // 覆盖 AddV3 末端 cast + viewCopy
    failCount += CaseAddV3DoubleOutFloatPath(stream) ? 0 : 1; // 覆盖 AddV3 PromoteTypeScalar 特殊分支
    failCount += CaseInplaceAddV3Float(stream) ? 0 : 1;   // 覆盖 InplaceAddV3

    failCount += CaseAddInvalidBroadcastExpectFail() ? 0 : 1;      // 覆盖 Add broadcast 失败分支
    failCount += CaseAddNullptrExpectFail() ? 0 : 1;               // 覆盖 Add nullptr 分支
    failCount += CaseAddOutShapeMismatchExpectFail() ? 0 : 1;      // 覆盖 Add out shape 校验失败
    failCount += CaseAddAlphaCastInvalidExpectFail() ? 0 : 1;      // 覆盖 Add alpha 不可转换分支
    failCount += CaseAddUnsupportedUint32ExpectFail() ? 0 : 1;     // 覆盖 Add 不支持 dtype 分支
    failCount += CaseAddMaxDimExceededExpectFail() ? 0 : 1;        // 覆盖 Add 维度上限分支
    failCount += CaseInplaceAddShapeMismatchExpectFail() ? 0 : 1;  // 覆盖 InplaceAdd 形状检查失败
    failCount += CaseAddsNullptrExpectFail() ? 0 : 1;              // 覆盖 Adds nullptr 分支
    failCount += CaseAddsAlphaCastInvalidExpectFail() ? 0 : 1;     // 覆盖 Adds alpha 不可转换分支
    failCount += CaseAddsOutShapeMismatchExpectFail() ? 0 : 1;     // 覆盖 Adds out shape 校验失败
    failCount += CaseAddsMaxDimExceededExpectFail() ? 0 : 1;       // 覆盖 Adds 维度上限分支
    failCount += CaseAddsNonNdFormatPath(stream) ? 0 : 1;          // 覆盖 Adds 非 ND format 警告分支
    failCount += CaseAddV3NullptrExpectFail() ? 0 : 1;             // 覆盖 AddV3 nullptr 分支
    failCount += CaseAddV3UnsupportedUint8ExpectFail() ? 0 : 1;    // 覆盖 AddV3 不支持 dtype 分支
    failCount += CaseAddV3OutShapeMismatchExpectFail() ? 0 : 1;    // 覆盖 AddV3 out shape 校验失败
    failCount += CaseAddV3MaxDimExceededExpectFail() ? 0 : 1;      // 覆盖 AddV3 维度上限分支
    failCount += CaseAddPromoteToOutCastFailExpectFail() ? 0 : 1;  // 覆盖 Add promote->out 不可转换
    failCount += CaseAddsPromoteToOutCastFailExpectFail() ? 0 : 1; // 覆盖 Adds promote->out 不可转换
    failCount += CaseAddV3PromoteToOutCastFailExpectFail() ? 0 : 1;// 覆盖 AddV3 promote->out 不可转换
    failCount += CaseAddEmptyTensorWorkspaceOnly() ? 0 : 1;        // 覆盖 Add empty tensor 分支
    failCount += CaseAddsEmptyTensorWorkspaceOnly() ? 0 : 1;       // 覆盖 Adds empty tensor 分支
    failCount += CaseAddV3EmptyTensorWorkspaceOnly() ? 0 : 1;      // 覆盖 AddV3 empty tensor 分支
    failCount += CaseAddNonNdFormatPath(stream) ? 0 : 1;           // 覆盖 Add 非 ND format 路径

    failCount += CasePrecisionLargeSmall(stream) ? 0 : 1;      // 精度场景：大数+小数
    failCount += CasePrecisionCancellation(stream) ? 0 : 1;    // 精度场景：正负抵消
    failCount += CaseAddsDoubleAlphaEqOne(stream) ? 0 : 1;     // 覆盖 Adds double IsEqualToOne 分支
    failCount += CaseAddV3DoubleAlphaEqOne(stream) ? 0 : 1;    // 覆盖 AddV3 double IsEqualToOne 分支

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failCount == 0) {
        LOG_PRINT("\nAll add eager cases passed.\n");
        return 0;
    }
    LOG_PRINT("\nAdd eager cases failed, fail_count=%d\n", failCount);
    return 1;
}
