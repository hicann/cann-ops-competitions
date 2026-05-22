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
#include <string>
#include <vector>
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

namespace {

constexpr double FLOAT_EPS = 1e-3;
struct Complex64 {
    float real;
    float imag;
};

struct TensorHandle {
    aclTensor* tensor = nullptr;
    void* deviceAddr = nullptr;
};

struct ScalarHandle {
    aclScalar* scalar = nullptr;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    /** 固定写法，初始化 ACL 运行时资源。 */
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorWithStorage(
    const std::vector<T>& hostData, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
    const std::vector<int64_t>& strides, int64_t offset, aclDataType dataType, TensorHandle* handle)
{
    auto storageElements = GetShapeSize(storageShape);
    auto bytes = static_cast<size_t>(storageElements) * sizeof(T);
    if (bytes > 0) {
        auto ret = aclrtMalloc(&handle->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(handle->deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
    }

    /** 使用 storageShape/strides 构造 view，可覆盖非连续 Tensor 分支。 */
    handle->tensor = aclCreateTensor(
        viewShape.data(), viewShape.size(), dataType, strides.data(), offset, aclFormat::ACL_FORMAT_ND,
        storageShape.data(), storageShape.size(), handle->deviceAddr);
    CHECK_RET(handle->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
    TensorHandle* handle)
{
    return CreateAclTensorWithStorage(hostData, shape, shape, MakeContiguousStrides(shape), 0, dataType, handle);
}

template <typename T>
ScalarHandle CreateScalar(T value, aclDataType dataType)
{
    ScalarHandle handle;
    handle.scalar = aclCreateScalar(&value, dataType);
    return handle;
}

void DestroyTensor(TensorHandle* handle)
{
    if (handle->tensor != nullptr) {
        aclDestroyTensor(handle->tensor);
        handle->tensor = nullptr;
    }
    if (handle->deviceAddr != nullptr) {
        aclrtFree(handle->deviceAddr);
        handle->deviceAddr = nullptr;
    }
}

void DestroyScalar(ScalarHandle* handle)
{
    if (handle->scalar != nullptr) {
        aclDestroyScalar(handle->scalar);
        handle->scalar = nullptr;
    }
}

template <typename T>
int CopyDeviceToHost(const TensorHandle& handle, const std::vector<int64_t>& shape, std::vector<T>* result)
{
    auto elements = GetShapeSize(shape);
    result->assign(static_cast<size_t>(elements), T{});
    auto bytes = static_cast<size_t>(elements) * sizeof(T);
    if (bytes == 0) {
        return ACL_SUCCESS;
    }
    auto ret = aclrtMemcpy(result->data(), bytes, handle.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

double ToDouble(float value)
{
    return static_cast<double>(value);
}

double ToDouble(double value)
{
    return value;
}

template <typename T>
double ToDouble(T value)
{
    return static_cast<double>(value);
}

bool NearlyEqual(float lhs, float rhs, double eps)
{
    return std::abs(ToDouble(lhs) - ToDouble(rhs)) <= eps;
}

bool NearlyEqual(double lhs, double rhs, double eps)
{
    return std::abs(lhs - rhs) <= eps;
}

bool NearlyEqual(const Complex64& lhs, const Complex64& rhs, double eps)
{
    return std::abs(lhs.real - rhs.real) <= eps && std::abs(lhs.imag - rhs.imag) <= eps;
}

template <typename T>
bool NearlyEqual(T lhs, T rhs, double eps)
{
    (void)eps;
    return lhs == rhs;
}

template <typename T>
bool CheckVectorEqual(const std::string& caseName, const std::vector<T>& actual, const std::vector<T>& expected,
    double eps)
{
    if (actual.size() != expected.size()) {
        LOG_PRINT("[%s] size mismatch, actual=%zu expected=%zu\n", caseName.c_str(), actual.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!NearlyEqual(actual[i], expected[i], eps)) {
            LOG_PRINT("[%s] result mismatch at %zu\n", caseName.c_str(), i);
            return false;
        }
    }
    return true;
}

int Execute(aclnnStatus (*runApi)(void*, uint64_t, aclOpExecutor*, aclrtStream), uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
{
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    auto ret = runApi(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnn run failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddCase(const std::string& caseName, const std::vector<SelfT>& selfData,
    const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<OtherT>& otherData,
    const std::vector<int64_t>& otherShape, aclDataType otherDtype, AlphaT alphaValue, aclDataType alphaDtype,
    const std::vector<OutT>& expected, const std::vector<int64_t>& outShape, aclDataType outDtype,
    aclrtStream stream, double eps)
{
    LOG_PRINT("[RUN] %s\n", caseName.c_str());
    TensorHandle self;
    TensorHandle other;
    TensorHandle out;
    ScalarHandle alpha = CreateScalar(alphaValue, alphaDtype);
    std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(outShape)), OutT{});

    bool ok = alpha.scalar != nullptr;
    ok = ok && CreateAclTensor(selfData, selfShape, selfDtype, &self) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(otherData, otherShape, otherDtype, &other) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(outInit, outShape, outDtype, &out) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ok ? aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor)
                  : -1;
    ok = ok && ret == ACL_SUCCESS;
    ok = ok && Execute(aclnnAdd, workspaceSize, executor, stream) == ACL_SUCCESS;
    std::vector<OutT> actual;
    ok = ok && CopyDeviceToHost(out, outShape, &actual) == ACL_SUCCESS;
    ok = ok && CheckVectorEqual(caseName, actual, expected, eps);

    DestroyTensor(&self);
    DestroyTensor(&other);
    DestroyTensor(&out);
    DestroyScalar(&alpha);
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", caseName.c_str());
    return ok;
}

template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddNonContiguousCase(const std::string& caseName, const std::vector<SelfT>& selfStorageData,
    const std::vector<int64_t>& selfViewShape, const std::vector<int64_t>& selfStorageShape,
    const std::vector<int64_t>& selfStrides, int64_t selfOffset, aclDataType selfDtype,
    const std::vector<OtherT>& otherData, const std::vector<int64_t>& otherShape, aclDataType otherDtype,
    AlphaT alphaValue, aclDataType alphaDtype, const std::vector<OutT>& expected,
    const std::vector<int64_t>& outShape, aclDataType outDtype, aclrtStream stream, double eps)
{
    LOG_PRINT("[RUN] %s\n", caseName.c_str());
    TensorHandle self;
    TensorHandle other;
    TensorHandle out;
    ScalarHandle alpha = CreateScalar(alphaValue, alphaDtype);
    std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(outShape)), OutT{});

    bool ok = alpha.scalar != nullptr;
    ok = ok && CreateAclTensorWithStorage(selfStorageData, selfViewShape, selfStorageShape, selfStrides, selfOffset,
        selfDtype, &self) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(otherData, otherShape, otherDtype, &other) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(outInit, outShape, outDtype, &out) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ok ? aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor)
                  : -1;
    ok = ok && ret == ACL_SUCCESS;
    ok = ok && Execute(aclnnAdd, workspaceSize, executor, stream) == ACL_SUCCESS;
    std::vector<OutT> actual;
    ok = ok && CopyDeviceToHost(out, outShape, &actual) == ACL_SUCCESS;
    ok = ok && CheckVectorEqual(caseName, actual, expected, eps);

    DestroyTensor(&self);
    DestroyTensor(&other);
    DestroyTensor(&out);
    DestroyScalar(&alpha);
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", caseName.c_str());
    return ok;
}

template <typename SelfT, typename OtherScalarT, typename AlphaT, typename OutT>
bool RunAddsCase(const std::string& caseName, const std::vector<SelfT>& selfData,
    const std::vector<int64_t>& selfShape, aclDataType selfDtype, OtherScalarT otherValue, aclDataType otherDtype,
    AlphaT alphaValue, aclDataType alphaDtype, const std::vector<OutT>& expected,
    const std::vector<int64_t>& outShape, aclDataType outDtype, aclrtStream stream, double eps)
{
    LOG_PRINT("[RUN] %s\n", caseName.c_str());
    TensorHandle self;
    TensorHandle out;
    ScalarHandle other = CreateScalar(otherValue, otherDtype);
    ScalarHandle alpha = CreateScalar(alphaValue, alphaDtype);
    std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(outShape)), OutT{});

    bool ok = other.scalar != nullptr && alpha.scalar != nullptr;
    ok = ok && CreateAclTensor(selfData, selfShape, selfDtype, &self) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(outInit, outShape, outDtype, &out) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ok ? aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor)
                  : -1;
    ok = ok && ret == ACL_SUCCESS;
    ok = ok && Execute(aclnnAdds, workspaceSize, executor, stream) == ACL_SUCCESS;
    std::vector<OutT> actual;
    ok = ok && CopyDeviceToHost(out, outShape, &actual) == ACL_SUCCESS;
    ok = ok && CheckVectorEqual(caseName, actual, expected, eps);

    DestroyTensor(&self);
    DestroyTensor(&out);
    DestroyScalar(&other);
    DestroyScalar(&alpha);
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", caseName.c_str());
    return ok;
}

template <typename SelfT, typename OtherT, typename AlphaT>
bool RunInplaceAddCase(const std::string& caseName, const std::vector<SelfT>& selfData,
    const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<OtherT>& otherData,
    const std::vector<int64_t>& otherShape, aclDataType otherDtype, AlphaT alphaValue, aclDataType alphaDtype,
    const std::vector<SelfT>& expected, aclrtStream stream, double eps)
{
    LOG_PRINT("[RUN] %s\n", caseName.c_str());
    TensorHandle self;
    TensorHandle other;
    ScalarHandle alpha = CreateScalar(alphaValue, alphaDtype);

    bool ok = alpha.scalar != nullptr;
    ok = ok && CreateAclTensor(selfData, selfShape, selfDtype, &self) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(otherData, otherShape, otherDtype, &other) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ok ? aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor)
                  : -1;
    ok = ok && ret == ACL_SUCCESS;
    ok = ok && Execute(aclnnInplaceAdd, workspaceSize, executor, stream) == ACL_SUCCESS;
    std::vector<SelfT> actual;
    ok = ok && CopyDeviceToHost(self, selfShape, &actual) == ACL_SUCCESS;
    ok = ok && CheckVectorEqual(caseName, actual, expected, eps);

    DestroyTensor(&self);
    DestroyTensor(&other);
    DestroyScalar(&alpha);
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", caseName.c_str());
    return ok;
}

template <typename SelfT, typename OtherScalarT, typename AlphaT>
bool RunInplaceAddsCase(const std::string& caseName, const std::vector<SelfT>& selfData,
    const std::vector<int64_t>& selfShape, aclDataType selfDtype, OtherScalarT otherValue, aclDataType otherDtype,
    AlphaT alphaValue, aclDataType alphaDtype, const std::vector<SelfT>& expected, aclrtStream stream, double eps)
{
    LOG_PRINT("[RUN] %s\n", caseName.c_str());
    TensorHandle self;
    ScalarHandle other = CreateScalar(otherValue, otherDtype);
    ScalarHandle alpha = CreateScalar(alphaValue, alphaDtype);

    bool ok = other.scalar != nullptr && alpha.scalar != nullptr;
    ok = ok && CreateAclTensor(selfData, selfShape, selfDtype, &self) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ok ? aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor)
                  : -1;
    ok = ok && ret == ACL_SUCCESS;
    ok = ok && Execute(aclnnInplaceAdds, workspaceSize, executor, stream) == ACL_SUCCESS;
    std::vector<SelfT> actual;
    ok = ok && CopyDeviceToHost(self, selfShape, &actual) == ACL_SUCCESS;
    ok = ok && CheckVectorEqual(caseName, actual, expected, eps);

    DestroyTensor(&self);
    DestroyScalar(&other);
    DestroyScalar(&alpha);
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", caseName.c_str());
    return ok;
}

template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddV3Case(const std::string& caseName, SelfScalarT selfValue, aclDataType selfDtype,
    const std::vector<OtherT>& otherData, const std::vector<int64_t>& otherShape, aclDataType otherDtype,
    AlphaT alphaValue, aclDataType alphaDtype, const std::vector<OutT>& expected,
    const std::vector<int64_t>& outShape, aclDataType outDtype, aclrtStream stream, double eps)
{
    LOG_PRINT("[RUN] %s\n", caseName.c_str());
    ScalarHandle self = CreateScalar(selfValue, selfDtype);
    TensorHandle other;
    ScalarHandle alpha = CreateScalar(alphaValue, alphaDtype);
    TensorHandle out;
    std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(outShape)), OutT{});

    bool ok = self.scalar != nullptr && alpha.scalar != nullptr;
    ok = ok && CreateAclTensor(otherData, otherShape, otherDtype, &other) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(outInit, outShape, outDtype, &out) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ok ? aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor)
                  : -1;
    ok = ok && ret == ACL_SUCCESS;
    ok = ok && Execute(aclnnAddV3, workspaceSize, executor, stream) == ACL_SUCCESS;
    std::vector<OutT> actual;
    ok = ok && CopyDeviceToHost(out, outShape, &actual) == ACL_SUCCESS;
    ok = ok && CheckVectorEqual(caseName, actual, expected, eps);

    DestroyScalar(&self);
    DestroyTensor(&other);
    DestroyScalar(&alpha);
    DestroyTensor(&out);
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", caseName.c_str());
    return ok;
}

template <typename SelfScalarT, typename OtherT, typename AlphaT>
bool RunInplaceAddV3Case(const std::string& caseName, SelfScalarT selfValue, aclDataType selfDtype,
    const std::vector<OtherT>& otherData, const std::vector<int64_t>& otherShape, aclDataType otherDtype,
    AlphaT alphaValue, aclDataType alphaDtype, const std::vector<OtherT>& expected, aclrtStream stream, double eps)
{
    LOG_PRINT("[RUN] %s\n", caseName.c_str());
    ScalarHandle self = CreateScalar(selfValue, selfDtype);
    TensorHandle other;
    ScalarHandle alpha = CreateScalar(alphaValue, alphaDtype);

    bool ok = self.scalar != nullptr && alpha.scalar != nullptr;
    ok = ok && CreateAclTensor(otherData, otherShape, otherDtype, &other) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = ok ? aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor)
                  : -1;
    ok = ok && ret == ACL_SUCCESS;
    ok = ok && Execute(aclnnInplaceAddV3, workspaceSize, executor, stream) == ACL_SUCCESS;
    std::vector<OtherT> actual;
    ok = ok && CopyDeviceToHost(other, otherShape, &actual) == ACL_SUCCESS;
    ok = ok && CheckVectorEqual(caseName, actual, expected, eps);

    DestroyScalar(&self);
    DestroyTensor(&other);
    DestroyScalar(&alpha);
    LOG_PRINT("[%s] %s\n", ok ? "PASS" : "FAIL", caseName.c_str());
    return ok;
}

bool RunInvalidParamCases(aclrtStream stream)
{
    (void)stream;
    LOG_PRINT("[RUN] invalid parameter cases\n");

    TensorHandle floatTensor;
    TensorHandle floatOut;
    TensorHandle badShapeOut;
    TensorHandle rank9Tensor;
    TensorHandle uint32Tensor;
    std::vector<float> sixFloats = {1, 2, 3, 4, 5, 6};
    std::vector<float> rank9Data = {1};
    std::vector<uint32_t> uint32Data = {1, 2, 3, 4};
    auto alpha = CreateScalar(1.0f, ACL_FLOAT);

    bool ok = alpha.scalar != nullptr;
    ok = ok && CreateAclTensor(sixFloats, {2, 3}, ACL_FLOAT, &floatTensor) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(sixFloats, {2, 3}, ACL_FLOAT, &floatOut) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(sixFloats, {3, 2}, ACL_FLOAT, &badShapeOut) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(rank9Data, {1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, &rank9Tensor) == ACL_SUCCESS;
    ok = ok && CreateAclTensor(uint32Data, {2, 2}, ACL_UINT32, &uint32Tensor) == ACL_SUCCESS;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(nullptr, floatTensor.tensor, alpha.scalar, floatOut.tensor, &workspaceSize,
        &executor);
    ok = ok && ret != ACL_SUCCESS;
    ret = aclnnAddGetWorkspaceSize(floatTensor.tensor, floatTensor.tensor, nullptr, floatOut.tensor, &workspaceSize,
        &executor);
    ok = ok && ret != ACL_SUCCESS;
    ret = aclnnAddGetWorkspaceSize(floatTensor.tensor, floatTensor.tensor, alpha.scalar, badShapeOut.tensor,
        &workspaceSize, &executor);
    ok = ok && ret != ACL_SUCCESS;
    ret = aclnnAddGetWorkspaceSize(rank9Tensor.tensor, rank9Tensor.tensor, alpha.scalar, rank9Tensor.tensor,
        &workspaceSize, &executor);
    ok = ok && ret != ACL_SUCCESS;
    ret = aclnnAddGetWorkspaceSize(uint32Tensor.tensor, uint32Tensor.tensor, alpha.scalar, uint32Tensor.tensor,
        &workspaceSize, &executor);
    ok = ok && ret != ACL_SUCCESS;

    auto otherScalar = CreateScalar(2.0f, ACL_FLOAT);
    ret = aclnnAddsGetWorkspaceSize(floatTensor.tensor, otherScalar.scalar, nullptr, floatOut.tensor, &workspaceSize,
        &executor);
    ok = ok && ret != ACL_SUCCESS;
    ret = aclnnAddV3GetWorkspaceSize(nullptr, floatTensor.tensor, alpha.scalar, floatOut.tensor, &workspaceSize,
        &executor);
    ok = ok && ret != ACL_SUCCESS;

    DestroyTensor(&floatTensor);
    DestroyTensor(&floatOut);
    DestroyTensor(&badShapeOut);
    DestroyTensor(&rank9Tensor);
    DestroyTensor(&uint32Tensor);
    DestroyScalar(&alpha);
    DestroyScalar(&otherScalar);
    LOG_PRINT("[%s] invalid parameter cases\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool RunAllCases(aclrtStream stream)
{
    bool ok = true;

    ok = RunAddCase("add_fp32_axpy_alpha_not_one", std::vector<float>{0, -1, 2, 4}, {4}, ACL_FLOAT,
             std::vector<float>{1, 2, -3, 0.5f}, {4}, ACL_FLOAT, 2.0f, ACL_FLOAT,
             std::vector<float>{2, 3, -4, 5}, {4}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddCase("add_int32_alpha_one_direct", std::vector<int32_t>{1, -2, 3, -4}, {2, 2}, ACL_INT32,
             std::vector<int32_t>{5, 6, -7, -8}, {2, 2}, ACL_INT32, int32_t{1}, ACL_INT32,
             std::vector<int32_t>{6, 4, -4, -12}, {2, 2}, ACL_INT32, stream, 0) &&
         ok;

    ok = RunAddCase("add_int64_axpy_v2_or_mul_path", std::vector<int64_t>{1, -2, 3}, {3}, ACL_INT64,
             std::vector<int64_t>{4, 5, -6}, {3}, ACL_INT64, int64_t{3}, ACL_INT64,
             std::vector<int64_t>{13, 13, -15}, {3}, ACL_INT64, stream, 0) &&
         ok;

    ok = RunAddCase("add_int16_aicpu_direct_path", std::vector<int16_t>{1, -2, 3}, {3}, ACL_INT16,
             std::vector<int16_t>{4, 5, -6}, {3}, ACL_INT16, int16_t{1}, ACL_INT16,
             std::vector<int16_t>{5, 3, -3}, {3}, ACL_INT16, stream, 0) &&
         ok;

    ok = RunAddCase("add_uint8_without_cast", std::vector<uint8_t>{1, 2, 3, 4}, {4}, ACL_UINT8,
             std::vector<uint8_t>{10, 20, 30, 40}, {4}, ACL_UINT8, int32_t{1}, ACL_INT32,
             std::vector<uint8_t>{11, 22, 33, 44}, {4}, ACL_UINT8, stream, 0) &&
         ok;

    ok = RunAddCase("add_int8_without_cast", std::vector<int8_t>{1, -2, 3, -4}, {4}, ACL_INT8,
             std::vector<int8_t>{2, 3, -4, -5}, {4}, ACL_INT8, int32_t{1}, ACL_INT32,
             std::vector<int8_t>{3, 1, -1, -9}, {4}, ACL_INT8, stream, 0) &&
         ok;

    ok = RunAddCase("add_bool_tensor", std::vector<uint8_t>{1, 0, 1, 0}, {4}, ACL_BOOL,
             std::vector<uint8_t>{0, 0, 1, 1}, {4}, ACL_BOOL, int32_t{1}, ACL_INT32,
             std::vector<uint8_t>{1, 0, 1, 1}, {4}, ACL_BOOL, stream, 0) &&
         ok;

    ok = RunAddCase("add_complex64_tensor", std::vector<Complex64>{{1, 2}, {-3, 4}}, {2}, ACL_COMPLEX64,
             std::vector<Complex64>{{5, -1}, {2, 2}}, {2}, ACL_COMPLEX64, 1.0f, ACL_FLOAT,
             std::vector<Complex64>{{6, 1}, {-1, 6}}, {2}, ACL_COMPLEX64, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddCase("add_broadcast_float", std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3, 1}, ACL_FLOAT,
             std::vector<float>{10, 20, 30}, {1, 3, 1}, ACL_FLOAT, 1.0f, ACL_FLOAT,
             std::vector<float>{11, 22, 33, 14, 25, 36}, {2, 3, 1}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddNonContiguousCase("add_non_contiguous_input", std::vector<float>{99, 1, 2, 3, 99, 4, 5, 6},
             {2, 3}, {2, 4}, {4, 1}, 1, ACL_FLOAT, std::vector<float>{10, 20, 30}, {1, 3}, ACL_FLOAT,
             1.0f, ACL_FLOAT, std::vector<float>{11, 22, 33, 14, 25, 36}, {2, 3}, ACL_FLOAT, stream,
             FLOAT_EPS) &&
         ok;

    ok = RunAddCase("add_empty_tensor", std::vector<int32_t>{}, {2, 0, 3}, ACL_INT32,
             std::vector<int32_t>{}, {1, 0, 3}, ACL_INT32, int32_t{5}, ACL_INT32, std::vector<int32_t>{},
             {2, 0, 3}, ACL_INT32, stream, 0) &&
         ok;

    ok = RunAddCase("add_fp16_fp32_mixed_alpha_one", std::vector<uint16_t>{0x3c00, 0xc000, 0x3800}, {3},
             ACL_FLOAT16, std::vector<float>{1.0f, 2.0f, 3.5f}, {3}, ACL_FLOAT, 1.0f, ACL_FLOAT,
             std::vector<float>{2.0f, 0.0f, 4.0f}, {3}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddCase("add_fp32_bf16_mixed_alpha_one", std::vector<float>{1.0f, 2.0f, -3.0f}, {3}, ACL_FLOAT,
             std::vector<uint16_t>{0x3f80, 0xc000, 0x4000}, {3}, ACL_BF16, 1.0f, ACL_FLOAT,
             std::vector<float>{2.0f, 0.0f, -1.0f}, {3}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddsCase("adds_float_scalar_axpy", std::vector<float>{1, 2, -3}, {3}, ACL_FLOAT, 2.0f, ACL_FLOAT,
             1.5f, ACL_FLOAT, std::vector<float>{4, 5, 0}, {3}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddsCase("adds_bool_scalar_special_cast", std::vector<uint8_t>{0, 1, 0, 1}, {4}, ACL_BOOL, true,
             ACL_BOOL, true, ACL_BOOL, std::vector<uint8_t>{1, 1, 1, 1}, {4}, ACL_UINT8, stream, 0) &&
         ok;

    ok = RunAddsCase("adds_empty_tensor", std::vector<int32_t>{}, {1, 0, 4}, ACL_INT32, int32_t{2}, ACL_INT32,
             int32_t{3}, ACL_INT32, std::vector<int32_t>{}, {1, 0, 4}, ACL_INT32, stream, 0) &&
         ok;

    ok = RunInplaceAddCase("inplace_add_fp32", std::vector<float>{1, 2, 3}, {3}, ACL_FLOAT,
             std::vector<float>{4, 5, 6}, {3}, ACL_FLOAT, 0.5f, ACL_FLOAT, std::vector<float>{3, 4.5f, 6},
             stream, FLOAT_EPS) &&
         ok;

    ok = RunInplaceAddsCase("inplace_adds_int32", std::vector<int32_t>{1, 2, 3}, {3}, ACL_INT32, int32_t{4},
             ACL_INT32, int32_t{2}, ACL_INT32, std::vector<int32_t>{9, 10, 11}, stream, 0) &&
         ok;

    ok = RunAddV3Case("add_v3_float_alpha_one", 2.0f, ACL_FLOAT, std::vector<float>{1, -2, 3}, {3},
             ACL_FLOAT, 1.0f, ACL_FLOAT, std::vector<float>{3, 0, 5}, {3}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddV3Case("add_v3_float_axpy", 1.0f, ACL_FLOAT, std::vector<float>{2, 4, -1}, {3}, ACL_FLOAT,
             2.5f, ACL_FLOAT, std::vector<float>{6, 11, -1.5f}, {3}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunAddV3Case("add_v3_int8_mul_path", int8_t{2}, ACL_INT8, std::vector<int8_t>{1, -2, 3}, {3},
             ACL_INT8, int8_t{3}, ACL_INT8, std::vector<int8_t>{5, -4, 11}, {3}, ACL_INT8, stream, 0) &&
         ok;

    ok = RunAddV3Case("add_v3_empty_tensor", 1.0f, ACL_FLOAT, std::vector<float>{}, {0, 3}, ACL_FLOAT,
             1.0f, ACL_FLOAT, std::vector<float>{}, {0, 3}, ACL_FLOAT, stream, FLOAT_EPS) &&
         ok;

    ok = RunInplaceAddV3Case("inplace_add_v3_float", 3.0f, ACL_FLOAT, std::vector<float>{1, 2, 3}, {3},
             ACL_FLOAT, 2.0f, ACL_FLOAT, std::vector<float>{5, 7, 9}, stream, FLOAT_EPS) &&
         ok;

    ok = RunInvalidParamCases(stream) && ok;
    return ok;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    auto success = RunAllCases(stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\nAdd end-to-end coverage result: %s\n", success ? "PASS" : "FAIL");
    return success ? 0 : 1;
}