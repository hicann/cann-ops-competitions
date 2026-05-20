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
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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

constexpr int kAclnnErrParamNullptr = 161001;
constexpr int kAclnnErrParamInvalid = 161002;

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
    ~TensorResource()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

struct ScalarResource {
    aclScalar* scalar = nullptr;
    ~ScalarResource()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
        }
    }
};

struct WorkspaceResource {
    void* addr = nullptr;
    ~WorkspaceResource()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static inline float Float16ToFloat(uint16_t value)
{
    unsigned int sign = (value >> 15) & 0x1;
    unsigned int exponent = (value >> 10) & 0x1f;
    unsigned int mantissa = value & 0x3ff;
    float result = 0.0f;
    if (exponent == 0) {
        result = mantissa * 0.0000019073486328125f;
    } else if (exponent == 31) {
        result = mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        result = (1.0f + mantissa * 0.0009765625f) * std::pow(2.0f, static_cast<int>(exponent) - 15);
    }
    return sign ? -result : result;
}

static inline uint16_t FloatToFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return sign | 0x7c00;
    }
    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

template <typename T>
double ToDoubleValue(const T& value)
{
    return static_cast<double>(value);
}

template <>
double ToDoubleValue<uint16_t>(const uint16_t& value)
{
    return static_cast<double>(Float16ToFloat(value));
}

template <typename T>
std::vector<double> ToDoubleVector(const std::vector<T>& input)
{
    std::vector<double> output;
    output.reserve(input.size());
    for (const auto& item : input) {
        output.push_back(ToDoubleValue(item));
    }
    return output;
}

bool AlmostEqual(double expected, double actual, double atol, double rtol)
{
    if (std::isnan(expected) && std::isnan(actual)) {
        return true;
    }
    if (std::isinf(expected) || std::isinf(actual)) {
        return std::isinf(expected) && std::isinf(actual) && ((expected > 0) == (actual > 0));
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType, TensorResource* resource)
{
    size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(&resource->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(resource->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    auto strides = GetStrides(shape);
    resource->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), resource->deviceAddr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CopyDeviceToHost(const TensorResource& resource, std::vector<T>* hostData)
{
    size_t size = hostData->size() * sizeof(T);
    auto ret = aclrtMemcpy(hostData->data(), size, resource.deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateScalar(T value, aclDataType dataType, ScalarResource* resource)
{
    resource->scalar = aclCreateScalar(&value, dataType);
    CHECK_RET(resource->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<double> ComputeBroadcastPow(const std::vector<double>& baseData,
                                        const std::vector<int64_t>& baseShape,
                                        const std::vector<double>& exponentData,
                                        const std::vector<int64_t>& exponentShape,
                                        const std::vector<int64_t>& outShape)
{
    const auto baseStrides = GetStrides(baseShape);
    const auto exponentStrides = GetStrides(exponentShape);
    const auto outStrides = GetStrides(outShape);
    const size_t outRank = outShape.size();
    const size_t baseOffset = outRank - baseShape.size();
    const size_t exponentOffset = outRank - exponentShape.size();
    const int64_t outSize = GetShapeSize(outShape);
    std::vector<double> expected(outSize, 0.0);

    for (int64_t flatIndex = 0; flatIndex < outSize; ++flatIndex) {
        int64_t remain = flatIndex;
        std::vector<int64_t> outIndex(outRank, 0);
        for (size_t dim = 0; dim < outRank; ++dim) {
            outIndex[dim] = remain / outStrides[dim];
            remain %= outStrides[dim];
        }

        int64_t baseFlatIndex = 0;
        for (size_t dim = 0; dim < baseShape.size(); ++dim) {
            int64_t idx = baseShape[dim] == 1 ? 0 : outIndex[dim + baseOffset];
            baseFlatIndex += idx * baseStrides[dim];
        }

        int64_t exponentFlatIndex = 0;
        for (size_t dim = 0; dim < exponentShape.size(); ++dim) {
            int64_t idx = exponentShape[dim] == 1 ? 0 : outIndex[dim + exponentOffset];
            exponentFlatIndex += idx * exponentStrides[dim];
        }

        expected[flatIndex] = std::pow(baseData[baseFlatIndex], exponentData[exponentFlatIndex]);
    }
    return expected;
}

template <typename T>
int CheckNumericResult(
    const std::string& caseName, const std::vector<T>& actual, const std::vector<double>& expected, double atol, double rtol)
{
    if (actual.size() != expected.size()) {
        LOG_PRINT("[FAIL] %s size mismatch: actual=%zu expected=%zu\n", caseName.c_str(), actual.size(), expected.size());
        return 1;
    }
    int mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double actualValue = ToDoubleValue(actual[i]);
        if (!AlmostEqual(expected[i], actualValue, atol, rtol)) {
            LOG_PRINT("  mismatch[%zu]: expected=%.8f actual=%.8f\n", i, expected[i], actualValue);
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s mismatches=%d\n", caseName.c_str(), mismatches);
    return 1;
}

template <typename TSelf, typename TExponent, typename TOut>
int RunPowTensorScalarCase(const std::string& caseName,
                           const std::vector<TSelf>& selfData,
                           const std::vector<int64_t>& shape,
                           aclDataType selfType,
                           TExponent exponentValue,
                           aclDataType exponentType,
                           aclDataType outType,
                           aclrtStream stream,
                           double atol,
                           double rtol)
{
    TensorResource self;
    TensorResource out;
    ScalarResource exponent;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<TOut> outData(GetShapeSize(shape), static_cast<TOut>(0));
    std::vector<double> expected = ToDoubleVector(selfData);
    double exponentDouble = ToDoubleValue(exponentValue);
    for (auto& item : expected) {
        item = std::pow(item, exponentDouble);
    }

    int ret = CreateAclTensor(selfData, shape, selfType, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(outData, shape, outType, &out);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateScalar(exponentValue, exponentType, &exponent);
    if (ret != ACL_SUCCESS) return 1;

    ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return 1;
    }
    ret = aclnnPowTensorScalar(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnPowTensorScalar failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return 1;
    ret = CopyDeviceToHost(out, &outData);
    if (ret != ACL_SUCCESS) return 1;
    return CheckNumericResult(caseName, outData, expected, atol, rtol);
}

template <typename TSelf, typename TExponent>
int RunInplacePowTensorScalarCase(const std::string& caseName,
                                  const std::vector<TSelf>& selfData,
                                  const std::vector<int64_t>& shape,
                                  aclDataType selfType,
                                  TExponent exponentValue,
                                  aclDataType exponentType,
                                  aclrtStream stream,
                                  double atol,
                                  double rtol)
{
    TensorResource self;
    ScalarResource exponent;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto actual = selfData;
    std::vector<double> expected = ToDoubleVector(selfData);
    double exponentDouble = ToDoubleValue(exponentValue);
    for (auto& item : expected) {
        item = std::pow(item, exponentDouble);
    }

    int ret = CreateAclTensor(selfData, shape, selfType, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateScalar(exponentValue, exponentType, &exponent);
    if (ret != ACL_SUCCESS) return 1;

    ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplacePowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return 1;
    }
    ret = aclnnInplacePowTensorScalar(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplacePowTensorScalar failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return 1;
    ret = CopyDeviceToHost(self, &actual);
    if (ret != ACL_SUCCESS) return 1;
    return CheckNumericResult(caseName, actual, expected, atol, rtol);
}

template <typename TSelfScalar, typename TExponent, typename TOut>
int RunPowScalarTensorCase(const std::string& caseName,
                           TSelfScalar selfValue,
                           aclDataType selfType,
                           const std::vector<TExponent>& exponentData,
                           const std::vector<int64_t>& shape,
                           aclDataType exponentType,
                           aclDataType outType,
                           aclrtStream stream,
                           double atol,
                           double rtol)
{
    TensorResource exponent;
    TensorResource out;
    ScalarResource self;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<TOut> outData(GetShapeSize(shape), static_cast<TOut>(0));
    std::vector<double> expected = ToDoubleVector(exponentData);
    double selfDouble = ToDoubleValue(selfValue);
    for (auto& item : expected) {
        item = std::pow(selfDouble, item);
    }

    int ret = CreateAclTensor(exponentData, shape, exponentType, &exponent);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(outData, shape, outType, &out);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateScalar(selfValue, selfType, &self);
    if (ret != ACL_SUCCESS) return 1;

    ret = aclnnPowScalarTensorGetWorkspaceSize(self.scalar, exponent.tensor, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnPowScalarTensorGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return 1;
    }
    ret = aclnnPowScalarTensor(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnPowScalarTensor failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return 1;
    ret = CopyDeviceToHost(out, &outData);
    if (ret != ACL_SUCCESS) return 1;
    return CheckNumericResult(caseName, outData, expected, atol, rtol);
}

template <typename TSelf, typename TExponent, typename TOut>
int RunPowTensorTensorCase(const std::string& caseName,
                           const std::vector<TSelf>& selfData,
                           const std::vector<int64_t>& selfShape,
                           aclDataType selfType,
                           const std::vector<TExponent>& exponentData,
                           const std::vector<int64_t>& exponentShape,
                           aclDataType exponentType,
                           const std::vector<int64_t>& outShape,
                           aclDataType outType,
                           aclrtStream stream,
                           double atol,
                           double rtol)
{
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<TOut> outData(GetShapeSize(outShape), static_cast<TOut>(0));
    std::vector<double> expected =
        ComputeBroadcastPow(ToDoubleVector(selfData), selfShape, ToDoubleVector(exponentData), exponentShape, outShape);

    int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(exponentData, exponentShape, exponentType, &exponent);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(outData, outShape, outType, &out);
    if (ret != ACL_SUCCESS) return 1;

    ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnPowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return 1;
    }
    ret = aclnnPowTensorTensor(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnPowTensorTensor failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return 1;
    ret = CopyDeviceToHost(out, &outData);
    if (ret != ACL_SUCCESS) return 1;
    return CheckNumericResult(caseName, outData, expected, atol, rtol);
}

template <typename TSelf, typename TExponent>
int RunInplacePowTensorTensorCase(const std::string& caseName,
                                  const std::vector<TSelf>& selfData,
                                  const std::vector<int64_t>& selfShape,
                                  aclDataType selfType,
                                  const std::vector<TExponent>& exponentData,
                                  const std::vector<int64_t>& exponentShape,
                                  aclDataType exponentType,
                                  aclrtStream stream,
                                  double atol,
                                  double rtol)
{
    TensorResource self;
    TensorResource exponent;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto actual = selfData;
    std::vector<double> expected =
        ComputeBroadcastPow(ToDoubleVector(selfData), selfShape, ToDoubleVector(exponentData), exponentShape, selfShape);

    int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(exponentData, exponentShape, exponentType, &exponent);
    if (ret != ACL_SUCCESS) return 1;

    ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplacePowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return 1;
    }
    ret = aclnnInplacePowTensorTensor(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplacePowTensorTensor failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return 1;
    ret = CopyDeviceToHost(self, &actual);
    if (ret != ACL_SUCCESS) return 1;
    return CheckNumericResult(caseName, actual, expected, atol, rtol);
}

template <typename TSelf, typename TOut>
int RunExp2Case(const std::string& caseName,
                const std::vector<TSelf>& selfData,
                const std::vector<int64_t>& shape,
                aclDataType selfType,
                aclDataType outType,
                aclrtStream stream,
                double atol,
                double rtol)
{
    TensorResource self;
    TensorResource out;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<TOut> outData(GetShapeSize(shape), static_cast<TOut>(0));
    std::vector<double> expected = ToDoubleVector(selfData);
    for (auto& item : expected) {
        item = std::pow(2.0, item);
    }

    int ret = CreateAclTensor(selfData, shape, selfType, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(outData, shape, outType, &out);
    if (ret != ACL_SUCCESS) return 1;

    ret = aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnExp2GetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return 1;
    }
    ret = aclnnExp2(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnExp2 failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return 1;
    ret = CopyDeviceToHost(out, &outData);
    if (ret != ACL_SUCCESS) return 1;
    return CheckNumericResult(caseName, outData, expected, atol, rtol);
}

template <typename TSelf>
int RunInplaceExp2Case(const std::string& caseName,
                       const std::vector<TSelf>& selfData,
                       const std::vector<int64_t>& shape,
                       aclDataType selfType,
                       aclrtStream stream,
                       double atol,
                       double rtol)
{
    TensorResource self;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto actual = selfData;
    std::vector<double> expected = ToDoubleVector(selfData);
    for (auto& item : expected) {
        item = std::pow(2.0, item);
    }

    int ret = CreateAclTensor(selfData, shape, selfType, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = aclnnInplaceExp2GetWorkspaceSize(self.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceExp2GetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return 1;
    }
    ret = aclnnInplaceExp2(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceExp2 failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return 1;
    ret = CopyDeviceToHost(self, &actual);
    if (ret != ACL_SUCCESS) return 1;
    return CheckNumericResult(caseName, actual, expected, atol, rtol);
}

int RunPowTensorScalarNullptrCase(const std::string& caseName)
{
    TensorResource out;
    ScalarResource exponent;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateScalar(2.0f, ACL_FLOAT, &exponent);
    if (ret != ACL_SUCCESS) return 1;
    ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, exponent.scalar, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int RunInplacePowTensorScalarNullptrCase(const std::string& caseName)
{
    ScalarResource exponent;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateScalar(2.0f, ACL_FLOAT, &exponent);
    if (ret != ACL_SUCCESS) return 1;
    ret = aclnnInplacePowTensorScalarGetWorkspaceSize(nullptr, exponent.scalar, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int RunPowScalarTensorNullptrCase(const std::string& caseName)
{
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}, ACL_FLOAT, &exponent);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) return 1;
    ret = aclnnPowScalarTensorGetWorkspaceSize(nullptr, exponent.tensor, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int RunPowTensorTensorInvalidShapeCase(const std::string& caseName)
{
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(std::vector<float>{1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, ACL_FLOAT, &exponent);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) return 1;
    ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamInvalid) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamInvalid, ret);
    return 1;
}

int RunInplacePowTensorTensorNullptrCase(const std::string& caseName)
{
    TensorResource exponent;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(std::vector<float>{1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &exponent);
    if (ret != ACL_SUCCESS) return 1;
    ret = aclnnInplacePowTensorTensorGetWorkspaceSize(nullptr, exponent.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int RunExp2InvalidShapeCase(const std::string& caseName)
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(std::vector<float>{-1, 0, 1, 2}, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) return 1;
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {4}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) return 1;
    ret = aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamInvalid) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamInvalid, ret);
    return 1;
}

int RunInplaceExp2NullptrCase(const std::string& caseName)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = aclnnInplaceExp2GetWorkspaceSize(nullptr, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }
    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int failedCases = 0;
    failedCases += RunPowTensorScalarCase<float, float, float>(
        "pow_tensor_scalar_general_f32", {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, ACL_FLOAT, 4.1f, ACL_FLOAT, ACL_FLOAT, stream,
        1e-5, 1e-5);
    failedCases += RunPowTensorScalarCase<uint16_t, float, uint16_t>(
        "pow_tensor_scalar_sqrt_f16",
        {FloatToFloat16(1.0f), FloatToFloat16(4.0f), FloatToFloat16(9.0f), FloatToFloat16(16.0f)},
        {2, 2},
        ACL_FLOAT16,
        0.5f,
        ACL_FLOAT,
        ACL_FLOAT16,
        stream,
        2e-2,
        2e-2);
    failedCases += RunInplacePowTensorScalarCase<int32_t, int32_t>(
        "inplace_pow_tensor_scalar_cube_i32", {1, 2, 3, 4}, {2, 2}, ACL_INT32, 3, ACL_INT32, stream, 0.0, 0.0);
    failedCases += RunPowScalarTensorCase<float, float, float>(
        "pow_scalar_tensor_negative_exp_f32", 2.0f, ACL_FLOAT, {-1.0f, 0.0f, 1.0f, 2.0f}, {2, 2}, ACL_FLOAT, ACL_FLOAT, stream,
        1e-5, 1e-5);
    failedCases += RunPowScalarTensorCase<float, float, float>(
        "pow_scalar_tensor_general_f32", 2.0f, ACL_FLOAT, {0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}, ACL_FLOAT, ACL_FLOAT, stream,
        1e-5, 1e-5);
    failedCases += RunPowTensorTensorCase<float, float, float>(
        "pow_tensor_tensor_broadcast_f32",
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3},
        ACL_FLOAT,
        {1.0f, 2.0f, 3.0f},
        {1, 3},
        ACL_FLOAT,
        {2, 3},
        ACL_FLOAT,
        stream,
        1e-5,
        1e-5);
    failedCases += RunPowTensorTensorCase<int32_t, int32_t, int32_t>(
        "pow_tensor_tensor_i32", {2, 3, 4, 5}, {2, 2}, ACL_INT32, {1, 2, 3, 0}, {2, 2}, ACL_INT32, {2, 2}, ACL_INT32,
        stream, 0.0, 0.0);
    failedCases += RunInplacePowTensorTensorCase<uint16_t, uint16_t>(
        "inplace_pow_tensor_tensor_f16",
        {FloatToFloat16(2.0f), FloatToFloat16(3.0f), FloatToFloat16(4.0f), FloatToFloat16(5.0f)},
        {2, 2},
        ACL_FLOAT16,
        {FloatToFloat16(1.0f), FloatToFloat16(2.0f), FloatToFloat16(3.0f), FloatToFloat16(0.0f)},
        {2, 2},
        ACL_FLOAT16,
        stream,
        2e-2,
        2e-2);
    failedCases += RunExp2Case<float, float>(
        "exp2_f32", {-1.0f, 0.0f, 1.0f, 2.0f}, {2, 2}, ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);
    failedCases += RunInplaceExp2Case<uint16_t>(
        "inplace_exp2_f16",
        {FloatToFloat16(0.0f), FloatToFloat16(1.0f), FloatToFloat16(2.0f), FloatToFloat16(3.0f)},
        {2, 2},
        ACL_FLOAT16,
        stream,
        2e-2,
        2e-2);

    failedCases += RunPowTensorScalarNullptrCase("pow_tensor_scalar_nullptr_check");
    failedCases += RunInplacePowTensorScalarNullptrCase("inplace_pow_tensor_scalar_nullptr_check");
    failedCases += RunPowScalarTensorNullptrCase("pow_scalar_tensor_nullptr_check");
    failedCases += RunPowTensorTensorInvalidShapeCase("pow_tensor_tensor_invalid_shape_check");
    failedCases += RunInplacePowTensorTensorNullptrCase("inplace_pow_tensor_tensor_nullptr_check");
    failedCases += RunExp2InvalidShapeCase("exp2_invalid_shape_check");
    failedCases += RunInplaceExp2NullptrCase("inplace_exp2_nullptr_check");

    LOG_PRINT("\n=== Summary: %d failed ===\n", failedCases);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failedCases == 0 ? 0 : 1;
}
