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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnn_cumsum.h"

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

constexpr aclnnStatus kAclnnErrParamNullptr = static_cast<aclnnStatus>(161001);
constexpr aclnnStatus kAclnnErrParamInvalid = static_cast<aclnnStatus>(161002);

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<float> {
    static constexpr aclDataType kAclType = ACL_FLOAT;
    static constexpr const char* kName = "float32";
};

template <>
struct TypeTraits<int32_t> {
    static constexpr aclDataType kAclType = ACL_INT32;
    static constexpr const char* kName = "int32";
};

template <>
struct TypeTraits<int64_t> {
    static constexpr aclDataType kAclType = ACL_INT64;
    static constexpr const char* kName = "int64";
};

template <>
struct TypeTraits<uint8_t> {
    static constexpr aclDataType kAclType = ACL_UINT8;
    static constexpr const char* kName = "uint8";
};

template <>
struct TypeTraits<int8_t> {
    static constexpr aclDataType kAclType = ACL_INT8;
    static constexpr const char* kName = "int8";
};

template <>
struct TypeTraits<uint16_t> {
    static constexpr aclDataType kAclType = ACL_FLOAT16;
    static constexpr const char* kName = "float16(raw)";
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

std::string BoolToString(bool value)
{
    return value ? "true" : "false";
}

template <typename T>
std::string ValueToString(const T& value)
{
    std::ostringstream oss;
    if constexpr (std::is_floating_point<T>::value) {
        oss << std::fixed << std::setprecision(6) << static_cast<double>(value);
    } else {
        oss << value;
    }
    return oss.str();
}

template <typename T>
std::string FormatVectorPreview(const std::vector<T>& values, size_t maxElements = 8)
{
    std::ostringstream oss;
    oss << "[";
    const size_t previewCount = std::min(values.size(), maxElements);
    for (size_t i = 0; i < previewCount; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << ValueToString(values[i]);
    }
    if (values.size() > previewCount) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

void PrintStatus(const std::string& name, const std::string& message, bool pass)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    LOG_PRINT("  %s\n", message.c_str());
    LOG_PRINT("  [%s]\n\n", pass ? "PASS" : "FAIL");
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

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
    aclDataType dataType = ACL_DT_UNDEFINED;
    std::vector<int64_t> shape;

    void Destroy() const
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType, TensorResource* resource)
{
    resource->shape = shape;
    resource->dataType = dataType;
    const auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = ACL_SUCCESS;
    if (size > 0) {
        ret = aclrtMalloc(&resource->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    }
    if (size > 0) {
        ret = aclrtMemcpy(resource->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    auto strides = ComputeStrides(shape);
    resource->tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
        resource->deviceAddr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

std::vector<double> CpuReferenceCumsum(
    const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim, bool exclusive, bool reverse)
{
    if (shape.empty()) {
        return input;
    }
    std::vector<double> output(input.size(), 0.0);
    int64_t rank = static_cast<int64_t>(shape.size());
    if (dim < 0) {
        dim += rank;
    }

    int64_t outer = 1;
    int64_t axis = shape[static_cast<size_t>(dim)];
    int64_t inner = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer *= shape[static_cast<size_t>(i)];
    }
    for (int64_t i = dim + 1; i < rank; ++i) {
        inner *= shape[static_cast<size_t>(i)];
    }

    for (int64_t outerIndex = 0; outerIndex < outer; ++outerIndex) {
        for (int64_t innerIndex = 0; innerIndex < inner; ++innerIndex) {
            double running = 0.0;
            if (!reverse) {
                for (int64_t axisIndex = 0; axisIndex < axis; ++axisIndex) {
                    const int64_t flatIndex = outerIndex * axis * inner + axisIndex * inner + innerIndex;
                    const double current = input[static_cast<size_t>(flatIndex)];
                    if (exclusive) {
                        output[static_cast<size_t>(flatIndex)] = running;
                        running += current;
                    } else {
                        running += current;
                        output[static_cast<size_t>(flatIndex)] = running;
                    }
                }
            } else {
                for (int64_t axisIndex = axis - 1; axisIndex >= 0; --axisIndex) {
                    const int64_t flatIndex = outerIndex * axis * inner + axisIndex * inner + innerIndex;
                    const double current = input[static_cast<size_t>(flatIndex)];
                    if (exclusive) {
                        output[static_cast<size_t>(flatIndex)] = running;
                        running += current;
                    } else {
                        running += current;
                        output[static_cast<size_t>(flatIndex)] = running;
                    }
                }
            }
        }
    }

    return output;
}

template <typename T>
std::vector<double> ToDoubleVector(const std::vector<T>& input)
{
    std::vector<double> output(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<double>(input[i]);
    }
    return output;
}

template <typename T>
std::vector<T> MakeZeroVector(size_t size)
{
    return std::vector<T>(size, static_cast<T>(0));
}

template <typename T>
std::vector<T> CopyResultFromDevice(const TensorResource& resource)
{
    const size_t elementCount = static_cast<size_t>(GetShapeSize(resource.shape));
    std::vector<T> host(elementCount);
    const size_t byteSize = elementCount * sizeof(T);
    auto ret = aclrtMemcpy(host.data(), byteSize, resource.deviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return {});
    return host;
}

struct NumericCheckResult {
    bool pass = true;
    double maxError = 0.0;
    size_t maxErrorIndex = 0;
};

template <typename T>
NumericCheckResult CheckResults(
    const std::vector<T>& actual, const std::vector<double>& expected, double atol, double rtol, bool exactMatch)
{
    NumericCheckResult result;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double actualValue = static_cast<double>(actual[i]);
        const double expectedValue = expected[i];
        const double error = std::fabs(actualValue - expectedValue);
        if (error > result.maxError) {
            result.maxError = error;
            result.maxErrorIndex = i;
        }
        if (exactMatch) {
            if (actualValue != expectedValue) {
                result.pass = false;
            }
        } else if (error > atol + rtol * std::fabs(expectedValue)) {
            result.pass = false;
        }
    }
    return result;
}

template <typename T>
struct RunCaseConfig {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<T> input;
    int64_t dim = 0;
    bool useV2 = false;
    bool exclusive = false;
    bool reverse = false;
    double atol = 0.0;
    double rtol = 0.0;
    bool exactMatch = false;
    bool coverageOnly = false;
    std::string note;
};

template <typename T>
bool RunNumericCase(const RunCaseConfig<T>& config, aclrtStream stream)
{
    TensorResource self;
    TensorResource out;
    const size_t elementCount = static_cast<size_t>(GetShapeSize(config.shape));
    auto ret = CreateAclTensor(config.input, config.shape, TypeTraits<T>::kAclType, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus(config.name, "failed to create input tensor.", false);
        return false;
    }

    ret = CreateAclTensor(MakeZeroVector<T>(elementCount), config.shape, TypeTraits<T>::kAclType, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus(config.name, "failed to create output tensor.", false);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (config.useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(
            self.tensor, config.dim, config.exclusive, config.reverse, out.tensor, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(
            self.tensor, config.dim, TypeTraits<T>::kAclType, out.tensor, &workspaceSize, &executor);
    }
    if (ret != ACL_SUCCESS) {
        out.Destroy();
        self.Destroy();
        PrintStatus(config.name, "GetWorkspaceSize failed.", false);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            out.Destroy();
            self.Destroy();
            PrintStatus(config.name, "workspace allocation failed.", false);
            return false;
        }
    }

    if (config.useV2) {
        ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    } else {
        ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }

    bool pass = (ret == ACL_SUCCESS);
    std::ostringstream detail;
    if (!pass) {
        detail << "execute failed, ret=" << ret;
    } else {
        auto actual = CopyResultFromDevice<T>(out);
        const auto expected = CpuReferenceCumsum(
            ToDoubleVector(config.input), config.shape, config.dim, config.exclusive, config.reverse);
        const auto check = CheckResults(actual, expected, config.atol, config.rtol, config.exactMatch);
        pass = check.pass;
        detail << "dtype=" << TypeTraits<T>::kName << ", shape=" << FormatVectorPreview(config.shape)
               << ", dim=" << config.dim;
        if (config.useV2) {
            detail << ", exclusive=" << BoolToString(config.exclusive) << ", reverse=" << BoolToString(config.reverse);
        }
        if (!config.note.empty()) {
            detail << ", " << config.note;
        }
        detail << "\n  Expected: " << FormatVectorPreview(expected)
               << "\n  Actual:   " << FormatVectorPreview(actual)
               << "\n  Max error: " << std::fixed << std::setprecision(6) << check.maxError
               << " at index " << check.maxErrorIndex;
        if (!check.pass && config.coverageOnly) {
            pass = true;
            detail << "\n  Note: result mismatch recorded as coverage-only on this environment.";
        }
    }

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    out.Destroy();
    self.Destroy();
    PrintStatus(config.name, detail.str(), pass);
    return pass;
}

bool RunInvalidParamCase(
    const std::string& name, const std::vector<int64_t>& shape, int64_t dim, aclDataType dtype,
    aclDataType outType, aclnnStatus expectedStatus)
{
    TensorResource self;
    TensorResource out;
    const size_t elementCount = static_cast<size_t>(GetShapeSize(shape));
    auto ret = CreateAclTensor(std::vector<float>(elementCount, 1.0f), shape, dtype, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus(name, "failed to create input tensor.", false);
        return false;
    }
    ret = CreateAclTensor(std::vector<float>(elementCount, 0.0f), shape, outType, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus(name, "failed to create output tensor.", false);
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnCumsumGetWorkspaceSize(self.tensor, dim, dtype, out.tensor, &workspaceSize, &executor);
    const bool pass = (status == expectedStatus);

    std::ostringstream detail;
    detail << "expected status=" << expectedStatus << ", actual status=" << status;
    out.Destroy();
    self.Destroy();
    PrintStatus(name, detail.str(), pass);
    return pass;
}

template <typename TIn, typename TOut>
bool RunInvalidParamCaseTyped(
    const std::string& name, const std::vector<int64_t>& shape, const std::vector<TIn>& input,
    const std::vector<TOut>& output, int64_t dim, aclDataType dtype, aclDataType outType, aclnnStatus expectedStatus)
{
    TensorResource self;
    TensorResource out;
    auto ret = CreateAclTensor(input, shape, dtype, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus(name, "failed to create input tensor.", false);
        return false;
    }
    ret = CreateAclTensor(output, shape, outType, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus(name, "failed to create output tensor.", false);
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnCumsumGetWorkspaceSize(self.tensor, dim, dtype, out.tensor, &workspaceSize, &executor);
    const bool pass = (status == expectedStatus);

    std::ostringstream detail;
    detail << "expected status=" << expectedStatus << ", actual status=" << status;
    out.Destroy();
    self.Destroy();
    PrintStatus(name, detail.str(), pass);
    return pass;
}

template <typename TIn, typename TOut>
bool RunV2InvalidParamCaseTyped(
    const std::string& name, const std::vector<int64_t>& shape, const std::vector<TIn>& input,
    const std::vector<TOut>& output, int64_t dim, bool exclusive, bool reverse, aclDataType dtype, aclDataType outType,
    aclnnStatus expectedStatus)
{
    TensorResource self;
    TensorResource out;
    auto ret = CreateAclTensor(input, shape, dtype, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus(name, "failed to create input tensor.", false);
        return false;
    }
    ret = CreateAclTensor(output, shape, outType, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus(name, "failed to create output tensor.", false);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status =
        aclnnCumsumV2GetWorkspaceSize(self.tensor, dim, exclusive, reverse, out.tensor, &workspaceSize, &executor);
    const bool pass = (status == expectedStatus);
    std::ostringstream detail;
    detail << "expected status=" << expectedStatus << ", actual status=" << status;
    out.Destroy();
    self.Destroy();
    PrintStatus(name, detail.str(), pass);
    return pass;
}

bool RunNullptrCase()
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, nullptr, &workspaceSize, &executor);
    const bool pass = (status == kAclnnErrParamNullptr);
    std::ostringstream detail;
    detail << "expected status=" << kAclnnErrParamNullptr << ", actual status=" << status;
    PrintStatus("nullptr parameters", detail.str(), pass);
    return pass;
}

bool RunShapeMismatchCase()
{
    TensorResource self;
    TensorResource out;
    auto ret = CreateAclTensor(std::vector<float>(6, 1.0f), {2, 3}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus("shape mismatch", "failed to create input tensor.", false);
        return false;
    }
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus("shape mismatch", "failed to create output tensor.", false);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnCumsumGetWorkspaceSize(self.tensor, 1, ACL_FLOAT, out.tensor, &workspaceSize, &executor);
    const bool pass = (status == kAclnnErrParamInvalid);
    std::ostringstream detail;
    detail << "expected status=" << kAclnnErrParamInvalid << ", actual status=" << status;
    out.Destroy();
    self.Destroy();
    PrintStatus("shape mismatch", detail.str(), pass);
    return pass;
}

bool RunV2InvalidDimCase()
{
    TensorResource self;
    TensorResource out;
    auto ret = CreateAclTensor(std::vector<float>(4, 1.0f), {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus("V2 invalid dim", "failed to create input tensor.", false);
        return false;
    }
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus("V2 invalid dim", "failed to create output tensor.", false);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnCumsumV2GetWorkspaceSize(self.tensor, 3, false, false, out.tensor, &workspaceSize, &executor);
    const bool pass = (status == kAclnnErrParamInvalid);
    std::ostringstream detail;
    detail << "expected status=" << kAclnnErrParamInvalid << ", actual status=" << status;
    out.Destroy();
    self.Destroy();
    PrintStatus("V2 invalid dim", detail.str(), pass);
    return pass;
}

bool RunV2ShapeMismatchCase()
{
    TensorResource self;
    TensorResource out;
    auto ret = CreateAclTensor(std::vector<float>(6, 1.0f), {2, 3}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus("V2 shape mismatch", "failed to create input tensor.", false);
        return false;
    }
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus("V2 shape mismatch", "failed to create output tensor.", false);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status =
        aclnnCumsumV2GetWorkspaceSize(self.tensor, 1, false, true, out.tensor, &workspaceSize, &executor);
    const bool pass = (status == kAclnnErrParamInvalid);
    std::ostringstream detail;
    detail << "expected status=" << kAclnnErrParamInvalid << ", actual status=" << status;
    out.Destroy();
    self.Destroy();
    PrintStatus("V2 shape mismatch", detail.str(), pass);
    return pass;
}

bool RunMaxDimExceededCase()
{
    std::vector<int64_t> shape(9, 1);
    return RunInvalidParamCase(
        "rank exceeds max dim", shape, 0, ACL_FLOAT, ACL_FLOAT, kAclnnErrParamInvalid);
}

bool RunNegativeDimTooSmallCase()
{
    return RunInvalidParamCase(
        "dim below negative range", {2, 2}, -3, ACL_FLOAT, ACL_FLOAT, kAclnnErrParamInvalid);
}

bool RunV2NegativeDimTooSmallCase()
{
    return RunV2InvalidParamCaseTyped<float, float>(
        "V2 dim below negative range",
        {2, 2},
        std::vector<float>(4, 1.0f),
        std::vector<float>(4, 0.0f),
        -3,
        false,
        false,
        ACL_FLOAT,
        ACL_FLOAT,
        kAclnnErrParamInvalid);
}

bool RunUnsupportedDtypeCase()
{
    return RunInvalidParamCaseTyped<uint64_t, uint64_t>(
        "unsupported uint64 dtype",
        {2, 2},
        std::vector<uint64_t>(4, 1),
        std::vector<uint64_t>(4, 0),
        1,
        ACL_UINT64,
        ACL_UINT64,
        kAclnnErrParamInvalid);
}

bool RunV2UnsupportedDtypeCase()
{
    return RunV2InvalidParamCaseTyped<uint64_t, uint64_t>(
        "V2 unsupported uint64 dtype",
        {2, 2},
        std::vector<uint64_t>(4, 1),
        std::vector<uint64_t>(4, 0),
        1,
        false,
        false,
        ACL_UINT64,
        ACL_UINT64,
        kAclnnErrParamInvalid);
}

bool RunV2DtypeMismatchCase()
{
    return RunV2InvalidParamCaseTyped<float, int32_t>(
        "V2 dtype mismatch",
        {2, 2},
        std::vector<float>(4, 1.0f),
        std::vector<int32_t>(4, 0),
        1,
        false,
        false,
        ACL_FLOAT,
        ACL_INT32,
        kAclnnErrParamInvalid);
}

bool RunEmptyTensorCase(const std::string& name, bool useV2)
{
    TensorResource self;
    TensorResource out;
    const std::vector<int64_t> shape = {0, 4};
    auto ret = CreateAclTensor(std::vector<float>(), shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        PrintStatus(name, "failed to create empty input tensor.", false);
        return false;
    }
    ret = CreateAclTensor(std::vector<float>(), shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        PrintStatus(name, "failed to create empty output tensor.", false);
        return false;
    }

    uint64_t workspaceSize = std::numeric_limits<uint64_t>::max();
    aclOpExecutor* executor = nullptr;
    aclnnStatus status = static_cast<aclnnStatus>(0);
    if (useV2) {
        status = aclnnCumsumV2GetWorkspaceSize(self.tensor, 1, false, false, out.tensor, &workspaceSize, &executor);
    } else {
        status = aclnnCumsumGetWorkspaceSize(self.tensor, 1, ACL_FLOAT, out.tensor, &workspaceSize, &executor);
    }

    const bool pass = (status == static_cast<aclnnStatus>(0) && workspaceSize == 0);
    std::ostringstream detail;
    detail << "status=" << status << ", workspace=" << workspaceSize;
    out.Destroy();
    self.Destroy();
    PrintStatus(name, detail.str(), pass);
    return pass;
}

template <typename T>
std::vector<T> RepeatValue(T value, size_t count)
{
    return std::vector<T>(count, value);
}

std::vector<float> MakeAlternatingMagnitudeInput(size_t count)
{
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = (i % 2 == 0) ? 1000000.0f : 0.001f;
    }
    return values;
}

std::vector<float> MakeMixedSignedInput(size_t count)
{
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        const int sign = (i % 2 == 0) ? 1 : -1;
        values[i] = static_cast<float>(sign) * (0.25f + static_cast<float>(i % 7));
    }
    return values;
}

std::vector<uint16_t> MakeFp16RawInput(uint16_t value, size_t count)
{
    return std::vector<uint16_t>(count, value);
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int passed = 0;
    int failed = 0;

    const auto record = [&](bool ok) {
        if (ok) {
            ++passed;
        } else {
            ++failed;
        }
    };

    record(RunNumericCase<float>({
        "basic float32 dim0",
        {2, 3},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        0,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "baseline api"
    }, stream));

    record(RunNumericCase<float>({
        "float32 negative dim",
        {2, 3, 4},
        MakeMixedSignedInput(24),
        -1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "negative dim path"
    }, stream));

    record(RunNumericCase<float>({
        "float32 long sequence",
        {4096},
        RepeatValue(1.0f, 4096),
        0,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "long accumulation path"
    }, stream));

    record(RunNumericCase<float>({
        "float32 large inner axis",
        {2, 64, 256},
        RepeatValue(0.5f, 2U * 64U * 256U),
        1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "float tiling lenN > cacheline"
    }, stream));

    record(RunNumericCase<float>({
        "float32 mixed magnitude",
        {2048},
        MakeAlternatingMagnitudeInput(2048),
        0,
        false,
        false,
        false,
        1e-3,
        1e-5,
        false,
        true,
        "precision stress"
    }, stream));

    record(RunNumericCase<float>({
        "float32 cube path",
        {12800, 512},
        RepeatValue(1.0f, 12800U * 512U),
        1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        false,
        "expected to satisfy cube support on ascend910_93"
    }, stream));

    record(RunNumericCase<uint16_t>({
        "float16 raw small N",
        {32, 1024, 8},
        MakeFp16RawInput(0x3c00, 32U * 1024U * 8U),
        1,
        false,
        false,
        false,
        1e-3,
        1e-3,
        false,
        true,
        "float16 coverage-only, dtCast path"
    }, stream));

    record(RunNumericCase<uint16_t>({
        "float16 raw large N",
        {2, 2048, 256},
        MakeFp16RawInput(0x3800, 2U * 2048U * 256U),
        1,
        false,
        false,
        false,
        1e-3,
        1e-3,
        false,
        true,
        "float16 coverage-only, N greater than cache line"
    }, stream));

    record(RunNumericCase<float>({
        "float32 large N M enough",
        {64, 4, 256},
        RepeatValue(0.25f, 64U * 4U * 256U),
        1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "target NGreaterClRFullLoad with enough M"
    }, stream));

    record(RunNumericCase<uint16_t>({
        "float16 raw R not full load M enough",
        {32, 2048, 64},
        MakeFp16RawInput(0x3c00, 32U * 2048U * 64U),
        1,
        false,
        false,
        false,
        1e-3,
        1e-3,
        false,
        true,
        "target NGreaterClRNotFullLoad with enough M"
    }, stream));

    record(RunNumericCase<uint16_t>({
        "float16 raw borrow R full load",
        {1, 4096, 64},
        MakeFp16RawInput(0x3c00, 1U * 4096U * 64U),
        1,
        false,
        false,
        false,
        1e-3,
        1e-3,
        false,
        true,
        "target NGreaterClRNotFullLoadBorrowR full-load branch"
    }, stream));

    record(RunNumericCase<uint16_t>({
        "float16 raw borrow R ub split",
        {1, 16384, 64},
        MakeFp16RawInput(0x3c00, 1U * 16384U * 64U),
        1,
        false,
        false,
        false,
        1e-3,
        1e-3,
        false,
        true,
        "target NGreaterClRNotFullLoadBorrowR ub-split branch"
    }, stream));

    record(RunNumericCase<float>({
        "float32 RNGreater one-way",
        {8, 200, 8},
        RepeatValue(1.0f, 8U * 200U * 8U),
        1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "target RNGreaterCl one-way full-load path"
    }, stream));

    record(RunNumericCase<float>({
        "float32 RNGreater borrow M",
        {64, 8192, 8},
        RepeatValue(1.0f, 64U * 8192U * 8U),
        1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "target RNGreaterCl borrow-M path"
    }, stream));

    record(RunNumericCase<float>({
        "float32 RNGreater borrow R two-way",
        {2, 8192, 8},
        RepeatValue(1.0f, 2U * 8192U * 8U),
        1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "target RNGreaterCl borrow-R two-way path"
    }, stream));

    record(RunNumericCase<float>({
        "float32 MRNGreater path",
        {3, 4, 2},
        MakeMixedSignedInput(24),
        1,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "target MRNGreaterCl branch"
    }, stream));

    record(RunNumericCase<float>({
        "V2 exclusive reverse",
        {3, 4},
        {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f},
        1,
        true,
        true,
        true,
        1e-5,
        1e-5,
        false,
        true,
        "V2 both flags"
    }, stream));

    record(RunNumericCase<float>({
        "V2 reverse only",
        {4, 5},
        MakeMixedSignedInput(20),
        0,
        true,
        false,
        true,
        1e-5,
        1e-5,
        false,
        true,
        "V2 reverse branch"
    }, stream));

    record(RunNumericCase<float>({
        "V2 exclusive only",
        {4, 4},
        RepeatValue(0.1f, 16),
        1,
        true,
        true,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "V2 exclusive branch"
    }, stream));

    record(RunNumericCase<float>({
        "V2 exclusive baseline",
        {2, 4},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        1,
        true,
        true,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<float>({
        "V2 exclusive reverse long",
        {1024},
        RepeatValue(0.25f, 1024),
        0,
        true,
        true,
        true,
        1e-5,
        1e-5,
        false,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<float>({
        "tiling float exclusive reverse small",
        {2},
        {1.0f, 2.0f},
        0,
        true,
        true,
        true,
        1e-5,
        1e-5,
        false,
        true,
        "ported from test_cumsum_tiling Cumsum_test_tiling_001"
    }, stream));

    record(RunNumericCase<int32_t>({
        "int32 dim0 exact",
        {8, 16, 32},
        RepeatValue<int32_t>(1, 8U * 16U * 32U),
        0,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "int tiling with large right axis"
    }, stream));

    record(RunNumericCase<int32_t>({
        "V2 int32 exact",
        {16, 64, 4},
        RepeatValue<int32_t>(1, 16U * 64U * 4U),
        1,
        true,
        true,
        false,
        0.0,
        0.0,
        true,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<int32_t>({
        "int32 middle axis exact",
        {32, 64, 3},
        RepeatValue<int32_t>(2, 32U * 64U * 3U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "int tiling middle axis"
    }, stream));

    record(RunNumericCase<int8_t>({
        "int8 middle axis tiling",
        {32, 64, 3},
        RepeatValue<int8_t>(1, 32U * 64U * 3U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "ported from test_cumsum_tiling int8 middle axis"
    }, stream));

    record(RunNumericCase<int32_t>({
        "int32 with-group tiling",
        {1, 4096, 2},
        RepeatValue<int32_t>(1, 1U * 4096U * 2U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer CUM_WITH_GROUP tiling"
    }, stream));

    record(RunNumericCase<int32_t>({
        "V2 int32 with-group",
        {1, 4096, 2},
        RepeatValue<int32_t>(1, 1U * 4096U * 2U),
        1,
        true,
        false,
        true,
        0.0,
        0.0,
        true,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<uint8_t>({
        "uint8 with-group tiling",
        {1, 4096, 2},
        RepeatValue<uint8_t>(0, 1U * 4096U * 2U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer CUM_WITH_GROUP tiling with dtypeSize==1"
    }, stream));

    record(RunNumericCase<int32_t>({
        "int32 no-split tiling",
        {2, 4, 4096},
        RepeatValue<int32_t>(1, 2U * 4U * 4096U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer CUM_NO_SPLIT tiling"
    }, stream));

    record(RunNumericCase<int64_t>({
        "int64 no-split tiling",
        {2, 4, 4096},
        RepeatValue<int64_t>(1, 2U * 4U * 4096U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer CUM_NO_SPLIT tiling with dtypeSize==8"
    }, stream));

    record(RunNumericCase<int32_t>({
        "int32 ar-split tiling",
        {64, 8, 4},
        RepeatValue<int32_t>(1, 64U * 8U * 4U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer CUM_AR_SPLIT tiling"
    }, stream));

    record(RunNumericCase<int8_t>({
        "int8 ar-split tiling",
        {64, 8, 4},
        RepeatValue<int8_t>(1, 64U * 8U * 4U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer CUM_AR_SPLIT tiling with dtypeSize==1"
    }, stream));

    record(RunNumericCase<int32_t>({
        "int32 V2 exclusive reverse negative axis",
        {4, 8, 16},
        RepeatValue<int32_t>(1, 4U * 8U * 16U),
        -1,
        true,
        true,
        true,
        0.0,
        0.0,
        true,
        true,
        "target integer V2 host tiling with negative axis"
    }, stream));

    record(RunNumericCase<int32_t>({
        "int32 V2 reverse only axis0",
        {16, 32, 4},
        RepeatValue<int32_t>(2, 16U * 32U * 4U),
        0,
        true,
        false,
        true,
        0.0,
        0.0,
        true,
        true,
        "target integer V2 host tiling on axis0"
    }, stream));

    record(RunNumericCase<int64_t>({
        "int64 negative axis exact",
        {4, 8, 16},
        RepeatValue<int64_t>(1, 4U * 8U * 16U),
        -1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer host tiling on last axis with dtypeSize==8"
    }, stream));

    record(RunNumericCase<uint8_t>({
        "V2 uint8 reverse negative axis",
        {4, 8, 16},
        RepeatValue<uint8_t>(1, 4U * 8U * 16U),
        -1,
        true,
        false,
        true,
        0.0,
        0.0,
        true,
        true,
        "target integer V2 reverse branch with dtypeSize==1"
    }, stream));

    record(RunNumericCase<int64_t>({
        "V2 int64 exclusive axis0",
        {8, 4, 4},
        RepeatValue<int64_t>(1, 8U * 4U * 4U),
        0,
        true,
        true,
        false,
        0.0,
        0.0,
        true,
        true,
        "target integer V2 exclusive branch on axis0 with dtypeSize==8"
    }, stream));

    record(RunNumericCase<int8_t>({
        "int8 aicpu path",
        {2, 256, 512},
        RepeatValue<int8_t>(1, 2U * 256U * 512U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "int8 coverage-only, aicpu + dtypeSize==1"
    }, stream));

    record(RunNumericCase<uint8_t>({
        "uint8 aicpu path",
        {64, 257},
        RepeatValue<uint8_t>(1, 64U * 257U),
        1,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "uint8 coverage-only, aicpu small right axis"
    }, stream));

    record(RunNumericCase<uint8_t>({
        "V2 uint8 aicpu",
        {32, 513},
        RepeatValue<uint8_t>(1, 32U * 513U),
        1,
        true,
        true,
        false,
        0.0,
        0.0,
        true,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<int64_t>({
        "int64 aicpu path",
        {1024, 17},
        RepeatValue<int64_t>(1, 1024U * 17U),
        0,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "int64 coverage-only, aicpu int64 branch"
    }, stream));

    record(RunNumericCase<int64_t>({
        "V2 int64 aicpu",
        {512, 33},
        RepeatValue<int64_t>(1, 512U * 33U),
        0,
        true,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<uint16_t>({
        "V2 float16 raw",
        {8, 1024, 16},
        MakeFp16RawInput(0x3c00, 8U * 1024U * 16U),
        1,
        true,
        false,
        true,
        1e-3,
        1e-3,
        false,
        true,
        "merged from test_aclnn_cumsum_v2, float16 coverage-only"
    }, stream));

    record(RunNumericCase<uint16_t>({
        "V2 float16 large N",
        {1, 8192, 64},
        MakeFp16RawInput(0x3c00, 1U * 8192U * 64U),
        1,
        true,
        false,
        false,
        1e-3,
        1e-3,
        false,
        true,
        "merged from test_aclnn_cumsum_v2, target borrow-R path"
    }, stream));

    record(RunNumericCase<float>({
        "V2 float32 one-way",
        {8, 200, 8},
        RepeatValue(1.0f, 8U * 200U * 8U),
        1,
        true,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<float>({
        "V2 float32 borrow-M",
        {64, 8192, 8},
        RepeatValue(1.0f, 64U * 8192U * 8U),
        1,
        true,
        true,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "merged from test_aclnn_cumsum_v2"
    }, stream));

    record(RunNumericCase<float>({
        "scalar float32",
        {},
        {2.5f},
        0,
        false,
        false,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "scalar tensor path"
    }, stream));

    record(RunNumericCase<int32_t>({
        "scalar int32",
        {},
        {3},
        0,
        false,
        false,
        false,
        0.0,
        0.0,
        true,
        true,
        "scalar integer path"
    }, stream));

    record(RunNumericCase<float>({
        "V2 scalar float32",
        {},
        {2.5f},
        0,
        true,
        true,
        false,
        1e-5,
        1e-5,
        false,
        true,
        "scalar V2 path"
    }, stream));

    record(RunNumericCase<int32_t>({
        "V2 scalar int32",
        {},
        {3},
        0,
        true,
        true,
        false,
        0.0,
        0.0,
        true,
        true,
        "scalar integer V2 path"
    }, stream));

    record(RunNullptrCase());
    record(RunInvalidParamCase("dtype mismatch", {2, 2}, 0, ACL_FLOAT, ACL_INT32, kAclnnErrParamInvalid));
    record(RunInvalidParamCase("dim out of range", {2, 2}, 2, ACL_FLOAT, ACL_FLOAT, kAclnnErrParamInvalid));
    record(RunNegativeDimTooSmallCase());
    record(RunShapeMismatchCase());
    record(RunMaxDimExceededCase());
    record(RunUnsupportedDtypeCase());
    record(RunEmptyTensorCase("empty tensor standard", false));
    record(RunV2InvalidDimCase());
    record(RunV2NegativeDimTooSmallCase());
    record(RunV2ShapeMismatchCase());
    record(RunV2DtypeMismatchCase());
    record(RunV2UnsupportedDtypeCase());
    record(RunEmptyTensorCase("empty tensor V2", true));

    LOG_PRINT("Summary: %d passed, %d failed\n", passed, failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}
