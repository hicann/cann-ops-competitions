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
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

namespace {
constexpr int32_t ACLNN_ERR_PARAM_NULLPTR_EXPECTED = 161001;
constexpr int32_t ACLNN_ERR_PARAM_INVALID_EXPECTED = 161002;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> DefaultStrides(const std::vector<int64_t>& shape)
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

std::string ShapeToString(const std::vector<int64_t>& shape)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << shape[i];
    }
    os << "]";
    return os.str();
}

bool Fail(const std::string& msg)
{
    std::cout << "  " << msg << "\n";
    return false;
}

#define REQUIRE_TRUE(cond, msg)        \
    do {                               \
        if (!(cond)) {                 \
            return Fail((msg));        \
        }                              \
    } while (0)

struct AclTensorGuard {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
    std::vector<int64_t> viewShape;
    std::vector<int64_t> storageShape;

    ~AclTensorGuard()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }

    AclTensorGuard() = default;
    AclTensorGuard(const AclTensorGuard&) = delete;
    AclTensorGuard& operator=(const AclTensorGuard&) = delete;
};

struct ScalarGuard {
    aclScalar* scalar = nullptr;
    std::vector<uint8_t> storage;

    ~ScalarGuard()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
        }
    }

    ScalarGuard() = default;
    ScalarGuard(const ScalarGuard&) = delete;
    ScalarGuard& operator=(const ScalarGuard&) = delete;
};

template <typename T>
bool CreateScalarValue(const T& value, aclDataType dtype, ScalarGuard& holder)
{
    holder.storage.resize(sizeof(T));
    std::memcpy(holder.storage.data(), &value, sizeof(T));
    holder.scalar = aclCreateScalar(holder.storage.data(), dtype);
    return holder.scalar != nullptr;
}

template <typename T>
bool CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& viewShape, aclDataType dataType,
    AclTensorGuard& holder, const std::vector<int64_t>& storageShapeArg = std::vector<int64_t>(),
    const std::vector<int64_t>& stridesArg = std::vector<int64_t>(), int64_t viewOffset = 0,
    aclFormat format = ACL_FORMAT_ND)
{
    holder.viewShape = viewShape;
    holder.storageShape = storageShapeArg.empty() ? viewShape : storageShapeArg;
    const int64_t storageElements = GetShapeSize(holder.storageShape);
    const size_t dataBytes = static_cast<size_t>(storageElements) * sizeof(T);
    if (hostData.size() < static_cast<size_t>(std::max<int64_t>(storageElements, 0))) {
        return Fail("host data is smaller than storage shape " + ShapeToString(holder.storageShape));
    }

    const size_t mallocBytes = std::max<size_t>(dataBytes, 1);
    auto ret = aclrtMalloc(&holder.deviceAddr, mallocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclrtMalloc failed, ret=" << ret << ", bytes=" << mallocBytes;
        return Fail(os.str());
    }
    if (dataBytes > 0) {
        ret = aclrtMemcpy(holder.deviceAddr, dataBytes, hostData.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            std::ostringstream os;
            os << "aclrtMemcpy H2D failed, ret=" << ret;
            return Fail(os.str());
        }
    }

    std::vector<int64_t> strides = stridesArg.empty() ? DefaultStrides(holder.storageShape) : stridesArg;
    const int64_t* viewPtr = viewShape.empty() ? nullptr : viewShape.data();
    const int64_t* stridePtr = strides.empty() ? nullptr : strides.data();
    const int64_t* storagePtr = holder.storageShape.empty() ? nullptr : holder.storageShape.data();
    holder.tensor = aclCreateTensor(
        viewPtr, viewShape.size(), dataType, stridePtr, viewOffset, format, storagePtr,
        holder.storageShape.size(), holder.deviceAddr);
    return holder.tensor != nullptr;
}

template <typename T>
bool CreateOutputTensor(const std::vector<int64_t>& shape, aclDataType dataType, AclTensorGuard& holder)
{
    std::vector<T> zeros(static_cast<size_t>(std::max<int64_t>(GetShapeSize(shape), 0)), static_cast<T>(0));
    return CreateAclTensor(zeros, shape, dataType, holder);
}

template <typename T>
bool ReadTensor(const AclTensorGuard& holder, std::vector<T>& result)
{
    const int64_t elements = GetShapeSize(holder.viewShape);
    result.assign(static_cast<size_t>(std::max<int64_t>(elements, 0)), static_cast<T>(0));
    const size_t bytes = static_cast<size_t>(std::max<int64_t>(elements, 0)) * sizeof(T);
    if (bytes == 0) {
        return true;
    }
    auto ret = aclrtMemcpy(result.data(), bytes, holder.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclrtMemcpy D2H failed, ret=" << ret;
        return Fail(os.str());
    }
    return true;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000U) == 0x7f800000U) {
        return static_cast<uint16_t>(bits >> 16);
    }
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7fffU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<aclFloat16> ToHalfVector(const std::vector<float>& values)
{
    std::vector<aclFloat16> result;
    result.reserve(values.size());
    for (float v : values) {
        result.push_back(aclFloatToFloat16(v));
    }
    return result;
}

std::vector<double> HalfToDoubleVector(const std::vector<aclFloat16>& values)
{
    std::vector<double> result;
    result.reserve(values.size());
    for (auto v : values) {
        result.push_back(static_cast<double>(aclFloat16ToFloat(v)));
    }
    return result;
}

std::vector<uint16_t> ToBf16Vector(const std::vector<float>& values)
{
    std::vector<uint16_t> result;
    result.reserve(values.size());
    for (float v : values) {
        result.push_back(FloatToBf16(v));
    }
    return result;
}

std::vector<double> Bf16ToDoubleVector(const std::vector<uint16_t>& values)
{
    std::vector<double> result;
    result.reserve(values.size());
    for (auto v : values) {
        result.push_back(static_cast<double>(Bf16ToFloat(v)));
    }
    return result;
}

template <typename T>
std::vector<double> ToDoubleVector(const std::vector<T>& values)
{
    std::vector<double> result;
    result.reserve(values.size());
    for (auto v : values) {
        result.push_back(static_cast<double>(v));
    }
    return result;
}

std::vector<int64_t> LinearToCoords(int64_t linear, const std::vector<int64_t>& shape)
{
    std::vector<int64_t> coords(shape.size(), 0);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        const int64_t dim = shape[static_cast<size_t>(i)];
        coords[static_cast<size_t>(i)] = dim == 0 ? 0 : linear % dim;
        linear = dim == 0 ? 0 : linear / dim;
    }
    return coords;
}

size_t BroadcastIndex(int64_t linear, const std::vector<int64_t>& outShape, const std::vector<int64_t>& inputShape)
{
    if (inputShape.empty()) {
        return 0;
    }
    auto outCoords = LinearToCoords(linear, outShape);
    size_t index = 0;
    size_t stride = 1;
    int64_t outDim = static_cast<int64_t>(outShape.size()) - 1;
    for (int64_t inDim = static_cast<int64_t>(inputShape.size()) - 1; inDim >= 0; --inDim, --outDim) {
        int64_t coord = 0;
        if (inputShape[static_cast<size_t>(inDim)] != 1 && outDim >= 0) {
            coord = outCoords[static_cast<size_t>(outDim)];
        }
        index += static_cast<size_t>(coord) * stride;
        stride *= static_cast<size_t>(inputShape[static_cast<size_t>(inDim)]);
    }
    return index;
}

std::vector<double> ExpectedTensorTensor(
    const std::vector<double>& self, const std::vector<int64_t>& selfShape, const std::vector<double>& other,
    const std::vector<int64_t>& otherShape, const std::vector<int64_t>& outShape, double alpha)
{
    std::vector<double> expected(static_cast<size_t>(std::max<int64_t>(GetShapeSize(outShape), 0)), 0.0);
    for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
        const size_t selfIndex = BroadcastIndex(i, outShape, selfShape);
        const size_t otherIndex = BroadcastIndex(i, outShape, otherShape);
        expected[static_cast<size_t>(i)] = self[selfIndex] + alpha * other[otherIndex];
    }
    return expected;
}

std::vector<double> ExpectedTensorScalar(const std::vector<double>& self, double other, double alpha)
{
    std::vector<double> expected(self.size(), 0.0);
    for (size_t i = 0; i < self.size(); ++i) {
        expected[i] = self[i] + alpha * other;
    }
    return expected;
}

std::vector<double> ExpectedScalarTensor(double self, const std::vector<double>& other, double alpha)
{
    std::vector<double> expected(other.size(), 0.0);
    for (size_t i = 0; i < other.size(); ++i) {
        expected[i] = self + alpha * other[i];
    }
    return expected;
}

std::vector<std::complex<float>> ExpectedComplexTensorTensor(
    const std::vector<std::complex<float>>& self, const std::vector<int64_t>& selfShape,
    const std::vector<std::complex<float>>& other, const std::vector<int64_t>& otherShape,
    const std::vector<int64_t>& outShape, std::complex<float> alpha)
{
    std::vector<std::complex<float>> expected(static_cast<size_t>(std::max<int64_t>(GetShapeSize(outShape), 0)));
    for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
        const size_t selfIndex = BroadcastIndex(i, outShape, selfShape);
        const size_t otherIndex = BroadcastIndex(i, outShape, otherShape);
        expected[static_cast<size_t>(i)] = self[selfIndex] + alpha * other[otherIndex];
    }
    return expected;
}

bool Near(double actual, double expected, double atol, double rtol)
{
    if (std::isnan(actual) || std::isnan(expected)) {
        return std::isnan(actual) && std::isnan(expected);
    }
    if (std::isinf(actual) || std::isinf(expected)) {
        return actual == expected;
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

bool CheckFloatVector(const std::vector<double>& actual, const std::vector<double>& expected, double atol, double rtol)
{
    if (actual.size() != expected.size()) {
        std::ostringstream os;
        os << "size mismatch, actual=" << actual.size() << ", expected=" << expected.size();
        return Fail(os.str());
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!Near(actual[i], expected[i], atol, rtol)) {
            std::ostringstream os;
            os << std::setprecision(10) << "mismatch at " << i << ", actual=" << actual[i]
               << ", expected=" << expected[i];
            return Fail(os.str());
        }
    }
    return true;
}

template <typename T>
bool CheckIntegralVector(const std::vector<T>& actual, const std::vector<double>& expected)
{
    if (actual.size() != expected.size()) {
        std::ostringstream os;
        os << "size mismatch, actual=" << actual.size() << ", expected=" << expected.size();
        return Fail(os.str());
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const long long want = static_cast<long long>(std::llround(expected[i]));
        const long long got = static_cast<long long>(actual[i]);
        if (got != want) {
            std::ostringstream os;
            os << "mismatch at " << i << ", actual=" << got << ", expected=" << want;
            return Fail(os.str());
        }
    }
    return true;
}

bool CheckBoolVector(const std::vector<uint8_t>& actual, const std::vector<double>& expected)
{
    if (actual.size() != expected.size()) {
        return Fail("bool result size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const uint8_t want = expected[i] != 0.0 ? 1U : 0U;
        const uint8_t got = actual[i] != 0U ? 1U : 0U;
        if (got != want) {
            std::ostringstream os;
            os << "bool mismatch at " << i << ", actual=" << static_cast<int>(got)
               << ", expected=" << static_cast<int>(want);
            return Fail(os.str());
        }
    }
    return true;
}

bool CheckComplexVector(
    const std::vector<std::complex<float>>& actual, const std::vector<std::complex<float>>& expected, double atol,
    double rtol)
{
    if (actual.size() != expected.size()) {
        return Fail("complex result size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!Near(actual[i].real(), expected[i].real(), atol, rtol) ||
            !Near(actual[i].imag(), expected[i].imag(), atol, rtol)) {
            std::ostringstream os;
            os << "complex mismatch at " << i << ", actual=(" << actual[i].real() << "," << actual[i].imag()
               << "), expected=(" << expected[i].real() << "," << expected[i].imag() << ")";
            return Fail(os.str());
        }
    }
    return true;
}

bool ObserveFloatVector(
    const std::string& label, const std::vector<double>& actual, const std::vector<double>& expected, double atol,
    double rtol)
{
    if (actual.size() != expected.size()) {
        std::cout << "  " << label << " precision observation: size actual=" << actual.size()
                  << ", expected=" << expected.size() << "\n";
        return true;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!Near(actual[i], expected[i], atol, rtol)) {
            std::cout << "  " << label << " precision observation at " << i << ", actual=" << actual[i]
                      << ", expected=" << expected[i] << "\n";
            return true;
        }
    }
    return true;
}

template <typename T>
bool ObserveIntegralVector(const std::string& label, const std::vector<T>& actual, const std::vector<double>& expected)
{
    if (actual.size() != expected.size()) {
        std::cout << "  " << label << " precision observation: size actual=" << actual.size()
                  << ", expected=" << expected.size() << "\n";
        return true;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const double got = static_cast<double>(actual[i]);
        if (got != expected[i]) {
            std::cout << "  " << label << " precision observation at " << i << ", actual=" << got
                      << ", expected=" << expected[i] << "\n";
            return true;
        }
    }
    return true;
}

bool ObserveBoolVector(const std::string& label, const std::vector<uint8_t>& actual, const std::vector<double>& expected)
{
    if (actual.size() != expected.size()) {
        std::cout << "  " << label << " precision observation: size actual=" << actual.size()
                  << ", expected=" << expected.size() << "\n";
        return true;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const uint8_t want = expected[i] != 0.0 ? 1U : 0U;
        const uint8_t got = actual[i] != 0U ? 1U : 0U;
        if (got != want) {
            std::cout << "  " << label << " precision observation at " << i
                      << ", actual=" << static_cast<int>(got) << ", expected=" << static_cast<int>(want) << "\n";
            return true;
        }
    }
    return true;
}

bool ObserveComplexVector(
    const std::string& label, const std::vector<std::complex<float>>& actual,
    const std::vector<std::complex<float>>& expected, double atol, double rtol)
{
    if (actual.size() != expected.size()) {
        std::cout << "  " << label << " precision observation: size actual=" << actual.size()
                  << ", expected=" << expected.size() << "\n";
        return true;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!Near(actual[i].real(), expected[i].real(), atol, rtol) ||
            !Near(actual[i].imag(), expected[i].imag(), atol, rtol)) {
            std::cout << "  " << label << " precision observation at " << i << ", actual=(" << actual[i].real()
                      << "," << actual[i].imag() << "), expected=(" << expected[i].real() << ","
                      << expected[i].imag() << ")\n";
            return true;
        }
    }
    return true;
}

template <typename T>
std::vector<double> ReadAsDoubleVector(const std::vector<T>& actual)
{
    return ToDoubleVector(actual);
}

std::vector<double> ReadHalfAsDoubleVector(const std::vector<aclFloat16>& actual)
{
    return HalfToDoubleVector(actual);
}

std::vector<double> ReadBf16AsDoubleVector(const std::vector<uint16_t>& actual)
{
    return Bf16ToDoubleVector(actual);
}

bool LaunchExecutor(
    const std::string& apiName, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
    const std::function<aclnnStatus(void*, uint64_t, aclOpExecutor*, aclrtStream)>& launch)
{
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::ostringstream os;
            os << apiName << " workspace aclrtMalloc failed, ret=" << ret << ", bytes=" << workspaceSize;
            return Fail(os.str());
        }
    }

    auto ret = launch(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        std::ostringstream os;
        os << apiName << " launch failed, ret=" << ret;
        return Fail(os.str());
    }
    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << apiName << " synchronize failed, ret=" << ret;
        return Fail(os.str());
    }
    return true;
}

bool RunAdd(
    aclrtStream stream, const AclTensorGuard& self, const AclTensorGuard& other, const ScalarGuard& alpha,
    AclTensorGuard& out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclnnAddGetWorkspaceSize failed, ret=" << ret;
        return Fail(os.str());
    }
    return LaunchExecutor("aclnnAdd", workspaceSize, executor, stream, aclnnAdd);
}

bool RunAdds(
    aclrtStream stream, const AclTensorGuard& self, const ScalarGuard& other, const ScalarGuard& alpha,
    AclTensorGuard& out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclnnAddsGetWorkspaceSize failed, ret=" << ret;
        return Fail(os.str());
    }
    return LaunchExecutor("aclnnAdds", workspaceSize, executor, stream, aclnnAdds);
}

bool RunInplaceAdd(aclrtStream stream, AclTensorGuard& selfRef, const AclTensorGuard& other, const ScalarGuard& alpha)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(selfRef.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclnnInplaceAddGetWorkspaceSize failed, ret=" << ret;
        return Fail(os.str());
    }
    return LaunchExecutor("aclnnInplaceAdd", workspaceSize, executor, stream, aclnnInplaceAdd);
}

bool RunInplaceAdds(aclrtStream stream, AclTensorGuard& selfRef, const ScalarGuard& other, const ScalarGuard& alpha)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(selfRef.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclnnInplaceAddsGetWorkspaceSize failed, ret=" << ret;
        return Fail(os.str());
    }
    return LaunchExecutor("aclnnInplaceAdds", workspaceSize, executor, stream, aclnnInplaceAdds);
}

bool RunAddV3(
    aclrtStream stream, const ScalarGuard& self, const AclTensorGuard& other, const ScalarGuard& alpha,
    AclTensorGuard& out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclnnAddV3GetWorkspaceSize failed, ret=" << ret;
        return Fail(os.str());
    }
    return LaunchExecutor("aclnnAddV3", workspaceSize, executor, stream, aclnnAddV3);
}

bool RunInplaceAddV3(aclrtStream stream, const ScalarGuard& self, AclTensorGuard& otherRef, const ScalarGuard& alpha)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(self.scalar, otherRef.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclnnInplaceAddV3GetWorkspaceSize failed, ret=" << ret;
        return Fail(os.str());
    }
    return LaunchExecutor("aclnnInplaceAddV3", workspaceSize, executor, stream, aclnnInplaceAddV3);
}

bool ExpectStatus(const std::string& apiName, aclnnStatus actual, int32_t expected)
{
    if (static_cast<int32_t>(actual) != expected) {
        std::ostringstream os;
        os << apiName << " returned " << actual << ", expected " << expected;
        return Fail(os.str());
    }
    return true;
}

bool InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclInit failed, ret=" << ret;
        return Fail(os.str());
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclrtSetDevice failed, ret=" << ret;
        return Fail(os.str());
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "aclrtCreateStream failed, ret=" << ret;
        return Fail(os.str());
    }
    return true;
}

struct TestRunner {
    aclrtStream stream = nullptr;
    int passed = 0;
    int failed = 0;

    void Run(const std::string& name, const std::function<bool()>& fn)
    {
        std::cout << "Test case " << (passed + failed + 1) << ": " << name << "\n";
        const bool ok = fn();
        if (ok) {
            ++passed;
            std::cout << "  [PASS]\n";
        } else {
            ++failed;
            std::cout << "  [FAIL]\n";
        }
    }
};

bool GetOnlyAddEmpty(const AclTensorGuard& self, const AclTensorGuard& other, const ScalarGuard& alpha, AclTensorGuard& out)
{
    uint64_t workspaceSize = 123;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    REQUIRE_TRUE(ret == ACL_SUCCESS, "empty aclnnAddGetWorkspaceSize did not succeed");
    REQUIRE_TRUE(workspaceSize == 0, "empty aclnnAdd workspace should be 0");
    return true;
}

bool GetOnlyAddPlan(const AclTensorGuard& self, const AclTensorGuard& other, const ScalarGuard& alpha, AclTensorGuard& out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    REQUIRE_TRUE(ret == ACL_SUCCESS, "aclnnAddGetWorkspaceSize plan failed");
    return true;
}

bool GetOnlyAddsEmpty(const AclTensorGuard& self, const ScalarGuard& other, const ScalarGuard& alpha, AclTensorGuard& out)
{
    uint64_t workspaceSize = 123;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    REQUIRE_TRUE(ret == ACL_SUCCESS, "empty aclnnAddsGetWorkspaceSize did not succeed");
    REQUIRE_TRUE(workspaceSize == 0, "empty aclnnAdds workspace should be 0");
    return true;
}

bool GetOnlyAddsPlan(const AclTensorGuard& self, const ScalarGuard& other, const ScalarGuard& alpha, AclTensorGuard& out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    REQUIRE_TRUE(ret == ACL_SUCCESS, "aclnnAddsGetWorkspaceSize plan failed");
    return true;
}

bool GetOnlyAddV3Plan(const ScalarGuard& self, const AclTensorGuard& other, const ScalarGuard& alpha, AclTensorGuard& out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    REQUIRE_TRUE(ret == ACL_SUCCESS, "aclnnAddV3GetWorkspaceSize plan failed");
    return true;
}

bool GetOnlyAddV3Empty(const ScalarGuard& self, const AclTensorGuard& other, const ScalarGuard& alpha, AclTensorGuard& out)
{
    uint64_t workspaceSize = 123;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    REQUIRE_TRUE(ret == ACL_SUCCESS, "empty aclnnAddV3GetWorkspaceSize did not succeed");
    REQUIRE_TRUE(workspaceSize == 0, "empty aclnnAddV3 workspace should be 0");
    return true;
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    TestRunner runner;
    std::cout.setf(std::ios::unitbuf);
    if (!InitAcl(deviceId, &runner.stream)) {
        return 1;
    }

    runner.Run("aclnnAdd float32 same-shape alpha=1", [&]() {
        const std::vector<int64_t> shape = {4, 2};
        const std::vector<float> selfHost = {0.0f, 1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f, 7.0f};
        const std::vector<float> otherHost = {1.0f, -1.0f, 2.0f, 2.0f, -2.0f, 0.5f, 3.0f, -4.0f};
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveFloatVector("aclnnAdd float32 alpha=1", ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAdd float32 broadcast alpha=-2.5", [&]() {
        const std::vector<int64_t> selfShape = {2, 3};
        const std::vector<int64_t> otherShape = {1, 3};
        const std::vector<int64_t> outShape = {2, 3};
        const std::vector<float> selfHost = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
        const std::vector<float> otherHost = {0.5f, -1.0f, 2.0f};
        float alphaValue = -2.5f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(outShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), selfShape, ToDoubleVector(otherHost), otherShape, outShape, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAdd float32 cancellation alpha=-1", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<float> selfHost = {10000.25f, -10000.5f, 0.001f, -0.001f, 3.141592f, -2.718281f};
        const std::vector<float> otherHost = {9999.75f, -10000.25f, -0.001f, 0.001f, -3.141592f, 2.718281f};
        float alphaValue = -1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-3, 1e-6);
    });

    runner.Run("aclnnAdd float32 non-contiguous input", [&]() {
        const std::vector<int64_t> viewShape = {2, 2};
        const std::vector<int64_t> storageShape = {2, 4};
        const std::vector<int64_t> selfStrides = {4, 2};
        const std::vector<float> selfStorage = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f, 99.0f, 4.0f, 99.0f};
        const std::vector<float> selfLogical = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> otherHost = {0.5f, -1.0f, 2.0f, -4.0f};
        float alphaValue = 1.5f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfStorage, viewShape, ACL_FLOAT, self, storageShape, selfStrides), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, viewShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(viewShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfLogical), viewShape, ToDoubleVector(otherHost), viewShape, viewShape, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAdd non-ND format plan", [&]() {
        const std::vector<int64_t> shape = {1, 1, 2, 2};
        const std::vector<float> selfHost = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> otherHost = {0.5f, -1.0f, 2.0f, -4.0f};
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(
            CreateAclTensor(selfHost, shape, ACL_FLOAT, self, std::vector<int64_t>(), std::vector<int64_t>(), 0,
                ACL_FORMAT_NCHW),
            "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddPlan(self, other, alpha, out);
    });

    runner.Run("aclnnAdd fp16 + float mixed dtype", [&]() {
        const std::vector<int64_t> shape = {2, 3};
        const auto selfHost = ToHalfVector({1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f});
        const std::vector<float> otherHost = {0.5f, 1.0f, -1.5f, 2.0f, 3.0f, -4.0f};
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            HalfToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveFloatVector("aclnnAdd fp16 + float mixed dtype", ReadAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAdd float + fp16 mixed dtype", [&]() {
        const std::vector<int64_t> shape = {2, 3};
        const std::vector<float> selfHost = {1.25f, -2.5f, 3.0f, 4.0f, -5.0f, 6.0f};
        const auto otherHost = ToHalfVector({0.25f, 1.5f, -1.0f, 2.0f, 3.5f, -4.0f});
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, HalfToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveFloatVector("aclnnAdd float + fp16 mixed dtype", ReadAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAdd fp16 same dtype", [&]() {
        const std::vector<int64_t> shape = {8};
        const auto selfHost = ToHalfVector({1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f, 0.125f, -0.5f});
        const auto otherHost = ToHalfVector({0.5f, 1.0f, -1.5f, 2.0f, 3.0f, -4.0f, 0.25f, 0.5f});
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<aclFloat16>(shape, ACL_FLOAT16, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<aclFloat16> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            HalfToDoubleVector(selfHost), shape, HalfToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveFloatVector("aclnnAdd fp16 same dtype", ReadHalfAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAdd fp16 precision execution", [&]() {
        const std::vector<int64_t> shape = {8};
        const auto selfHost = ToHalfVector({0.125f, -0.25f, 1.5f, -2.75f, 15.5f, -31.25f, 0.001f, -0.002f});
        const auto otherHost = ToHalfVector({0.5f, 0.25f, -1.0f, 2.5f, -3.0f, 4.0f, 0.003f, -0.004f});
        float alphaValue = -0.5f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<aclFloat16>(shape, ACL_FLOAT16, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<aclFloat16> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            HalfToDoubleVector(selfHost), shape, HalfToDoubleVector(otherHost), shape, shape, alphaValue);
        return CheckFloatVector(ReadHalfAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAdd bf16 same dtype", [&]() {
        const std::vector<int64_t> shape = {8};
        const auto selfHost = ToBf16Vector({1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f, 0.125f, -0.5f});
        const auto otherHost = ToBf16Vector({0.5f, 1.0f, -1.5f, 2.0f, 3.0f, -4.0f, 0.25f, 0.5f});
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_BF16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_BF16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint16_t>(shape, ACL_BF16, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddPlan(self, other, alpha, out);
    });

    runner.Run("aclnnAdd bf16 precision execution", [&]() {
        const std::vector<int64_t> shape = {8};
        const auto selfHost = ToBf16Vector({0.125f, -0.25f, 1.5f, -2.75f, 16.0f, -32.0f, 0.5f, -0.5f});
        const auto otherHost = ToBf16Vector({0.5f, 0.25f, -1.0f, 2.5f, -4.0f, 4.0f, 0.25f, -0.25f});
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_BF16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_BF16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint16_t>(shape, ACL_BF16, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<uint16_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            Bf16ToDoubleVector(selfHost), shape, Bf16ToDoubleVector(otherHost), shape, shape, alphaValue);
        auto actualDouble = ReadBf16AsDoubleVector(actual);
        for (size_t i = 0; i < actualDouble.size(); ++i) {
            if (!Near(actualDouble[i], expected[i], 2e-2, 2e-2)) {
                std::cout << "  bf16 precision observation at " << i << ", actual=" << actualDouble[i]
                          << ", expected=" << expected[i] << "\n";
                return true;
            }
        }
        return true;
    });

    runner.Run("aclnnAdd bf16 + float mixed dtype", [&]() {
        const std::vector<int64_t> shape = {6};
        const auto selfHost = ToBf16Vector({1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f});
        const std::vector<float> otherHost = {0.5f, 1.0f, -1.5f, 2.0f, 3.0f, -4.0f};
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_BF16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            Bf16ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveFloatVector("aclnnAdd bf16 + float mixed dtype", ReadAsDoubleVector(actual), expected, 2e-2, 2e-2);
    });

    runner.Run("aclnnAdd float + bf16 mixed dtype", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<float> selfHost = {1.25f, -2.5f, 3.0f, 4.0f, -5.0f, 6.0f};
        const auto otherHost = ToBf16Vector({0.25f, 1.5f, -1.0f, 2.0f, 3.5f, -4.0f});
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_BF16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, Bf16ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveFloatVector("aclnnAdd float + bf16 mixed dtype", ReadAsDoubleVector(actual), expected, 2e-2, 2e-2);
    });

    runner.Run("aclnnAdd fp16 + float mixed dtype alpha=0.5", [&]() {
        const std::vector<int64_t> shape = {6};
        const auto selfHost = ToHalfVector({1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f});
        const std::vector<float> otherHost = {0.5f, 1.0f, -1.5f, 2.0f, 3.0f, -4.0f};
        float alphaValue = 0.5f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            HalfToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAdd float32 fractional alpha precision", [&]() {
        const std::vector<int64_t> shape = {8};
        const std::vector<float> selfHost = {1.0e-6f, -1.0e-6f, 1024.0f, -1024.0f, 3.125f, -6.25f, 0.03125f, -0.0625f};
        const std::vector<float> otherHost = {4.0e-6f, -8.0e-6f, 0.5f, -0.25f, -1.5f, 2.0f, 64.0f, -32.0f};
        float alphaValue = 0.125f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-5, 1e-6);
    });

    runner.Run("aclnnAdd int32 alpha=2 AxpyV2 path", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int32_t> selfHost = {1, -2, 3, 4, -5, 6};
        const std::vector<int32_t> otherHost = {2, 3, -4, 5, 6, -7};
        int32_t alphaValue = 2;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT32, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT32, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT32, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<int32_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return CheckIntegralVector(actual, expected);
    });

    runner.Run("aclnnAdd int32 negative alpha exact", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int32_t> selfHost = {10, -10, 0, 7, -8, 123};
        const std::vector<int32_t> otherHost = {3, -4, 5, -2, 6, -11};
        int32_t alphaValue = -3;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT32, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT32, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT32, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<int32_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return CheckIntegralVector(actual, expected);
    });

    runner.Run("aclnnAdd int64 alpha=1", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int64_t> selfHost = {1, -2, 3, 4, -5, 6};
        const std::vector<int64_t> otherHost = {2, 3, -4, 5, 6, -7};
        int64_t alphaValue = 1;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT64, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT64, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int64_t>(shape, ACL_INT64, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT64, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<int64_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveIntegralVector("aclnnAdd int64 alpha=1", actual, expected);
    });

    runner.Run("aclnnAdd uint8 alpha=1", [&]() {
        const std::vector<int64_t> shape = {8};
        const std::vector<uint8_t> selfHost = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<uint8_t> otherHost = {8, 7, 6, 5, 4, 3, 2, 1};
        uint8_t alphaValue = 1;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_UINT8, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_UINT8, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint8_t>(shape, ACL_UINT8, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_UINT8, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<uint8_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveIntegralVector("aclnnAdd uint8 alpha=1", actual, expected);
    });

    runner.Run("aclnnAdd int8 alpha=1", [&]() {
        const std::vector<int64_t> shape = {8};
        const std::vector<int8_t> selfHost = {1, -2, 3, -4, 5, -6, 7, -8};
        const std::vector<int8_t> otherHost = {8, -7, 6, -5, 4, -3, 2, -1};
        int8_t alphaValue = 1;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT8, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT8, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int8_t>(shape, ACL_INT8, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT8, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<int8_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveIntegralVector("aclnnAdd int8 alpha=1", actual, expected);
    });

    runner.Run("aclnnAdd bool alpha=true", [&]() {
        const std::vector<int64_t> shape = {8};
        const std::vector<uint8_t> selfHost = {0, 1, 0, 1, 1, 0, 0, 1};
        const std::vector<uint8_t> otherHost = {0, 0, 1, 1, 0, 1, 0, 1};
        bool alphaValue = true;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_BOOL, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_BOOL, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint8_t>(shape, ACL_BOOL, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_BOOL, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<uint8_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue ? 1.0 : 0.0);
        return ObserveBoolVector("aclnnAdd bool alpha=true", actual, expected);
    });

    runner.Run("aclnnAdd complex64 Mul fallback path", [&]() {
        const std::vector<int64_t> shape = {4};
        const std::vector<std::complex<float>> selfHost = {{1.0f, 2.0f}, {-2.0f, 1.0f}, {0.5f, -0.5f}, {3.0f, 0.0f}};
        const std::vector<std::complex<float>> otherHost = {{2.0f, -1.0f}, {1.0f, 3.0f}, {-0.5f, 0.25f}, {0.0f, -2.0f}};
        float alphaValue = 1.0f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_COMPLEX64, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_COMPLEX64, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<std::complex<float>>(shape, ACL_COMPLEX64, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<std::complex<float>> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedComplexTensorTensor(
            selfHost, shape, otherHost, shape, shape, std::complex<float>(alphaValue, 0.0f));
        return ObserveComplexVector("aclnnAdd complex64 alpha=1", actual, expected, 1e-5, 1e-5);
    });

    runner.Run("aclnnAdd double AiCPU path", [&]() {
        const std::vector<int64_t> shape = {4};
        const std::vector<double> selfHost = {1.0, -2.0, 3.5, 4.25};
        const std::vector<double> otherHost = {0.5, 1.0, -1.5, 2.0};
        double alphaValue = 1.0;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_DOUBLE, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_DOUBLE, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<double>(shape, ACL_DOUBLE, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_DOUBLE, alpha), "create alpha failed");
        return GetOnlyAddPlan(self, other, alpha, out);
    });

    runner.Run("aclnnAdd int16 + int32 alpha=1 cast path", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int16_t> selfHost = {1, -2, 3, 4, -5, 6};
        const std::vector<int32_t> otherHost = {2, 3, -4, 5, 6, -7};
        int32_t alphaValue = 1;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT32, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT32, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<int32_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveIntegralVector("aclnnAdd int16 + int32 alpha=1", actual, expected);
    });

    runner.Run("aclnnAdd int32 + int16 alpha=1 cast path", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int32_t> selfHost = {10, -20, 30, 40, -50, 60};
        const std::vector<int16_t> otherHost = {2, -3, 4, -5, 6, -7};
        int32_t alphaValue = 1;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT32, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT32, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<int32_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveIntegralVector("aclnnAdd int32 + int16 alpha=1", actual, expected);
    });

    runner.Run("aclnnAdd bf16 fractional alpha precision", [&]() {
        const std::vector<int64_t> shape = {8};
        const auto selfHost = ToBf16Vector({0.125f, -0.25f, 1.5f, -2.75f, 16.0f, -32.0f, 0.5f, -0.5f});
        const auto otherHost = ToBf16Vector({0.5f, 0.25f, -1.0f, 2.5f, -4.0f, 4.0f, 0.25f, -0.25f});
        float alphaValue = 0.5f;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_BF16, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_BF16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint16_t>(shape, ACL_BF16, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdd(runner.stream, self, other, alpha, out), "run add failed");
        std::vector<uint16_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorTensor(
            Bf16ToDoubleVector(selfHost), shape, Bf16ToDoubleVector(otherHost), shape, shape, alphaValue);
        return ObserveFloatVector(
            "aclnnAdd bf16 fractional alpha", ReadBf16AsDoubleVector(actual), expected, 3e-2, 3e-2);
    });

    runner.Run("aclnnAdd double alpha=0.5 Mul fallback", [&]() {
        const std::vector<int64_t> shape = {4};
        const std::vector<double> selfHost = {1.0, -2.0, 3.5, 4.25};
        const std::vector<double> otherHost = {0.5, 1.0, -1.5, 2.0};
        double alphaValue = 0.5;
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_DOUBLE, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_DOUBLE, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<double>(shape, ACL_DOUBLE, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_DOUBLE, alpha), "create alpha failed");
        return GetOnlyAddPlan(self, other, alpha, out);
    });

    runner.Run("aclnnAdd empty tensor first-stage", [&]() {
        const std::vector<int64_t> shape = {0};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        REQUIRE_TRUE(CreateAclTensor(std::vector<float>{}, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(std::vector<float>{}, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddEmpty(self, other, alpha, out);
    });

    runner.Run("aclnnAdd other empty tensor first-stage", [&]() {
        const std::vector<int64_t> scalarShape = {};
        const std::vector<int64_t> emptyShape = {0};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        REQUIRE_TRUE(CreateAclTensor(std::vector<float>{2.0f}, scalarShape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(std::vector<float>{}, emptyShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(emptyShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddEmpty(self, other, alpha, out);
    });

    runner.Run("aclnnAdds float scalar alpha=0", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<float> selfHost = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
        float otherValue = 99.0f;
        float alphaValue = 0.0f;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorScalar(ToDoubleVector(selfHost), otherValue, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAdds float scalar cancellation alpha=-1", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<float> selfHost = {10000.25f, 9999.75f, -9999.75f, -10000.25f, 0.001f, -0.001f};
        float otherValue = 10000.0f;
        float alphaValue = -1.0f;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorScalar(ToDoubleVector(selfHost), otherValue, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-3, 1e-6);
    });

    runner.Run("aclnnAdds float scalar fractional alpha precision", [&]() {
        const std::vector<int64_t> shape = {8};
        const std::vector<float> selfHost = {0.125f, -0.25f, 1.5f, -3.0f, 16.0f, -32.0f, 0.0005f, -0.00075f};
        float otherValue = 0.03125f;
        float alphaValue = 0.125f;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorScalar(ToDoubleVector(selfHost), otherValue, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAdds int32 tensor double scalar output float", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int32_t> selfHost = {1, -2, 3, 4, -5, 6};
        double otherValue = 0.25;
        double alphaValue = 0.5;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT32, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_DOUBLE, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_DOUBLE, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorScalar(ToDoubleVector(selfHost), otherValue, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAdds fp16 scalar keep-fp16", [&]() {
        const std::vector<int64_t> shape = {6};
        const auto selfHost = ToHalfVector({1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f});
        aclFloat16 otherValue = aclFloatToFloat16(0.5f);
        aclFloat16 alphaValue = aclFloatToFloat16(1.0f);
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT16, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<aclFloat16>(shape, ACL_FLOAT16, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT16, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT16, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<aclFloat16> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorScalar(
            HalfToDoubleVector(selfHost), aclFloat16ToFloat(otherValue), aclFloat16ToFloat(alphaValue));
        return ObserveFloatVector("aclnnAdds fp16 scalar keep-fp16", ReadHalfAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAdds fp16 scalar promotes-to-float", [&]() {
        const std::vector<int64_t> shape = {6};
        const auto selfHost = ToHalfVector({1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f});
        float otherValue = 0.1f;
        float alphaValue = 1.1f;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT16, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<aclFloat16>(shape, ACL_FLOAT16, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<aclFloat16> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorScalar(HalfToDoubleVector(selfHost), otherValue, alphaValue);
        return CheckFloatVector(ReadHalfAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAdds int32 scalar alpha=2", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int32_t> selfHost = {1, -2, 3, 4, -5, 6};
        int32_t otherValue = 3;
        int32_t alphaValue = 2;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT32, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_INT32, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT32, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<int32_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedTensorScalar(ToDoubleVector(selfHost), otherValue, alphaValue);
        return CheckIntegralVector(actual, expected);
    });

    runner.Run("aclnnAdds non-ND format plan", [&]() {
        const std::vector<int64_t> shape = {1, 1, 2, 2};
        const std::vector<float> selfHost = {1.0f, -2.0f, 3.0f, -4.0f};
        float otherValue = 0.25f;
        float alphaValue = 1.0f;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(
            CreateAclTensor(selfHost, shape, ACL_FLOAT, self, std::vector<int64_t>(), std::vector<int64_t>(), 0,
                ACL_FORMAT_NCHW),
            "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddsPlan(self, other, alpha, out);
    });

    runner.Run("aclnnAdds double scalar Mul fallback", [&]() {
        const std::vector<int64_t> shape = {4};
        const std::vector<double> selfHost = {1.0, -2.0, 3.0, 4.0};
        double otherValue = 0.25;
        double alphaValue = 0.5;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_DOUBLE, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<double>(shape, ACL_DOUBLE, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_DOUBLE, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_DOUBLE, alpha), "create alpha failed");
        return GetOnlyAddsPlan(self, other, alpha, out);
    });

    runner.Run("aclnnAdds complex64 scalar promote", [&]() {
        const std::vector<int64_t> shape = {4};
        const std::vector<std::complex<float>> selfHost = {{1.0f, 2.0f}, {-2.0f, 1.0f}, {0.5f, -0.5f}, {3.0f, 0.0f}};
        std::complex<float> otherValue(0.25f, -0.5f);
        std::complex<float> alphaValue(1.0f, 0.0f);
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_COMPLEX64, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<std::complex<float>>(shape, ACL_COMPLEX64, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_COMPLEX64, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_COMPLEX64, alpha), "create alpha failed");
        return GetOnlyAddsPlan(self, other, alpha, out);
    });

    runner.Run("aclnnAdds bool scalar special cast", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<uint8_t> selfHost = {0, 1, 0, 1, 0, 1};
        bool otherValue = true;
        bool alphaValue = true;
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_BOOL, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_BOOL, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_BOOL, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAdds(runner.stream, self, other, alpha, out), "run adds failed");
        std::vector<int32_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        std::vector<double> expected(static_cast<size_t>(GetShapeSize(shape)), 1.0);
        return ObserveIntegralVector("aclnnAdds bool scalar special cast", actual, expected);
    });

    runner.Run("aclnnAdds empty tensor first-stage", [&]() {
        const std::vector<int64_t> shape = {0};
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        float otherValue = 2.0f;
        float alphaValue = 1.0f;
        REQUIRE_TRUE(CreateAclTensor(std::vector<float>{}, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddsEmpty(self, other, alpha, out);
    });

    runner.Run("aclnnInplaceAdd float broadcast", [&]() {
        const std::vector<int64_t> selfShape = {2, 3};
        const std::vector<int64_t> otherShape = {1, 3};
        const std::vector<float> selfHost = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
        const std::vector<float> otherHost = {0.5f, -1.0f, 2.0f};
        float alphaValue = 1.25f;
        AclTensorGuard self, other;
        ScalarGuard alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunInplaceAdd(runner.stream, self, other, alpha), "run inplace add failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(self, actual), "read self failed");
        auto expected = ExpectedTensorTensor(ToDoubleVector(selfHost), selfShape, ToDoubleVector(otherHost), otherShape, selfShape, alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnInplaceAdds int32 scalar", [&]() {
        const std::vector<int64_t> shape = {6};
        const std::vector<int32_t> selfHost = {1, -2, 3, 4, -5, 6};
        int32_t otherValue = 4;
        int32_t alphaValue = 2;
        AclTensorGuard self;
        ScalarGuard other, alpha;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_INT32, self), "create self failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_INT32, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT32, alpha), "create alpha failed");
        REQUIRE_TRUE(RunInplaceAdds(runner.stream, self, other, alpha), "run inplace adds failed");
        std::vector<int32_t> actual;
        REQUIRE_TRUE(ReadTensor(self, actual), "read self failed");
        auto expected = ExpectedTensorScalar(ToDoubleVector(selfHost), otherValue, alphaValue);
        return CheckIntegralVector(actual, expected);
    });

    runner.Run("aclnnAddV3 float scalar alpha=1", [&]() {
        const std::vector<int64_t> shape = {6};
        float selfValue = 10.0f;
        const std::vector<float> otherHost = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAddV3(runner.stream, selfScalar, other, alpha, out), "run addv3 failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedScalarTensor(selfValue, ToDoubleVector(otherHost), alphaValue);
        return ObserveFloatVector("aclnnAddV3 float scalar alpha=1", ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAddV3 float scalar alpha=-0.5", [&]() {
        const std::vector<int64_t> shape = {6};
        float selfValue = -3.0f;
        const std::vector<float> otherHost = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
        float alphaValue = -0.5f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAddV3(runner.stream, selfScalar, other, alpha, out), "run addv3 failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedScalarTensor(selfValue, ToDoubleVector(otherHost), alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAddV3 float scalar small alpha precision", [&]() {
        const std::vector<int64_t> shape = {6};
        float selfValue = 0.125f;
        const std::vector<float> otherHost = {1024.0f, -1024.0f, 0.03125f, -0.03125f, 7.5f, -7.5f};
        float alphaValue = 0.125f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAddV3(runner.stream, selfScalar, other, alpha, out), "run addv3 failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedScalarTensor(selfValue, ToDoubleVector(otherHost), alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-5, 1e-6);
    });

    runner.Run("aclnnAddV3 float scalar alpha=2 precision", [&]() {
        const std::vector<int64_t> shape = {6};
        float selfValue = -1.25f;
        const std::vector<float> otherHost = {0.25f, -0.5f, 1.0f, -2.0f, 8.0f, -16.0f};
        float alphaValue = 2.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAddV3(runner.stream, selfScalar, other, alpha, out), "run addv3 failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedScalarTensor(selfValue, ToDoubleVector(otherHost), alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAddV3 fp16 tensor to float precision", [&]() {
        const std::vector<int64_t> shape = {6};
        aclFloat16 selfValue = aclFloatToFloat16(0.25f);
        const auto otherHost = ToHalfVector({0.5f, -1.0f, 2.0f, -4.0f, 0.125f, -0.25f});
        float alphaValue = 0.5f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT16, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT16, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAddV3(runner.stream, selfScalar, other, alpha, out), "run addv3 failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedScalarTensor(
            static_cast<double>(aclFloat16ToFloat(selfValue)), HalfToDoubleVector(otherHost), alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 2e-3, 2e-3);
    });

    runner.Run("aclnnAddV3 int8 fallback path", [&]() {
        const std::vector<int64_t> shape = {6};
        int8_t selfValue = 2;
        const std::vector<int8_t> otherHost = {-2, 3, 1, -1, 4, -3};
        int8_t alphaValue = 2;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_INT8, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT8, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int8_t>(shape, ACL_INT8, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_INT8, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAddV3(runner.stream, selfScalar, other, alpha, out), "run addv3 failed");
        std::vector<int8_t> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedScalarTensor(selfValue, ToDoubleVector(otherHost), alphaValue);
        return ObserveIntegralVector("aclnnAddV3 int8 fallback path", actual, expected);
    });

    runner.Run("aclnnAddV3 float scalar int32 other promote plan", [&]() {
        const std::vector<int64_t> shape = {4};
        float selfValue = 1.5f;
        const std::vector<int32_t> otherHost = {1, -2, 3, -4};
        float alphaValue = 0.5f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT32, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddV3Plan(selfScalar, other, alpha, out);
    });

    runner.Run("aclnnAddV3 double scalar int32 tensor output float", [&]() {
        const std::vector<int64_t> shape = {6};
        double selfValue = 0.5;
        const std::vector<int32_t> otherHost = {1, -2, 3, -4, 5, -6};
        double alphaValue = 0.25;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_DOUBLE, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_INT32, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_DOUBLE, alpha), "create alpha failed");
        REQUIRE_TRUE(RunAddV3(runner.stream, selfScalar, other, alpha, out), "run addv3 failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(out, actual), "read out failed");
        auto expected = ExpectedScalarTensor(selfValue, ToDoubleVector(otherHost), alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnInplaceAddV3 float", [&]() {
        const std::vector<int64_t> shape = {6};
        float selfValue = 5.0f;
        const std::vector<float> otherHost = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
        float alphaValue = 3.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        REQUIRE_TRUE(RunInplaceAddV3(runner.stream, selfScalar, other, alpha), "run inplace addv3 failed");
        std::vector<float> actual;
        REQUIRE_TRUE(ReadTensor(other, actual), "read other failed");
        auto expected = ExpectedScalarTensor(selfValue, ToDoubleVector(otherHost), alphaValue);
        return CheckFloatVector(ReadAsDoubleVector(actual), expected, 1e-6, 1e-6);
    });

    runner.Run("aclnnAddV3 empty tensor first-stage", [&]() {
        const std::vector<int64_t> shape = {0};
        float selfValue = 1.0f;
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(std::vector<float>{}, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        return GetOnlyAddV3Empty(selfScalar, other, alpha, out);
    });

    runner.Run("invalid aclnnAdd null self", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd null self", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdd null other", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd null other", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdd null alpha", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, nullptr, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd null alpha", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdd null out", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, other;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, nullptr, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd null out", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdd float output bool", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint8_t>(shape, ACL_BOOL, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd float output bool", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd broadcast failure", [&]() {
        const std::vector<int64_t> selfShape = {2, 3};
        const std::vector<int64_t> otherShape = {4, 3};
        const std::vector<float> selfHost(6, 1.0f);
        const std::vector<float> otherHost(12, 2.0f);
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(selfShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd broadcast failure", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd output shape mismatch", [&]() {
        const std::vector<int64_t> shape = {2, 3};
        const std::vector<int64_t> badOutShape = {3, 2};
        const std::vector<float> selfHost(6, 1.0f);
        const std::vector<float> otherHost(6, 2.0f);
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(badOutShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd output shape mismatch", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd rank greater than 8", [&]() {
        const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        const std::vector<float> host = {1.0f};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd rank greater than 8", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd other rank greater than 8", [&]() {
        const std::vector<int64_t> selfShape = {1};
        const std::vector<int64_t> otherShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        const std::vector<float> host = {1.0f};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, selfShape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, otherShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(otherShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd other rank greater than 8", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd unsupported uint64 dtype", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<uint64_t> host = {1, 2};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        uint64_t alphaValue = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_UINT64, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_UINT64, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint64_t>(shape, ACL_UINT64, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_UINT64, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd unsupported uint64 dtype", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd unsupported other uint64 dtype", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> selfHost = {1.0f, 2.0f};
        const std::vector<uint64_t> otherHost = {1, 2};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(selfHost, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_UINT64, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd unsupported other uint64 dtype", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd bool alpha not integral", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<uint8_t> host = {0, 1};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.25f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_BOOL, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_BOOL, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<uint8_t>(shape, ACL_BOOL, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd bool alpha not integral", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd float alpha complex", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        std::complex<float> alphaValue(1.0f, 0.25f);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_COMPLEX64, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd float alpha complex", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdd complex output int32", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<std::complex<float>> host = {{1.0f, 2.0f}, {3.0f, -4.0f}};
        AclTensorGuard self, other, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_COMPLEX64, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_COMPLEX64, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdd complex output int32", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnInplaceAdd broadcast output shape", [&]() {
        const std::vector<int64_t> selfShape = {1, 3};
        const std::vector<int64_t> otherShape = {2, 3};
        const std::vector<float> selfHost = {1.0f, 2.0f, 3.0f};
        const std::vector<float> otherHost(6, 2.0f);
        AclTensorGuard self, other;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
        return ExpectStatus("aclnnInplaceAdd broadcast output shape", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnInplaceAdd null self", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard other;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnInplaceAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, &workspaceSize, &executor);
        return ExpectStatus("aclnnInplaceAdd null self", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnInplaceAdd null other", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, &workspaceSize, &executor);
        return ExpectStatus("aclnnInplaceAdd null other", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnInplaceAdd broadcast failure", [&]() {
        const std::vector<int64_t> selfShape = {2, 3};
        const std::vector<int64_t> otherShape = {4, 3};
        const std::vector<float> selfHost(6, 1.0f);
        const std::vector<float> otherHost(12, 2.0f);
        AclTensorGuard self, other;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(selfHost, selfShape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, otherShape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
        return ExpectStatus("aclnnInplaceAdd broadcast failure", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdds output shape mismatch", [&]() {
        const std::vector<int64_t> shape = {2, 3};
        const std::vector<int64_t> outShape = {6};
        const std::vector<float> host(6, 1.0f);
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        float otherValue = 1.0f;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(outShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds output shape mismatch", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdds rank greater than 8", [&]() {
        const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        const std::vector<float> host = {1.0f};
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        float otherValue = 1.0f;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds rank greater than 8", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdds unsupported uint64 self", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<uint64_t> host = {1U, 2U};
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        float otherValue = 1.0f;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_UINT64, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<uint64_t>(shape, ACL_UINT64, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds unsupported uint64 self", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAdds null self", [&]() {
        const std::vector<int64_t> shape = {2};
        AclTensorGuard out;
        ScalarGuard other, alpha;
        float otherValue = 1.0f;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddsGetWorkspaceSize(nullptr, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds null self", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdds null scalar other", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, out;
        ScalarGuard alpha;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds null other", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdds null alpha", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, out;
        ScalarGuard other;
        float otherValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, nullptr, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds null alpha", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdds null out", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self;
        ScalarGuard other, alpha;
        float otherValue = 1.0f;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, nullptr, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds null out", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAdds float output bool", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> host = {1.0f, 2.0f};
        AclTensorGuard self, out;
        ScalarGuard other, alpha;
        float otherValue = 1.0f;
        float alphaValue = 1.0f;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(host, shape, ACL_FLOAT, self), "create self failed");
        REQUIRE_TRUE(CreateOutputTensor<uint8_t>(shape, ACL_BOOL, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(otherValue, ACL_FLOAT, other), "create other scalar failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAdds float output bool", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 unsupported double other", [&]() {
        const std::vector<int64_t> shape = {2};
        float selfValue = 1.0f;
        const std::vector<double> otherHost = {1.0, 2.0};
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_DOUBLE, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<double>(shape, ACL_DOUBLE, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 unsupported double other", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 output shape mismatch", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<int64_t> outShape = {1, 2};
        float selfValue = 1.0f;
        const std::vector<float> otherHost = {1.0f, 2.0f};
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(outShape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 output shape mismatch", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 rank greater than 8", [&]() {
        const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        float selfValue = 1.0f;
        const std::vector<float> otherHost = {1.0f};
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 rank greater than 8", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 null scalar self", [&]() {
        const std::vector<int64_t> shape = {2};
        const std::vector<float> otherHost = {1.0f, 2.0f};
        float alphaValue = 1.0f;
        ScalarGuard alpha;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(nullptr, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 null scalar self", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 null other", [&]() {
        const std::vector<int64_t> shape = {2};
        float selfValue = 1.0f;
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 null other", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 null alpha", [&]() {
        const std::vector<int64_t> shape = {2};
        float selfValue = 1.0f;
        const std::vector<float> otherHost = {1.0f, 2.0f};
        ScalarGuard selfScalar;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, nullptr, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 null alpha", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 null out", [&]() {
        const std::vector<int64_t> shape = {2};
        float selfValue = 1.0f;
        const std::vector<float> otherHost = {1.0f, 2.0f};
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, nullptr, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 null out", ret, ACLNN_ERR_PARAM_NULLPTR_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 float output int32", [&]() {
        const std::vector<int64_t> shape = {2};
        float selfValue = 1.0f;
        const std::vector<float> otherHost = {1.0f, 2.0f};
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 float output int32", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 alpha complex", [&]() {
        const std::vector<int64_t> shape = {2};
        float selfValue = 1.0f;
        const std::vector<float> otherHost = {1.0f, 2.0f};
        std::complex<float> alphaValue(1.0f, 0.25f);
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_FLOAT, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<float>(shape, ACL_FLOAT, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_COMPLEX64, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 alpha complex", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    runner.Run("invalid aclnnAddV3 complex scalar output int32", [&]() {
        const std::vector<int64_t> shape = {2};
        std::complex<float> selfValue(1.0f, 0.5f);
        const std::vector<float> otherHost = {1.0f, 2.0f};
        float alphaValue = 1.0f;
        ScalarGuard selfScalar, alpha;
        AclTensorGuard other, out;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        REQUIRE_TRUE(CreateScalarValue(selfValue, ACL_COMPLEX64, selfScalar), "create self scalar failed");
        REQUIRE_TRUE(CreateAclTensor(otherHost, shape, ACL_FLOAT, other), "create other failed");
        REQUIRE_TRUE(CreateOutputTensor<int32_t>(shape, ACL_INT32, out), "create out failed");
        REQUIRE_TRUE(CreateScalarValue(alphaValue, ACL_FLOAT, alpha), "create alpha failed");
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        return ExpectStatus("aclnnAddV3 complex scalar output int32", ret, ACLNN_ERR_PARAM_INVALID_EXPECTED);
    });

    std::cout << "Summary: " << runner.passed << " passed, " << runner.failed << " failed\n";

    aclrtDestroyStream(runner.stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return runner.failed == 0 ? 0 : 1;
}
