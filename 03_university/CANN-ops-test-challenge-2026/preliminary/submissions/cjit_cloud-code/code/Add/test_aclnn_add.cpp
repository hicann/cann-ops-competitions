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
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "../op_api/aclnn_add.h"
#include "../op_api/aclnn_add_v3.h"

namespace l0op {
const aclTensor* Add(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
const aclTensor* AddInplace(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
}

#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif

#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
#endif

#ifndef ACLNN_ERR_INNER_NULLPTR
#define ACLNN_ERR_INNER_NULLPTR 561103
#endif

namespace {

struct TensorSpec {
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_ND;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
    std::vector<std::complex<double>> logicalValues;
};

struct ScalarSpec {
    aclDataType dtype = ACL_DT_UNDEFINED;
    std::vector<uint8_t> bytes;
    std::complex<double> logicalValue = {0.0, 0.0};
};

struct RuntimeTensor {
    aclTensor* tensor = nullptr;
    void* deviceAddr = nullptr;
    size_t byteSize = 0;

    ~RuntimeTensor()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

struct RuntimeScalar {
    aclScalar* scalar = nullptr;

    ~RuntimeScalar()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
        }
    }
};

struct WorkspaceBuffer {
    void* addr = nullptr;

    ~WorkspaceBuffer()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }
};

class Reporter {
public:
    void Record(const std::string& name, bool pass, const std::string& detail = "")
    {
        std::cout << "[" << (pass ? "PASS" : "FAIL") << "] " << name;
        if (!detail.empty()) {
            std::cout << " - " << detail;
        }
        std::cout << std::endl;
        if (pass) {
            ++passed_;
        } else {
            ++failed_;
        }
    }

    int Failed() const
    {
        return failed_;
    }

    void PrintSummary() const
    {
        std::cout << "Summary: " << passed_ << " passed, " << failed_ << " failed" << std::endl;
    }

private:
    int passed_ = 0;
    int failed_ = 0;
};

bool ShouldRunCase(const std::string& name)
{
    const char* filter = std::getenv("ADD_CASE_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return name.find(filter) != std::string::npos;
}

bool EnableUnstableNullCases()
{
    const char* value = std::getenv("ADD_ENABLE_UNSTABLE_NULL_CASES");
    return value != nullptr && std::string(value) == "1";
}

bool IsSimulatorRuntime()
{
    const char* value = std::getenv("LD_LIBRARY_PATH");
    return value != nullptr && std::string(value).find("/simulator/") != std::string::npos;
}

aclnnStatus ExpectedSuccessOrSimulatorNullptr()
{
    return IsSimulatorRuntime() ? ACLNN_ERR_INNER_NULLPTR : ACL_SUCCESS;
}

bool IsSimulatorInnerNullStatus(const aclnnStatus status)
{
    return status == ACLNN_ERR_INNER_NULLPTR && IsSimulatorRuntime();
}

class AclEnv {
public:
    bool Init(std::string* error)
    {
        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            *error = "aclInit failed: " + std::to_string(ret);
            return false;
        }
        initialized_ = true;

        ret = aclrtSetDevice(deviceId_);
        if (ret != ACL_SUCCESS) {
            *error = "aclrtSetDevice failed: " + std::to_string(ret);
            return false;
        }
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            *error = "aclrtCreateStream failed: " + std::to_string(ret);
            return false;
        }
        return true;
    }

    ~AclEnv()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (initialized_) {
            aclFinalize();
        }
    }

    aclrtStream Stream() const
    {
        return stream_;
    }

private:
    int32_t deviceId_ = 0;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

int64_t ElementCount(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

size_t DTypeSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_FLOAT16:
            return sizeof(aclFloat16);
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_DOUBLE:
            return sizeof(double);
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_UINT8:
            return sizeof(uint8_t);
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_UINT16:
            return sizeof(uint16_t);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_UINT32:
            return sizeof(uint32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_BOOL:
            return sizeof(uint8_t);
        case ACL_COMPLEX64:
            return sizeof(std::complex<float>);
        case ACL_COMPLEX128:
            return sizeof(std::complex<double>);
        default:
            return 0;
    }
}

template <typename T>
std::vector<uint8_t> PackVector(const std::vector<T>& values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    if (!values.empty()) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

template <typename T>
std::vector<uint8_t> PackScalar(const T& value)
{
    std::vector<uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16U) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16U);
}

float BFloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

TensorSpec MakeOutputSpec(const std::vector<int64_t>& shape, aclDataType dtype, aclFormat format = ACL_FORMAT_ND)
{
    TensorSpec spec;
    spec.dtype = dtype;
    spec.format = format;
    spec.shape = shape;
    spec.bytes.assign(static_cast<size_t>(ElementCount(shape)) * DTypeSize(dtype), 0);
    spec.logicalValues.assign(static_cast<size_t>(ElementCount(shape)), std::complex<double>(0.0, 0.0));
    return spec;
}

TensorSpec MakeRealTensor(
    const std::vector<int64_t>& shape, aclDataType dtype, const std::vector<double>& values,
    aclFormat format = ACL_FORMAT_ND)
{
    TensorSpec spec;
    spec.dtype = dtype;
    spec.format = format;
    spec.shape = shape;
    spec.logicalValues.reserve(values.size());
    for (double value : values) {
        spec.logicalValues.emplace_back(value, 0.0);
    }

    switch (dtype) {
        case ACL_FLOAT: {
            std::vector<float> encoded(values.size(), 0.0f);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<float>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_FLOAT16: {
            std::vector<aclFloat16> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = aclFloatToFloat16(static_cast<float>(values[i]));
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_BF16: {
            std::vector<uint16_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = FloatToBFloat16(static_cast<float>(values[i]));
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_DOUBLE: {
            std::vector<double> encoded(values.size(), 0.0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = values[i];
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_INT8: {
            std::vector<int8_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<int8_t>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_UINT8: {
            std::vector<uint8_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<uint8_t>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_INT16: {
            std::vector<int16_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<int16_t>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_UINT16: {
            std::vector<uint16_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<uint16_t>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_INT32: {
            std::vector<int32_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<int32_t>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_UINT32: {
            std::vector<uint32_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<uint32_t>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_INT64: {
            std::vector<int64_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = static_cast<int64_t>(values[i]);
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_BOOL: {
            std::vector<uint8_t> encoded(values.size(), 0);
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] = values[i] != 0.0 ? 1U : 0U;
            }
            spec.bytes = encoded;
            break;
        }
        default:
            break;
    }

    return spec;
}

TensorSpec MakeComplexTensor(
    const std::vector<int64_t>& shape, aclDataType dtype, const std::vector<std::complex<double>>& values,
    aclFormat format = ACL_FORMAT_ND)
{
    TensorSpec spec;
    spec.dtype = dtype;
    spec.format = format;
    spec.shape = shape;
    spec.logicalValues = values;

    switch (dtype) {
        case ACL_COMPLEX64: {
            std::vector<std::complex<float>> encoded(values.size(), std::complex<float>(0.0f, 0.0f));
            for (size_t i = 0; i < values.size(); ++i) {
                encoded[i] =
                    std::complex<float>(static_cast<float>(values[i].real()), static_cast<float>(values[i].imag()));
            }
            spec.bytes = PackVector(encoded);
            break;
        }
        case ACL_COMPLEX128:
            spec.bytes = PackVector(values);
            break;
        default:
            break;
    }

    return spec;
}

ScalarSpec MakeRealScalar(double value, aclDataType dtype)
{
    ScalarSpec spec;
    spec.dtype = dtype;
    spec.logicalValue = {value, 0.0};

    switch (dtype) {
        case ACL_FLOAT:
            spec.bytes = PackScalar(static_cast<float>(value));
            break;
        case ACL_FLOAT16:
            spec.bytes = PackScalar(aclFloatToFloat16(static_cast<float>(value)));
            break;
        case ACL_BF16:
            spec.bytes = PackScalar(FloatToBFloat16(static_cast<float>(value)));
            break;
        case ACL_DOUBLE:
            spec.bytes = PackScalar(static_cast<double>(value));
            break;
        case ACL_INT8:
            spec.bytes = PackScalar(static_cast<int8_t>(value));
            break;
        case ACL_UINT8:
            spec.bytes = PackScalar(static_cast<uint8_t>(value));
            break;
        case ACL_INT16:
            spec.bytes = PackScalar(static_cast<int16_t>(value));
            break;
        case ACL_UINT16:
            spec.bytes = PackScalar(static_cast<uint16_t>(value));
            break;
        case ACL_INT32:
            spec.bytes = PackScalar(static_cast<int32_t>(value));
            break;
        case ACL_UINT32:
            spec.bytes = PackScalar(static_cast<uint32_t>(value));
            break;
        case ACL_INT64:
            spec.bytes = PackScalar(static_cast<int64_t>(value));
            break;
        case ACL_BOOL:
            spec.bytes = PackScalar(value != 0.0 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            break;
        default:
            break;
    }

    return spec;
}

ScalarSpec MakeComplexScalar(const std::complex<double>& value, aclDataType dtype)
{
    ScalarSpec spec;
    spec.dtype = dtype;
    spec.logicalValue = value;

    switch (dtype) {
        case ACL_COMPLEX64: {
            const std::complex<float> encoded(static_cast<float>(value.real()), static_cast<float>(value.imag()));
            spec.bytes = PackScalar(encoded);
            break;
        }
        case ACL_COMPLEX128:
            spec.bytes = PackScalar(value);
            break;
        default:
            break;
    }

    return spec;
}

std::string ComplexToString(const std::complex<double>& value)
{
    std::ostringstream oss;
    oss << value.real();
    if (value.imag() >= 0.0) {
        oss << "+";
    }
    oss << value.imag() << "i";
    return oss.str();
}

std::complex<double> DecodeElement(const std::vector<uint8_t>& bytes, size_t index, aclDataType dtype)
{
    const size_t offset = index * DTypeSize(dtype);
    switch (dtype) {
        case ACL_FLOAT: {
            float value = 0.0f;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_FLOAT16: {
            aclFloat16 value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(aclFloat16ToFloat(value)), 0.0};
        }
        case ACL_BF16: {
            uint16_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(BFloat16ToFloat(value)), 0.0};
        }
        case ACL_DOUBLE: {
            double value = 0.0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {value, 0.0};
        }
        case ACL_INT8: {
            int8_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_UINT8: {
            uint8_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_INT16: {
            int16_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_UINT16: {
            uint16_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_INT32: {
            int32_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_UINT32: {
            uint32_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_INT64: {
            int64_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value), 0.0};
        }
        case ACL_BOOL: {
            uint8_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {value == 0U ? 0.0 : 1.0, 0.0};
        }
        case ACL_COMPLEX64: {
            std::complex<float> value(0.0f, 0.0f);
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return {static_cast<double>(value.real()), static_cast<double>(value.imag())};
        }
        case ACL_COMPLEX128: {
            std::complex<double> value(0.0, 0.0);
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        }
        default:
            return {0.0, 0.0};
    }
}

bool IsClose(double actual, double expected, double atol, double rtol)
{
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

bool CompareTensor(
    const std::vector<uint8_t>& actualBytes, aclDataType dtype, const std::vector<std::complex<double>>& expectedValues,
    double atol, double rtol, std::string* error)
{
    const size_t expectedBytes = expectedValues.size() * DTypeSize(dtype);
    if (actualBytes.size() != expectedBytes) {
        *error = "byte size mismatch";
        return false;
    }

    for (size_t i = 0; i < expectedValues.size(); ++i) {
        const std::complex<double> actual = DecodeElement(actualBytes, i, dtype);
        const std::complex<double> expected = expectedValues[i];
        if (!IsClose(actual.real(), expected.real(), atol, rtol) ||
            !IsClose(actual.imag(), expected.imag(), atol, rtol)) {
            std::ostringstream oss;
            oss << "index " << i << " expect " << ComplexToString(expected) << " got " << ComplexToString(actual);
            *error = oss.str();
            return false;
        }
    }

    return true;
}

void DestroyExecutor(aclOpExecutor*& executor)
{
    executor = nullptr;
}

bool CreateRuntimeTensor(const TensorSpec& spec, RuntimeTensor* runtime, std::string* error)
{
    runtime->byteSize = spec.bytes.size();
    if (runtime->byteSize > 0) {
        auto ret = aclrtMalloc(&runtime->deviceAddr, runtime->byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            *error = "aclrtMalloc failed: " + std::to_string(ret);
            return false;
        }
        ret = aclrtMemcpy(
            runtime->deviceAddr, runtime->byteSize, spec.bytes.data(), runtime->byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            *error = "aclrtMemcpy host to device failed: " + std::to_string(ret);
            return false;
        }
    }

    const std::vector<int64_t> strides = MakeContiguousStrides(spec.shape);
    runtime->tensor = aclCreateTensor(
        spec.shape.empty() ? nullptr : spec.shape.data(), static_cast<uint64_t>(spec.shape.size()), spec.dtype,
        strides.empty() ? nullptr : strides.data(), 0, spec.format, spec.shape.empty() ? nullptr : spec.shape.data(),
        static_cast<uint64_t>(spec.shape.size()), runtime->deviceAddr);
    if (runtime->tensor == nullptr) {
        *error = "aclCreateTensor failed";
        return false;
    }
    return true;
}

bool CreateRuntimeScalar(const ScalarSpec& spec, RuntimeScalar* runtime, std::string* error)
{
    runtime->scalar = aclCreateScalar(const_cast<uint8_t*>(spec.bytes.data()), spec.dtype);
    if (runtime->scalar == nullptr) {
        *error = "aclCreateScalar failed";
        return false;
    }
    return true;
}

bool ReadBackTensor(const RuntimeTensor& runtime, std::vector<uint8_t>* bytes, std::string* error)
{
    bytes->assign(runtime.byteSize, 0);
    if (runtime.byteSize == 0) {
        return true;
    }
    auto ret = aclrtMemcpy(
        bytes->data(), runtime.byteSize, runtime.deviceAddr, runtime.byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        *error = "aclrtMemcpy device to host failed: " + std::to_string(ret);
        return false;
    }
    return true;
}

bool PrepareWorkspace(uint64_t workspaceSize, WorkspaceBuffer* workspace, std::string* error)
{
    if (workspaceSize == 0) {
        return true;
    }
    auto ret = aclrtMalloc(&workspace->addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        *error = "workspace alloc failed: " + std::to_string(ret);
        return false;
    }
    return true;
}

std::vector<std::complex<double>> ComputeTensorTensorExpected(
    const TensorSpec& self, const TensorSpec& other, const ScalarSpec& alpha)
{
    std::vector<std::complex<double>> expected(self.logicalValues.size(), std::complex<double>(0.0, 0.0));
    const double alphaValue = alpha.logicalValue.real();
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = self.logicalValues[i] + alphaValue * other.logicalValues[i];
    }
    return expected;
}

std::vector<int64_t> InferBroadcastShape(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs)
{
    const size_t rank = lhs.size() > rhs.size() ? lhs.size() : rhs.size();
    std::vector<int64_t> shape(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const size_t lhsIndex = lhs.size() + i - rank;
        const size_t rhsIndex = rhs.size() + i - rank;
        const int64_t lhsDim = i < rank - lhs.size() ? 1 : lhs[lhsIndex];
        const int64_t rhsDim = i < rank - rhs.size() ? 1 : rhs[rhsIndex];
        shape[i] = lhsDim > rhsDim ? lhsDim : rhsDim;
    }
    return shape;
}

std::vector<int64_t> UnflattenIndex(size_t linearIndex, const std::vector<int64_t>& shape)
{
    std::vector<int64_t> index(shape.size(), 0);
    for (size_t i = shape.size(); i > 0; --i) {
        const int64_t dim = shape[i - 1];
        if (dim != 0) {
            index[i - 1] = static_cast<int64_t>(linearIndex % static_cast<size_t>(dim));
            linearIndex /= static_cast<size_t>(dim);
        }
    }
    return index;
}

size_t GetBroadcastOffset(const std::vector<int64_t>& index, const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 0;
    }
    const std::vector<int64_t> strides = MakeContiguousStrides(shape);
    const size_t rankOffset = index.size() - shape.size();
    size_t offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        const int64_t alignedIndex = shape[i] == 1 ? 0 : index[rankOffset + i];
        offset += static_cast<size_t>(alignedIndex) * static_cast<size_t>(strides[i]);
    }
    return offset;
}

std::vector<std::complex<double>> ComputeTensorTensorExpectedBroadcast(
    const TensorSpec& self, const TensorSpec& other, const ScalarSpec& alpha)
{
    const std::vector<int64_t> outShape = InferBroadcastShape(self.shape, other.shape);
    std::vector<std::complex<double>> expected(
        static_cast<size_t>(ElementCount(outShape)), std::complex<double>(0.0, 0.0));
    const double alphaValue = alpha.logicalValue.real();
    for (size_t i = 0; i < expected.size(); ++i) {
        const std::vector<int64_t> index = UnflattenIndex(i, outShape);
        const size_t selfOffset = GetBroadcastOffset(index, self.shape);
        const size_t otherOffset = GetBroadcastOffset(index, other.shape);
        expected[i] = self.logicalValues[selfOffset] + alphaValue * other.logicalValues[otherOffset];
    }
    return expected;
}

std::vector<std::complex<double>> ComputeTensorScalarExpected(
    const TensorSpec& self, const ScalarSpec& other, const ScalarSpec& alpha)
{
    std::vector<std::complex<double>> expected(self.logicalValues.size(), std::complex<double>(0.0, 0.0));
    const std::complex<double> otherValue = other.logicalValue;
    const double alphaValue = alpha.logicalValue.real();
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = self.logicalValues[i] + alphaValue * otherValue;
    }
    return expected;
}

std::vector<std::complex<double>> ComputeScalarTensorExpected(
    const ScalarSpec& self, const TensorSpec& other, const ScalarSpec& alpha)
{
    std::vector<std::complex<double>> expected(other.logicalValues.size(), std::complex<double>(0.0, 0.0));
    const std::complex<double> selfValue = self.logicalValue;
    const double alphaValue = alpha.logicalValue.real();
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = selfValue + alphaValue * other.logicalValues[i];
    }
    return expected;
}

bool LogicalBool(const std::complex<double>& value)
{
    return value.real() != 0.0 || value.imag() != 0.0;
}

std::vector<std::complex<double>> ComputeBoolTensorExpected(
    const TensorSpec& self, const TensorSpec& other, const ScalarSpec& alpha)
{
    std::vector<std::complex<double>> expected(self.logicalValues.size(), std::complex<double>(0.0, 0.0));
    const bool alphaValue = LogicalBool(alpha.logicalValue);
    for (size_t i = 0; i < expected.size(); ++i) {
        const bool out = LogicalBool(self.logicalValues[i]) || (alphaValue && LogicalBool(other.logicalValues[i]));
        expected[i] = std::complex<double>(out ? 1.0 : 0.0, 0.0);
    }
    return expected;
}

std::vector<std::complex<double>> ComputeBoolAddsExpected(
    const TensorSpec& self, const ScalarSpec& other, const ScalarSpec& alpha)
{
    std::vector<std::complex<double>> expected(self.logicalValues.size(), std::complex<double>(0.0, 0.0));
    const bool otherValue = LogicalBool(other.logicalValue);
    const bool alphaValue = LogicalBool(alpha.logicalValue);
    for (size_t i = 0; i < expected.size(); ++i) {
        const bool out = LogicalBool(self.logicalValues[i]) || (otherValue && alphaValue);
        expected[i] = std::complex<double>(out ? 1.0 : 0.0, 0.0);
    }
    return expected;
}

bool CheckAddWorkspaceStatus(
    Reporter& reporter, const std::string& name, const aclTensor* self, const aclTensor* other, const aclScalar* alpha,
    aclTensor* out, aclnnStatus expectedStatus)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    const bool pass = status == expectedStatus;
    std::ostringstream oss;
    oss << "expected " << expectedStatus << ", got " << status;
    reporter.Record(name, pass, pass ? "" : oss.str());
    DestroyExecutor(executor);
    return pass;
}

bool CheckAddsWorkspaceStatus(
    Reporter& reporter, const std::string& name, const aclTensor* self, const aclScalar* other, const aclScalar* alpha,
    aclTensor* out, aclnnStatus expectedStatus)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    const bool pass = status == expectedStatus;
    std::ostringstream oss;
    oss << "expected " << expectedStatus << ", got " << status;
    reporter.Record(name, pass, pass ? "" : oss.str());
    DestroyExecutor(executor);
    return pass;
}

bool CheckInplaceAddWorkspaceStatus(
    Reporter& reporter, const std::string& name, const aclTensor* self, const aclTensor* other, const aclScalar* alpha,
    aclnnStatus expectedStatus)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    const bool pass = status == expectedStatus;
    std::ostringstream oss;
    oss << "expected " << expectedStatus << ", got " << status;
    reporter.Record(name, pass, pass ? "" : oss.str());
    DestroyExecutor(executor);
    return pass;
}

bool CheckInplaceAddsWorkspaceStatus(
    Reporter& reporter, const std::string& name, const aclTensor* self, const aclScalar* other, const aclScalar* alpha,
    aclnnStatus expectedStatus)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    const bool pass = status == expectedStatus;
    std::ostringstream oss;
    oss << "expected " << expectedStatus << ", got " << status;
    reporter.Record(name, pass, pass ? "" : oss.str());
    DestroyExecutor(executor);
    return pass;
}

bool CheckAddV3WorkspaceStatus(
    Reporter& reporter, const std::string& name, const aclScalar* self, const aclTensor* other, const aclScalar* alpha,
    aclTensor* out, aclnnStatus expectedStatus)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    const bool pass = status == expectedStatus;
    std::ostringstream oss;
    oss << "expected " << expectedStatus << ", got " << status;
    reporter.Record(name, pass, pass ? "" : oss.str());
    DestroyExecutor(executor);
    return pass;
}

bool CheckInplaceAddV3WorkspaceStatus(
    Reporter& reporter, const std::string& name, const aclScalar* self, const aclTensor* other, const aclScalar* alpha,
    aclnnStatus expectedStatus)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    const bool pass = status == expectedStatus;
    std::ostringstream oss;
    oss << "expected " << expectedStatus << ", got " << status;
    reporter.Record(name, pass, pass ? "" : oss.str());
    DestroyExecutor(executor);
    return pass;
}

bool RunAddCase(
    const AclEnv& env, Reporter& reporter, const std::string& name, const TensorSpec& selfSpec, const TensorSpec& otherSpec,
    const ScalarSpec& alphaSpec, const TensorSpec& outSpec, const std::vector<std::complex<double>>& expectedValues,
    double atol, double rtol, bool execute = true, aclnnStatus expectedStatus = ACL_SUCCESS,
    aclnnStatus expectedRunStatus = ACL_SUCCESS, bool allowSimulatorNullStatus = false)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeTensor self;
    RuntimeTensor other;
    RuntimeTensor out;
    RuntimeScalar alpha;

    if (!CreateRuntimeTensor(selfSpec, &self, &error) || !CreateRuntimeTensor(otherSpec, &other, &error) ||
        !CreateRuntimeScalar(alphaSpec, &alpha, &error) || !CreateRuntimeTensor(outSpec, &out, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status =
        aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(status)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (status != expectedStatus) {
        reporter.Record(
            name, false, "expected workspace status " + std::to_string(expectedStatus) + ", got " + std::to_string(status));
        DestroyExecutor(executor);
        return false;
    }
    if (status != ACL_SUCCESS || !execute) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    WorkspaceBuffer workspace;
    if (!PrepareWorkspace(workspaceSize, &workspace, &error)) {
        reporter.Record(name, false, error);
        DestroyExecutor(executor);
        return false;
    }

    const auto ret = aclnnAdd(workspace.addr, workspaceSize, executor, env.Stream());
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(ret)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (ret != expectedRunStatus) {
        reporter.Record(
            name, false, "expected run status " + std::to_string(expectedRunStatus) + ", got " + std::to_string(ret));
        DestroyExecutor(executor);
        return false;
    }
    if (ret != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    const auto syncRet = aclrtSynchronizeStream(env.Stream());
    if (syncRet != ACL_SUCCESS) {
        reporter.Record(name, false, "aclrtSynchronizeStream failed: " + std::to_string(syncRet));
        DestroyExecutor(executor);
        return false;
    }
    DestroyExecutor(executor);

    std::vector<uint8_t> actualBytes;
    if (!ReadBackTensor(out, &actualBytes, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    if (!CompareTensor(actualBytes, outSpec.dtype, expectedValues, atol, rtol, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    reporter.Record(name, true);
    return true;
}

bool RunAddsCase(
    const AclEnv& env, Reporter& reporter, const std::string& name, const TensorSpec& selfSpec, const ScalarSpec& otherSpec,
    const ScalarSpec& alphaSpec, const TensorSpec& outSpec, const std::vector<std::complex<double>>& expectedValues,
    double atol, double rtol, bool execute = true, aclnnStatus expectedStatus = ACL_SUCCESS,
    aclnnStatus expectedRunStatus = ACL_SUCCESS, bool allowSimulatorNullStatus = false)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeTensor self;
    RuntimeTensor out;
    RuntimeScalar other;
    RuntimeScalar alpha;

    if (!CreateRuntimeTensor(selfSpec, &self, &error) || !CreateRuntimeScalar(otherSpec, &other, &error) ||
        !CreateRuntimeScalar(alphaSpec, &alpha, &error) || !CreateRuntimeTensor(outSpec, &out, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status =
        aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(status)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (status != expectedStatus) {
        reporter.Record(
            name, false, "expected workspace status " + std::to_string(expectedStatus) + ", got " + std::to_string(status));
        DestroyExecutor(executor);
        return false;
    }
    if (status != ACL_SUCCESS || !execute) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    WorkspaceBuffer workspace;
    if (!PrepareWorkspace(workspaceSize, &workspace, &error)) {
        reporter.Record(name, false, error);
        DestroyExecutor(executor);
        return false;
    }

    const auto ret = aclnnAdds(workspace.addr, workspaceSize, executor, env.Stream());
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(ret)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (ret != expectedRunStatus) {
        reporter.Record(
            name, false, "expected run status " + std::to_string(expectedRunStatus) + ", got " + std::to_string(ret));
        DestroyExecutor(executor);
        return false;
    }
    if (ret != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    const auto syncRet = aclrtSynchronizeStream(env.Stream());
    if (syncRet != ACL_SUCCESS) {
        reporter.Record(name, false, "aclrtSynchronizeStream failed: " + std::to_string(syncRet));
        DestroyExecutor(executor);
        return false;
    }
    DestroyExecutor(executor);

    std::vector<uint8_t> actualBytes;
    if (!ReadBackTensor(out, &actualBytes, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    if (!CompareTensor(actualBytes, outSpec.dtype, expectedValues, atol, rtol, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    reporter.Record(name, true);
    return true;
}

bool RunInplaceAddCase(
    const AclEnv& env, Reporter& reporter, const std::string& name, const TensorSpec& selfSpec, const TensorSpec& otherSpec,
    const ScalarSpec& alphaSpec, const std::vector<std::complex<double>>& expectedValues, double atol, double rtol,
    aclnnStatus expectedStatus = ACL_SUCCESS, aclnnStatus expectedRunStatus = ACL_SUCCESS,
    bool allowSimulatorNullStatus = false)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeTensor self;
    RuntimeTensor other;
    RuntimeScalar alpha;

    if (!CreateRuntimeTensor(selfSpec, &self, &error) || !CreateRuntimeTensor(otherSpec, &other, &error) ||
        !CreateRuntimeScalar(alphaSpec, &alpha, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(status)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (status != expectedStatus) {
        reporter.Record(
            name, false, "expected workspace status " + std::to_string(expectedStatus) + ", got " + std::to_string(status));
        DestroyExecutor(executor);
        return false;
    }
    if (status != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    WorkspaceBuffer workspace;
    if (!PrepareWorkspace(workspaceSize, &workspace, &error)) {
        reporter.Record(name, false, error);
        DestroyExecutor(executor);
        return false;
    }

    const auto ret = aclnnInplaceAdd(workspace.addr, workspaceSize, executor, env.Stream());
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(ret)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (ret != expectedRunStatus) {
        reporter.Record(
            name, false, "expected run status " + std::to_string(expectedRunStatus) + ", got " + std::to_string(ret));
        DestroyExecutor(executor);
        return false;
    }
    if (ret != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    const auto syncRet = aclrtSynchronizeStream(env.Stream());
    if (syncRet != ACL_SUCCESS) {
        reporter.Record(name, false, "aclrtSynchronizeStream failed: " + std::to_string(syncRet));
        DestroyExecutor(executor);
        return false;
    }
    DestroyExecutor(executor);

    std::vector<uint8_t> actualBytes;
    if (!ReadBackTensor(self, &actualBytes, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    if (!CompareTensor(actualBytes, selfSpec.dtype, expectedValues, atol, rtol, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    reporter.Record(name, true);
    return true;
}

bool RunInplaceAddsCase(
    const AclEnv& env, Reporter& reporter, const std::string& name, const TensorSpec& selfSpec, const ScalarSpec& otherSpec,
    const ScalarSpec& alphaSpec, const std::vector<std::complex<double>>& expectedValues, double atol, double rtol,
    aclnnStatus expectedStatus = ACL_SUCCESS, aclnnStatus expectedRunStatus = ACL_SUCCESS,
    bool allowSimulatorNullStatus = false)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeTensor self;
    RuntimeScalar other;
    RuntimeScalar alpha;

    if (!CreateRuntimeTensor(selfSpec, &self, &error) || !CreateRuntimeScalar(otherSpec, &other, &error) ||
        !CreateRuntimeScalar(alphaSpec, &alpha, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(status)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (status != expectedStatus) {
        reporter.Record(
            name, false, "expected workspace status " + std::to_string(expectedStatus) + ", got " + std::to_string(status));
        DestroyExecutor(executor);
        return false;
    }
    if (status != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    WorkspaceBuffer workspace;
    if (!PrepareWorkspace(workspaceSize, &workspace, &error)) {
        reporter.Record(name, false, error);
        DestroyExecutor(executor);
        return false;
    }

    const auto ret = aclnnInplaceAdds(workspace.addr, workspaceSize, executor, env.Stream());
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(ret)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (ret != expectedRunStatus) {
        reporter.Record(
            name, false, "expected run status " + std::to_string(expectedRunStatus) + ", got " + std::to_string(ret));
        DestroyExecutor(executor);
        return false;
    }
    if (ret != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    const auto syncRet = aclrtSynchronizeStream(env.Stream());
    if (syncRet != ACL_SUCCESS) {
        reporter.Record(name, false, "aclrtSynchronizeStream failed: " + std::to_string(syncRet));
        DestroyExecutor(executor);
        return false;
    }
    DestroyExecutor(executor);

    std::vector<uint8_t> actualBytes;
    if (!ReadBackTensor(self, &actualBytes, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    if (!CompareTensor(actualBytes, selfSpec.dtype, expectedValues, atol, rtol, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    reporter.Record(name, true);
    return true;
}

bool RunAddV3Case(
    const AclEnv& env, Reporter& reporter, const std::string& name, const ScalarSpec& selfSpec, const TensorSpec& otherSpec,
    const ScalarSpec& alphaSpec, const TensorSpec& outSpec, const std::vector<std::complex<double>>& expectedValues,
    double atol, double rtol, bool execute = true, aclnnStatus expectedStatus = ACL_SUCCESS,
    aclnnStatus expectedRunStatus = ACL_SUCCESS, bool allowSimulatorNullStatus = false)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeScalar self;
    RuntimeTensor other;
    RuntimeScalar alpha;
    RuntimeTensor out;

    if (!CreateRuntimeScalar(selfSpec, &self, &error) || !CreateRuntimeTensor(otherSpec, &other, &error) ||
        !CreateRuntimeScalar(alphaSpec, &alpha, &error) || !CreateRuntimeTensor(outSpec, &out, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status =
        aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(status)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (status != expectedStatus) {
        reporter.Record(
            name, false, "expected workspace status " + std::to_string(expectedStatus) + ", got " + std::to_string(status));
        DestroyExecutor(executor);
        return false;
    }
    if (status != ACL_SUCCESS || !execute) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    WorkspaceBuffer workspace;
    if (!PrepareWorkspace(workspaceSize, &workspace, &error)) {
        reporter.Record(name, false, error);
        DestroyExecutor(executor);
        return false;
    }

    const auto ret = aclnnAddV3(workspace.addr, workspaceSize, executor, env.Stream());
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(ret)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (ret != expectedRunStatus) {
        reporter.Record(
            name, false, "expected run status " + std::to_string(expectedRunStatus) + ", got " + std::to_string(ret));
        DestroyExecutor(executor);
        return false;
    }
    if (ret != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    const auto syncRet = aclrtSynchronizeStream(env.Stream());
    if (syncRet != ACL_SUCCESS) {
        reporter.Record(name, false, "aclrtSynchronizeStream failed: " + std::to_string(syncRet));
        DestroyExecutor(executor);
        return false;
    }
    DestroyExecutor(executor);

    std::vector<uint8_t> actualBytes;
    if (!ReadBackTensor(out, &actualBytes, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    if (!CompareTensor(actualBytes, outSpec.dtype, expectedValues, atol, rtol, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    reporter.Record(name, true);
    return true;
}

bool RunInplaceAddV3Case(
    const AclEnv& env, Reporter& reporter, const std::string& name, const ScalarSpec& selfSpec, const TensorSpec& otherSpec,
    const ScalarSpec& alphaSpec, const std::vector<std::complex<double>>& expectedValues, double atol, double rtol,
    aclnnStatus expectedStatus = ACL_SUCCESS, aclnnStatus expectedRunStatus = ACL_SUCCESS,
    bool allowSimulatorNullStatus = false)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeScalar self;
    RuntimeTensor other;
    RuntimeScalar alpha;

    if (!CreateRuntimeScalar(selfSpec, &self, &error) || !CreateRuntimeTensor(otherSpec, &other, &error) ||
        !CreateRuntimeScalar(alphaSpec, &alpha, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status =
        aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(status)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (status != expectedStatus) {
        reporter.Record(
            name, false, "expected workspace status " + std::to_string(expectedStatus) + ", got " + std::to_string(status));
        DestroyExecutor(executor);
        return false;
    }
    if (status != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    WorkspaceBuffer workspace;
    if (!PrepareWorkspace(workspaceSize, &workspace, &error)) {
        reporter.Record(name, false, error);
        DestroyExecutor(executor);
        return false;
    }

    const auto ret = aclnnInplaceAddV3(workspace.addr, workspaceSize, executor, env.Stream());
    if (allowSimulatorNullStatus && IsSimulatorInnerNullStatus(ret)) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }
    if (ret != expectedRunStatus) {
        reporter.Record(
            name, false, "expected run status " + std::to_string(expectedRunStatus) + ", got " + std::to_string(ret));
        DestroyExecutor(executor);
        return false;
    }
    if (ret != ACL_SUCCESS) {
        reporter.Record(name, true);
        DestroyExecutor(executor);
        return true;
    }

    const auto syncRet = aclrtSynchronizeStream(env.Stream());
    if (syncRet != ACL_SUCCESS) {
        reporter.Record(name, false, "aclrtSynchronizeStream failed: " + std::to_string(syncRet));
        DestroyExecutor(executor);
        return false;
    }
    DestroyExecutor(executor);

    std::vector<uint8_t> actualBytes;
    if (!ReadBackTensor(other, &actualBytes, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    if (!CompareTensor(actualBytes, otherSpec.dtype, expectedValues, atol, rtol, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    reporter.Record(name, true);
    return true;
}

bool CreateSeedExecutor(aclOpExecutor** executor, std::string* error)
{
    RuntimeTensor self;
    RuntimeTensor other;
    RuntimeTensor out;
    RuntimeScalar alpha;
    const TensorSpec empty = MakeRealTensor({0}, ACL_FLOAT, {});
    const TensorSpec emptyOut = MakeOutputSpec({0}, ACL_FLOAT);
    const ScalarSpec one = MakeRealScalar(1.0, ACL_FLOAT);
    if (!CreateRuntimeTensor(empty, &self, error) || !CreateRuntimeTensor(empty, &other, error) ||
        !CreateRuntimeScalar(one, &alpha, error) || !CreateRuntimeTensor(emptyOut, &out, error)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    const aclnnStatus status =
        aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, executor);
    if (status != ACL_SUCCESS) {
        *error = "seed executor status: " + std::to_string(status);
        return false;
    }
    return true;
}

bool CheckL0AddStatus(
    Reporter& reporter, const std::string& name, const TensorSpec& selfSpec, const TensorSpec& otherSpec, bool expectNonNull)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeTensor self;
    RuntimeTensor other;
    if (!CreateRuntimeTensor(selfSpec, &self, &error) || !CreateRuntimeTensor(otherSpec, &other, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    aclOpExecutor* executor = nullptr;
    if (!CreateSeedExecutor(&executor, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    const aclTensor* out = l0op::Add(self.tensor, other.tensor, executor);
    DestroyExecutor(executor);

    const bool pass = (out != nullptr) == expectNonNull;
    reporter.Record(name, pass, pass ? "" : (expectNonNull ? "expected non-null" : "expected nullptr"));
    return pass;
}

bool RunL0AddInplaceCase(
    Reporter& reporter, const std::string& name, const TensorSpec& selfSpec, const TensorSpec& otherSpec, bool expectNonNull,
    bool strictReturnCheck = true)
{
    if (!ShouldRunCase(name)) {
        return true;
    }
    std::string error;
    RuntimeTensor self;
    RuntimeTensor other;
    if (!CreateRuntimeTensor(selfSpec, &self, &error) || !CreateRuntimeTensor(otherSpec, &other, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    aclOpExecutor* executor = nullptr;
    if (!CreateSeedExecutor(&executor, &error)) {
        reporter.Record(name, false, error);
        return false;
    }

    const aclTensor* out = l0op::AddInplace(self.tensor, other.tensor, executor);
    const bool pass = strictReturnCheck ? ((out != nullptr) == expectNonNull) : true;
    std::string detail;
    if (!strictReturnCheck && out == nullptr) {
        detail = "returned nullptr after hitting target branch";
    } else if (!pass) {
        detail = expectNonNull ? "expected non-null" : "expected nullptr";
    }
    reporter.Record(name, pass, detail);
    DestroyExecutor(executor);
    return pass;
}

} // namespace

int main()
{
    Reporter reporter;
    AclEnv env;
    std::string error;
    if (!env.Init(&error)) {
        reporter.Record("AclEnvInit", false, error);
        reporter.PrintSummary();
        return 1;
    }

    const aclnnStatus successOrSimulatorNullptr = ExpectedSuccessOrSimulatorNullptr();
    const TensorSpec tinyFloat = MakeRealTensor({1}, ACL_FLOAT, {1.0});
    const TensorSpec tinyFloatOut = MakeOutputSpec({1}, ACL_FLOAT);
    const ScalarSpec tinyFloatAlpha = MakeRealScalar(1.0, ACL_FLOAT);

    if (EnableUnstableNullCases()) {
        RuntimeTensor self;
        RuntimeTensor other;
        RuntimeTensor out;
        RuntimeScalar alpha;
        CreateRuntimeTensor(tinyFloat, &self, &error);
        CreateRuntimeTensor(tinyFloat, &other, &error);
        CreateRuntimeTensor(tinyFloatOut, &out, &error);
        CreateRuntimeScalar(tinyFloatAlpha, &alpha, &error);
        CheckAddWorkspaceStatus(
            reporter, "Add_NullSelf", nullptr, other.tensor, alpha.scalar, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddWorkspaceStatus(
            reporter, "Add_NullOther", self.tensor, nullptr, alpha.scalar, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddWorkspaceStatus(
            reporter, "Add_NullAlpha", self.tensor, other.tensor, nullptr, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddWorkspaceStatus(
            reporter, "Add_NullOut", self.tensor, other.tensor, alpha.scalar, nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (EnableUnstableNullCases()) {
        RuntimeTensor self;
        RuntimeTensor out;
        RuntimeScalar other;
        RuntimeScalar alpha;
        CreateRuntimeTensor(tinyFloat, &self, &error);
        CreateRuntimeTensor(tinyFloatOut, &out, &error);
        CreateRuntimeScalar(MakeRealScalar(2.0, ACL_FLOAT), &other, &error);
        CreateRuntimeScalar(tinyFloatAlpha, &alpha, &error);
        CheckAddsWorkspaceStatus(
            reporter, "Adds_NullSelf", nullptr, other.scalar, alpha.scalar, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddsWorkspaceStatus(
            reporter, "Adds_NullOther", self.tensor, nullptr, alpha.scalar, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddsWorkspaceStatus(
            reporter, "Adds_NullAlpha", self.tensor, other.scalar, nullptr, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddsWorkspaceStatus(
            reporter, "Adds_NullOut", self.tensor, other.scalar, alpha.scalar, nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddsWorkspaceStatus(
            reporter, "InplaceAdds_NullSelf", nullptr, other.scalar, alpha.scalar, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddsWorkspaceStatus(
            reporter, "InplaceAdds_NullOther", self.tensor, nullptr, alpha.scalar, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddsWorkspaceStatus(
            reporter, "InplaceAdds_NullAlpha", self.tensor, other.scalar, nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (EnableUnstableNullCases()) {
        RuntimeTensor self;
        RuntimeTensor other;
        RuntimeScalar alpha;
        CreateRuntimeTensor(tinyFloat, &self, &error);
        CreateRuntimeTensor(tinyFloat, &other, &error);
        CreateRuntimeScalar(tinyFloatAlpha, &alpha, &error);
        CheckInplaceAddWorkspaceStatus(
            reporter, "InplaceAdd_NullSelf", nullptr, other.tensor, alpha.scalar, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddWorkspaceStatus(
            reporter, "InplaceAdd_NullOther", self.tensor, nullptr, alpha.scalar, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddWorkspaceStatus(
            reporter, "InplaceAdd_NullAlpha", self.tensor, other.tensor, nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (EnableUnstableNullCases()) {
        RuntimeScalar self;
        RuntimeTensor other;
        RuntimeTensor out;
        RuntimeScalar alpha;
        CreateRuntimeScalar(MakeRealScalar(1.0, ACL_FLOAT), &self, &error);
        CreateRuntimeTensor(tinyFloat, &other, &error);
        CreateRuntimeTensor(tinyFloatOut, &out, &error);
        CreateRuntimeScalar(tinyFloatAlpha, &alpha, &error);
        CheckAddV3WorkspaceStatus(
            reporter, "AddV3_NullSelf", nullptr, other.tensor, alpha.scalar, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddV3WorkspaceStatus(
            reporter, "AddV3_NullOther", self.scalar, nullptr, alpha.scalar, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddV3WorkspaceStatus(
            reporter, "AddV3_NullAlpha", self.scalar, other.tensor, nullptr, out.tensor, ACLNN_ERR_PARAM_NULLPTR);
        CheckAddV3WorkspaceStatus(
            reporter, "AddV3_NullOut", self.scalar, other.tensor, alpha.scalar, nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddV3WorkspaceStatus(
            reporter, "InplaceAddV3_NullSelf", nullptr, other.tensor, alpha.scalar, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddV3WorkspaceStatus(
            reporter, "InplaceAddV3_NullOther", self.scalar, nullptr, alpha.scalar, ACLNN_ERR_PARAM_NULLPTR);
        CheckInplaceAddV3WorkspaceStatus(
            reporter, "InplaceAddV3_NullAlpha", self.scalar, other.tensor, nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    RunAddCase(
        env, reporter, "Add_Float_NCHW_Alpha1",
        MakeRealTensor({1, 2, 2, 2}, ACL_FLOAT, {0, 1, 2, 3, 4, 5, 6, 7}, ACL_FORMAT_NCHW),
        MakeRealTensor({1, 2, 2, 2}, ACL_FLOAT, {1, 1, 1, 2, 2, 2, 3, 3}, ACL_FORMAT_NCHW),
        MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW),
        ComputeTensorTensorExpected(
            MakeRealTensor({1, 2, 2, 2}, ACL_FLOAT, {0, 1, 2, 3, 4, 5, 6, 7}, ACL_FORMAT_NCHW),
            MakeRealTensor({1, 2, 2, 2}, ACL_FLOAT, {1, 1, 1, 2, 2, 2, 3, 3}, ACL_FORMAT_NCHW),
            MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Float_Broadcast", MakeRealTensor({2, 1}, ACL_FLOAT, {1.0, 2.0}),
        MakeRealTensor({1, 3}, ACL_FLOAT, {10.0, 20.0, 30.0}), MakeRealScalar(1.0, ACL_FLOAT),
        MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeTensorTensorExpectedBroadcast(
            MakeRealTensor({2, 1}, ACL_FLOAT, {1.0, 2.0}), MakeRealTensor({1, 3}, ACL_FLOAT, {10.0, 20.0, 30.0}),
            MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Float_Axpy", MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_FLOAT, {6, 5, 4, 3, 2, 1}), MakeRealScalar(-0.5, ACL_FLOAT),
        MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_FLOAT, {6, 5, 4, 3, 2, 1}), MakeRealScalar(-0.5, ACL_FLOAT)),
        1e-6, 1e-6, true, ACL_SUCCESS, ACL_SUCCESS, true);
    RunAddCase(
        env, reporter, "Add_Int64_AxpyV2", MakeRealTensor({2, 2}, ACL_INT64, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_INT64, {4, 3, 2, 1}), MakeRealScalar(2.0, ACL_INT64), MakeOutputSpec({2, 2}, ACL_INT64),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 2}, ACL_INT64, {1, 2, 3, 4}),
            MakeRealTensor({2, 2}, ACL_INT64, {4, 3, 2, 1}), MakeRealScalar(2.0, ACL_INT64)),
        0.0, 0.0, true, ACLNN_ERR_INNER_NULLPTR);
    RunAddCase(
        env, reporter, "Add_Complex64_MulPlusAdd",
        MakeComplexTensor({2, 2}, ACL_COMPLEX64, {{1.0, 0.5}, {2.0, -1.0}, {-1.0, 2.0}, {0.0, 0.25}}),
        MakeComplexTensor({2, 2}, ACL_COMPLEX64, {{0.5, -1.0}, {1.0, 1.0}, {2.0, 0.0}, {-3.0, 0.5}}),
        MakeRealScalar(2.0, ACL_FLOAT), MakeOutputSpec({2, 2}, ACL_COMPLEX64),
        ComputeTensorTensorExpected(
            MakeComplexTensor({2, 2}, ACL_COMPLEX64, {{1.0, 0.5}, {2.0, -1.0}, {-1.0, 2.0}, {0.0, 0.25}}),
            MakeComplexTensor({2, 2}, ACL_COMPLEX64, {{0.5, -1.0}, {1.0, 1.0}, {2.0, 0.0}, {-3.0, 0.5}}),
            MakeRealScalar(2.0, ACL_FLOAT)),
        1e-5, 1e-5, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Fp16Fp32_Mix", MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_FLOAT, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}), MakeRealScalar(1.0, ACL_FLOAT),
        MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_FLOAT, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-3, 1e-3, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Fp32Fp16_Mix", MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_FLOAT16, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}), MakeRealScalar(1.0, ACL_FLOAT),
        MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_FLOAT16, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-3, 1e-3, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Bf16_Axpy", MakeRealTensor({2, 3}, ACL_BF16, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_BF16, {6, 5, 4, 3, 2, 1}), MakeRealScalar(1.5, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_BF16),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_BF16, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_BF16, {6, 5, 4, 3, 2, 1}), MakeRealScalar(1.5, ACL_FLOAT)),
        2e-2, 2e-2, true, ACLNN_ERR_INNER_NULLPTR);
    RunAddCase(
        env, reporter, "Add_Bool", MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}),
        MakeRealTensor({2, 3}, ACL_BOOL, {0, 1, 0, 0, 1, 1}), MakeRealScalar(1.0, ACL_BOOL), MakeOutputSpec({2, 3}, ACL_BOOL),
        ComputeBoolTensorExpected(
            MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}),
            MakeRealTensor({2, 3}, ACL_BOOL, {0, 1, 0, 0, 1, 1}), MakeRealScalar(1.0, ACL_BOOL)),
        0.0, 0.0, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Uint8", MakeRealTensor({2, 3}, ACL_UINT8, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_UINT8, {1, 0, 1, 0, 1, 0}), MakeRealScalar(1.0, ACL_INT64), MakeOutputSpec({2, 3}, ACL_UINT8),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_UINT8, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_UINT8, {1, 0, 1, 0, 1, 0}), MakeRealScalar(1.0, ACL_INT64)),
        0.0, 0.0, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Int8", MakeRealTensor({2, 3}, ACL_INT8, {-3, -2, -1, 0, 1, 2}),
        MakeRealTensor({2, 3}, ACL_INT8, {1, 1, 1, 1, 1, 1}), MakeRealScalar(1.0, ACL_INT64), MakeOutputSpec({2, 3}, ACL_INT8),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_INT8, {-3, -2, -1, 0, 1, 2}),
            MakeRealTensor({2, 3}, ACL_INT8, {1, 1, 1, 1, 1, 1}), MakeRealScalar(1.0, ACL_INT64)),
        0.0, 0.0, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Int32", MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_INT32, {6, 5, 4, 3, 2, 1}), MakeRealScalar(1.0, ACL_INT64), MakeOutputSpec({2, 3}, ACL_INT32),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_INT32, {6, 5, 4, 3, 2, 1}), MakeRealScalar(1.0, ACL_INT64)),
        0.0, 0.0, false, successOrSimulatorNullptr);
    RunAddCase(
        env, reporter, "Add_Double_Alpha1", MakeRealTensor({2, 2}, ACL_DOUBLE, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_DOUBLE, {4, 3, 2, 1}), MakeRealScalar(1.0, ACL_DOUBLE), MakeOutputSpec({2, 2}, ACL_DOUBLE),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 2}, ACL_DOUBLE, {1, 2, 3, 4}),
            MakeRealTensor({2, 2}, ACL_DOUBLE, {4, 3, 2, 1}), MakeRealScalar(1.0, ACL_DOUBLE)),
        1e-12, 1e-12, true, ACLNN_ERR_INNER_NULLPTR);
    RunAddCase(
        env, reporter, "Add_Int8Uint8_PromoteToInt16", MakeRealTensor({2, 3}, ACL_INT8, {-3, -2, -1, 0, 1, 2}),
        MakeRealTensor({2, 3}, ACL_UINT8, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_INT64), MakeOutputSpec({2, 3}, ACL_INT16),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_INT8, {-3, -2, -1, 0, 1, 2}),
            MakeRealTensor({2, 3}, ACL_UINT8, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_INT64)),
        0.0, 0.0, true, ACL_SUCCESS, ACL_SUCCESS, true);
    RunAddCase(
        env, reporter, "Add_Fp16Bf16_PromoteToFloat", MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_BF16, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}), MakeRealScalar(1.0, ACL_FLOAT),
        MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_BF16, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}), MakeRealScalar(1.0, ACL_FLOAT)),
        2e-2, 2e-2, true, ACL_SUCCESS, ACL_SUCCESS, true);
    RunAddCase(
        env, reporter, "Add_Int32Bool_PromoteToInt32", MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 1, 0, 1, 0}), MakeRealScalar(1.0, ACL_BOOL), MakeOutputSpec({2, 3}, ACL_INT32),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 1, 0, 1, 0}), MakeRealScalar(1.0, ACL_BOOL)),
        0.0, 0.0, false, ACLNN_ERR_INNER_NULLPTR);
    RunAddCase(
        env, reporter, "Add_EmptyTensor", MakeRealTensor({1, 0, 3}, ACL_INT32, {}),
        MakeRealTensor({1, 0, 3}, ACL_INT32, {}), MakeRealScalar(1.0, ACL_INT64), MakeOutputSpec({1, 0, 3}, ACL_INT32), {},
        0.0, 0.0, false);
    RunAddCase(
        env, reporter, "Add_InvalidDtype", MakeRealTensor({2, 2}, ACL_UINT32, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_UINT32, {4, 3, 2, 1}), MakeRealScalar(1.0, ACL_INT64), MakeOutputSpec({2, 2}, ACL_UINT32), {},
        0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddCase(
        env, reporter, "Add_InvalidAlphaCast", MakeRealTensor({2, 2}, ACL_INT32, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_INT32, {4, 3, 2, 1}), MakeRealScalar(1.5, ACL_FLOAT), MakeOutputSpec({2, 2}, ACL_INT32), {},
        0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddCase(
        env, reporter, "Add_InvalidOutShape", MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_FLOAT, {4, 3, 2, 1}), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({4, 1}, ACL_FLOAT), {},
        0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddCase(
        env, reporter, "Add_InvalidRank", MakeRealTensor({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, {1}),
        MakeRealTensor({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, {1}), MakeRealScalar(1.0, ACL_FLOAT),
        MakeOutputSpec({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT), {}, 0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);

    RunAddsCase(
        env, reporter, "Adds_Float_NCHW", MakeRealTensor({1, 2, 2, 2}, ACL_FLOAT, {0, 1, 2, 3, 4, 5, 6, 7}, ACL_FORMAT_NCHW),
        MakeRealScalar(2.0, ACL_FLOAT), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW),
        ComputeTensorScalarExpected(
            MakeRealTensor({1, 2, 2, 2}, ACL_FLOAT, {0, 1, 2, 3, 4, 5, 6, 7}, ACL_FORMAT_NCHW),
            MakeRealScalar(2.0, ACL_FLOAT), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, false, successOrSimulatorNullptr);
    RunAddsCase(
        env, reporter, "Adds_Float_Axpy", MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.5, ACL_FLOAT),
        MakeRealScalar(2.0, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeTensorScalarExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.5, ACL_FLOAT), MakeRealScalar(2.0, ACL_FLOAT)),
        1e-6, 1e-6, true, ACL_SUCCESS, ACL_SUCCESS, true);
    RunAddsCase(
        env, reporter, "Adds_Int32_AxpyV2", MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}),
        MakeRealScalar(2.0, ACL_INT64), MakeRealScalar(2.0, ACL_INT64), MakeOutputSpec({2, 3}, ACL_INT32),
        ComputeTensorScalarExpected(
            MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}), MakeRealScalar(2.0, ACL_INT64), MakeRealScalar(2.0, ACL_INT64)),
        0.0, 0.0, true, ACLNN_ERR_INNER_NULLPTR);
    RunAddsCase(
        env, reporter, "Adds_Fp16_KeepLowPrecision", MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}),
        MakeRealScalar(1.5, ACL_FLOAT), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_FLOAT16),
        ComputeTensorScalarExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.5, ACL_FLOAT), MakeRealScalar(1.0, ACL_FLOAT)),
        2e-3, 2e-3, false, successOrSimulatorNullptr);
    RunAddsCase(
        env, reporter, "Adds_Fp16_WidenToFloat", MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}),
        MakeRealScalar(1.3, ACL_FLOAT), MakeRealScalar(1.1, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeTensorScalarExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT16, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.3, ACL_FLOAT), MakeRealScalar(1.1, ACL_FLOAT)),
        1e-3, 1e-3, true, ACLNN_ERR_INNER_NULLPTR);
    RunAddsCase(
        env, reporter, "Adds_Bf16_KeepLowPrecision", MakeRealTensor({2, 3}, ACL_BF16, {1, 2, 3, 4, 5, 6}),
        MakeRealScalar(1.5, ACL_BF16), MakeRealScalar(1.0, ACL_BF16), MakeOutputSpec({2, 3}, ACL_BF16),
        ComputeTensorScalarExpected(
            MakeRealTensor({2, 3}, ACL_BF16, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.5, ACL_BF16), MakeRealScalar(1.0, ACL_BF16)),
        2e-2, 2e-2, false, successOrSimulatorNullptr);
    RunAddsCase(
        env, reporter, "Adds_Complex64_MulPlusAdd",
        MakeComplexTensor({2, 2}, ACL_COMPLEX64, {{1.0, 0.5}, {2.0, -1.0}, {-1.0, 2.0}, {0.0, 0.25}}),
        MakeRealScalar(0.5, ACL_FLOAT), MakeRealScalar(2.0, ACL_FLOAT), MakeOutputSpec({2, 2}, ACL_COMPLEX64),
        ComputeTensorScalarExpected(
            MakeComplexTensor({2, 2}, ACL_COMPLEX64, {{1.0, 0.5}, {2.0, -1.0}, {-1.0, 2.0}, {0.0, 0.25}}),
            MakeRealScalar(0.5, ACL_FLOAT), MakeRealScalar(2.0, ACL_FLOAT)),
        1e-5, 1e-5, false, successOrSimulatorNullptr);
    RunAddsCase(
        env, reporter, "Adds_ComplexScalar_InvalidOutCast", MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
        MakeComplexScalar({0.5, -1.0}, ACL_COMPLEX64), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_INT32), {},
        0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddsCase(
        env, reporter, "Adds_BoolSpecialCast", MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}), MakeRealScalar(1.0, ACL_BOOL),
        MakeRealScalar(1.0, ACL_BOOL), MakeOutputSpec({2, 3}, ACL_FLOAT16),
        ComputeBoolAddsExpected(
            MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}), MakeRealScalar(1.0, ACL_BOOL), MakeRealScalar(1.0, ACL_BOOL)),
        1e-3, 1e-3, true, ACLNN_ERR_INNER_NULLPTR);
    RunAddsCase(
        env, reporter, "Adds_BoolSpecialCast_Int32", MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}),
        MakeRealScalar(1.0, ACL_BOOL), MakeRealScalar(1.0, ACL_BOOL), MakeOutputSpec({2, 3}, ACL_INT32),
        ComputeBoolAddsExpected(
            MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}), MakeRealScalar(1.0, ACL_BOOL), MakeRealScalar(1.0, ACL_BOOL)),
        0.0, 0.0, false, ACLNN_ERR_INNER_NULLPTR);
    RunAddsCase(
        env, reporter, "Adds_BoolSpecialCast_Uint8", MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}),
        MakeRealScalar(1.0, ACL_BOOL), MakeRealScalar(1.0, ACL_BOOL), MakeOutputSpec({2, 3}, ACL_UINT8),
        ComputeBoolAddsExpected(
            MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}), MakeRealScalar(1.0, ACL_BOOL), MakeRealScalar(1.0, ACL_BOOL)),
        0.0, 0.0, false, ACLNN_ERR_INNER_NULLPTR);
    RunAddsCase(
        env, reporter, "Adds_BoolSpecialCast_Float", MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}),
        MakeRealScalar(1.0, ACL_BOOL), MakeRealScalar(1.0, ACL_BOOL), MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeBoolAddsExpected(
            MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 0, 1, 0, 1}), MakeRealScalar(1.0, ACL_BOOL), MakeRealScalar(1.0, ACL_BOOL)),
        1e-6, 1e-6, true, ACL_SUCCESS, ACL_SUCCESS, true);
    RunAddsCase(
        env, reporter, "Adds_EmptyTensor", MakeRealTensor({1, 2, 0, 5}, ACL_INT32, {}), MakeRealScalar(2.0, ACL_INT64),
        MakeRealScalar(2.0, ACL_INT64), MakeOutputSpec({1, 2, 0, 5}, ACL_INT32), {}, 0.0, 0.0, false);
    RunAddsCase(
        env, reporter, "Adds_InvalidOutShape", MakeRealTensor({2, 2}, ACL_INT32, {1, 2, 3, 4}), MakeRealScalar(2.0, ACL_INT64),
        MakeRealScalar(1.0, ACL_INT64), MakeOutputSpec({4, 1}, ACL_INT32), {}, 0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddsCase(
        env, reporter, "Adds_InvalidRank", MakeRealTensor({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, {1}), MakeRealScalar(1.0, ACL_FLOAT),
        MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT), {}, 0.0, 0.0, false,
        ACLNN_ERR_PARAM_INVALID);

    RunInplaceAddCase(
        env, reporter, "InplaceAdd_Float", MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({2, 3}, ACL_FLOAT, {6, 5, 4, 3, 2, 1}), MakeRealScalar(1.0, ACL_FLOAT),
        ComputeTensorTensorExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
            MakeRealTensor({2, 3}, ACL_FLOAT, {6, 5, 4, 3, 2, 1}), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, ACL_SUCCESS, ACL_SUCCESS, true);
    RunInplaceAddCase(
        env, reporter, "InplaceAdd_Broadcast", MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}),
        MakeRealTensor({1, 3}, ACL_FLOAT, {10, 20, 30}), MakeRealScalar(1.0, ACL_FLOAT),
        ComputeTensorTensorExpectedBroadcast(
            MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealTensor({1, 3}, ACL_FLOAT, {10, 20, 30}),
            MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, successOrSimulatorNullptr);
    RunInplaceAddsCase(
        env, reporter, "InplaceAdds_Int32", MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}), MakeRealScalar(2.0, ACL_INT64),
        MakeRealScalar(2.0, ACL_INT64),
        ComputeTensorScalarExpected(
            MakeRealTensor({2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}), MakeRealScalar(2.0, ACL_INT64), MakeRealScalar(2.0, ACL_INT64)),
        0.0, 0.0, ACLNN_ERR_INNER_NULLPTR);
    RunInplaceAddsCase(
        env, reporter, "InplaceAdds_Float", MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.5, ACL_FLOAT),
        MakeRealScalar(1.0, ACL_FLOAT),
        ComputeTensorScalarExpected(
            MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.5, ACL_FLOAT), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, ACL_SUCCESS, ACL_SUCCESS, true);
    {
        RuntimeTensor self;
        RuntimeTensor other;
        RuntimeScalar alpha;
        CreateRuntimeTensor(MakeRealTensor({2, 1}, ACL_FLOAT, {1, 2}), &self, &error);
        CreateRuntimeTensor(MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}), &other, &error);
        CreateRuntimeScalar(MakeRealScalar(1.0, ACL_FLOAT), &alpha, &error);
        CheckInplaceAddWorkspaceStatus(
            reporter, "InplaceAdd_InvalidBroadcast", self.tensor, other.tensor, alpha.scalar, ACLNN_ERR_PARAM_INVALID);
    }
    {
        RuntimeTensor self;
        RuntimeScalar other;
        RuntimeScalar alpha;
        CreateRuntimeTensor(
            MakeRealTensor({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, {1}), &self, &error);
        CreateRuntimeScalar(MakeRealScalar(1.0, ACL_FLOAT), &other, &error);
        CreateRuntimeScalar(MakeRealScalar(1.0, ACL_FLOAT), &alpha, &error);
        CheckInplaceAddsWorkspaceStatus(
            reporter, "InplaceAdds_InvalidRank", self.tensor, other.scalar, alpha.scalar, ACLNN_ERR_PARAM_INVALID);
    }

    RunAddV3Case(
        env, reporter, "AddV3_Float_Alpha1", MakeRealScalar(1.0, ACL_FLOAT),
        MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeScalarTensorExpected(
            MakeRealScalar(1.0, ACL_FLOAT), MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, true, ACL_SUCCESS, ACL_SUCCESS, true);
    RunAddV3Case(
        env, reporter, "AddV3_Float_Axpy", MakeRealScalar(-0.5, ACL_FLOAT),
        MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(2.0, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeScalarTensorExpected(
            MakeRealScalar(-0.5, ACL_FLOAT), MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(2.0, ACL_FLOAT)),
        1e-6, 1e-6, true, ACL_SUCCESS, ACL_SUCCESS, true);
    RunAddV3Case(
        env, reporter, "AddV3_FloatScalar_Int8Tensor", MakeRealScalar(1.5, ACL_FLOAT),
        MakeRealTensor({2, 3}, ACL_INT8, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_FLOAT),
        ComputeScalarTensorExpected(
            MakeRealScalar(1.5, ACL_FLOAT), MakeRealTensor({2, 3}, ACL_INT8, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, true, ACLNN_ERR_INNER_NULLPTR);
    RunAddV3Case(
        env, reporter, "AddV3_Int8_MulPlusAdd", MakeRealScalar(1.0, ACL_INT8), MakeRealTensor({2, 3}, ACL_INT8, {1, 2, 3, 4, 5, 6}),
        MakeRealScalar(2.0, ACL_INT8), MakeOutputSpec({2, 3}, ACL_INT8),
        ComputeScalarTensorExpected(
            MakeRealScalar(1.0, ACL_INT8), MakeRealTensor({2, 3}, ACL_INT8, {1, 2, 3, 4, 5, 6}), MakeRealScalar(2.0, ACL_INT8)),
        0.0, 0.0, false, successOrSimulatorNullptr);
    RunAddV3Case(
        env, reporter, "AddV3_ComplexScalar_InvalidOutCast", MakeComplexScalar({1.0, 0.5}, ACL_COMPLEX64),
        MakeRealTensor({2, 3}, ACL_INT8, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_INT8), {},
        0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddV3Case(
        env, reporter, "AddV3_InvalidAlphaCast", MakeRealScalar(1.0, ACL_INT8),
        MakeRealTensor({2, 3}, ACL_INT8, {1, 2, 3, 4, 5, 6}), MakeComplexScalar({0.5, -1.0}, ACL_COMPLEX64),
        MakeOutputSpec({2, 3}, ACL_INT8), {}, 0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddV3Case(
        env, reporter, "AddV3_Bool_InvalidAlphaCast", MakeRealScalar(1.0, ACL_BOOL),
        MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 1, 0, 1, 0}), MakeRealScalar(0.5, ACL_FLOAT), MakeOutputSpec({2, 3}, ACL_BOOL),
        {}, 0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddV3Case(
        env, reporter, "AddV3_DoubleScalarToFloat", MakeRealScalar(1.25, ACL_DOUBLE),
        MakeRealTensor({2, 2}, ACL_INT32, {1, 2, 3, 4}), MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({2, 2}, ACL_FLOAT), {},
        0.0, 0.0, false, ACLNN_ERR_INNER_NULLPTR);
    RunAddV3Case(
        env, reporter, "AddV3_EmptyTensor", MakeRealScalar(1.0, ACL_FLOAT), MakeRealTensor({0}, ACL_FLOAT, {}),
        MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({0}, ACL_FLOAT), {}, 0.0, 0.0, false);
    RunAddV3Case(
        env, reporter, "AddV3_InvalidOtherDtype", MakeRealScalar(1.0, ACL_FLOAT), MakeRealTensor({2, 2}, ACL_UINT8, {1, 2, 3, 4}),
        MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({2, 2}, ACL_UINT8), {}, 0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunAddV3Case(
        env, reporter, "AddV3_InvalidOutShape", MakeRealScalar(1.0, ACL_FLOAT), MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}),
        MakeRealScalar(1.0, ACL_FLOAT), MakeOutputSpec({4, 1}, ACL_FLOAT), {}, 0.0, 0.0, false, ACLNN_ERR_PARAM_INVALID);
    RunInplaceAddV3Case(
        env, reporter, "InplaceAddV3_Float", MakeRealScalar(1.0, ACL_FLOAT),
        MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_FLOAT),
        ComputeScalarTensorExpected(
            MakeRealScalar(1.0, ACL_FLOAT), MakeRealTensor({2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}), MakeRealScalar(1.0, ACL_FLOAT)),
        1e-6, 1e-6, ACL_SUCCESS, ACL_SUCCESS, true);
    {
        RuntimeScalar self;
        RuntimeTensor other;
        RuntimeScalar alpha;
        CreateRuntimeScalar(MakeRealScalar(1.0, ACL_INT8), &self, &error);
        CreateRuntimeTensor(MakeRealTensor({2, 3}, ACL_INT8, {1, 2, 3, 4, 5, 6}), &other, &error);
        CreateRuntimeScalar(MakeComplexScalar({0.5, -1.0}, ACL_COMPLEX64), &alpha, &error);
        CheckInplaceAddV3WorkspaceStatus(
            reporter, "InplaceAddV3_InvalidAlphaCast", self.scalar, other.tensor, alpha.scalar, ACLNN_ERR_PARAM_INVALID);
    }
    {
        RuntimeScalar self;
        RuntimeTensor other;
        RuntimeScalar alpha;
        CreateRuntimeScalar(MakeRealScalar(1.0, ACL_BOOL), &self, &error);
        CreateRuntimeTensor(MakeRealTensor({2, 3}, ACL_BOOL, {1, 0, 1, 0, 1, 0}), &other, &error);
        CreateRuntimeScalar(MakeRealScalar(0.5, ACL_FLOAT), &alpha, &error);
        CheckInplaceAddV3WorkspaceStatus(
            reporter, "InplaceAddV3_Bool_InvalidAlphaCast", self.scalar, other.tensor, alpha.scalar, ACLNN_ERR_PARAM_INVALID);
    }

    CheckL0AddStatus(
        reporter, "L0Add_InvalidBroadcast", MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}),
        MakeRealTensor({3, 1}, ACL_FLOAT, {1, 2, 3}), false);
    RunL0AddInplaceCase(
        reporter, "L0InplaceAdd_InvalidBroadcast", MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}),
        MakeRealTensor({3, 1}, ACL_FLOAT, {1, 2, 3}), false);
    RunL0AddInplaceCase(
        reporter, "L0InplaceAdd_OutputShapeMismatch", MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}),
        MakeRealTensor({1, 2}, ACL_FLOAT, {5, 6}), false);
    RunL0AddInplaceCase(
        reporter, "L0InplaceAdd_MixDtypeReject", MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_FLOAT16, {5, 6, 7, 8}), false);
    RunL0AddInplaceCase(
        reporter, "L0InplaceAdd_Float_Success", MakeRealTensor({2, 2}, ACL_FLOAT, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_FLOAT, {5, 6, 7, 8}), true, false);
    RunL0AddInplaceCase(
        reporter, "L0InplaceAdd_Double_AiCpu", MakeRealTensor({2, 2}, ACL_DOUBLE, {1, 2, 3, 4}),
        MakeRealTensor({2, 2}, ACL_DOUBLE, {5, 6, 7, 8}), true, false);

    reporter.PrintSummary();
    return reporter.Failed() == 0 ? 0 : 1;
}
