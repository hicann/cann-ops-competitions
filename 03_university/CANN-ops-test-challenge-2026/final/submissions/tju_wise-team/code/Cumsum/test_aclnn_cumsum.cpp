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
#include <cstdio>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)          \
    do {                                 \
        printf(message, ##__VA_ARGS__);  \
    } while (0)

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS ACL_SUCCESS
#endif

#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID ACL_ERROR_INVALID_PARAM
#endif

#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR ACL_ERROR_INVALID_PARAM
#endif

namespace {

static const aclnnStatus STATUS_SUCCESS = static_cast<aclnnStatus>(0);
static const aclnnStatus STATUS_PARAM_NULLPTR = static_cast<aclnnStatus>(161001);
static const aclnnStatus STATUS_PARAM_INVALID = static_cast<aclnnStatus>(161002);
static const aclnnStatus STATUS_INNER_NULLPTR = static_cast<aclnnStatus>(561103);

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

size_t GetDataTypeSize(aclDataType dataType)
{
    switch (dataType) {
        case ACL_BOOL:
        case ACL_INT8:
        case ACL_UINT8:
            return 1;
        case ACL_INT16:
        case ACL_UINT16:
        case ACL_FLOAT16:
        case ACL_BF16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
        case ACL_UINT32:
            return 4;
        case ACL_DOUBLE:
        case ACL_INT64:
        case ACL_UINT64:
            return 8;
        case ACL_COMPLEX64:
            return 8;
        case ACL_COMPLEX128:
            return 16;
        default:
            return 0;
    }
}

const char* GetDataTypeName(aclDataType dataType)
{
    switch (dataType) {
        case ACL_INT8:
            return "INT8";
        case ACL_INT16:
            return "INT16";
        case ACL_INT32:
            return "INT32";
        case ACL_INT64:
            return "INT64";
        case ACL_UINT8:
            return "UINT8";
        case ACL_UINT16:
            return "UINT16";
        case ACL_UINT32:
            return "UINT32";
        case ACL_UINT64:
            return "UINT64";
        case ACL_FLOAT16:
            return "FLOAT16";
        case ACL_FLOAT:
            return "FLOAT";
        case ACL_DOUBLE:
            return "DOUBLE";
        case ACL_COMPLEX64:
            return "COMPLEX64";
        case ACL_COMPLEX128:
            return "COMPLEX128";
        case ACL_BOOL:
            return "BOOL";
        default:
            return "UNKNOWN";
    }
}

std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

class TensorHolder {
public:
    aclTensor* tensor = nullptr;
    void* deviceAddr = nullptr;
    size_t bytes = 0;

    ~TensorHolder()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
            deviceAddr = nullptr;
        }
    }
};

int CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dataType, const void* hostData, TensorHolder& holder)
{
    auto elemSize = GetDataTypeSize(dataType);
    CHECK_RET(elemSize > 0, LOG_PRINT("Unsupported dtype: %d\n", static_cast<int>(dataType)); return 1);
    auto elemNum = GetShapeSize(shape);
    CHECK_RET(elemNum >= 0, LOG_PRINT("Invalid shape size.\n"); return 1);
    holder.bytes = static_cast<size_t>(elemNum) * elemSize;

    if (holder.bytes > 0) {
        auto ret = aclrtMalloc(&holder.deviceAddr, holder.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        std::vector<uint8_t> zeros(holder.bytes, 0);
        const void* src = (hostData == nullptr) ? static_cast<const void*>(zeros.data()) : hostData;
        ret = aclrtMemcpy(holder.deviceAddr, holder.bytes, src, holder.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
    }

    auto strides = GetContiguousStrides(shape);
    const int64_t* shapeData = shape.empty() ? nullptr : shape.data();
    const int64_t* strideData = strides.empty() ? nullptr : strides.data();
    holder.tensor = aclCreateTensor(shapeData, shape.size(), dataType, strideData, 0, aclFormat::ACL_FORMAT_ND,
                                    shapeData, shape.size(), holder.deviceAddr);
    CHECK_RET(holder.tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
    return ACL_SUCCESS;
}

template <typename T>
bool ValueClose(const T& actual, const T& expected, float)
{
    return actual == expected;
}

template <>
bool ValueClose<float>(const float& actual, const float& expected, float tol)
{
    return std::fabs(actual - expected) <= tol;
}

template <>
bool ValueClose<double>(const double& actual, const double& expected, float tol)
{
    return std::fabs(actual - expected) <= tol;
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

aclnnStatus CallGetWorkspace(const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out);
aclnnStatus CallGetWorkspaceV2(const aclTensor* self, int64_t dim, bool exclusive, bool reverse, aclTensor* out);

template <typename T>
std::vector<T> CumsumRef(const std::vector<T>& input, const std::vector<int64_t>& shape, int64_t dim,
                         bool exclusive, bool reverse)
{
    std::vector<int64_t> dims = shape.empty() ? std::vector<int64_t>{1} : shape;
    int64_t rank = static_cast<int64_t>(dims.size());
    if (dim < 0) {
        dim += rank;
    }

    int64_t outer = 1;
    int64_t inner = 1;
    int64_t axis = dims[dim];
    for (int64_t i = 0; i < dim; ++i) {
        outer *= dims[i];
    }
    for (int64_t i = dim + 1; i < rank; ++i) {
        inner *= dims[i];
    }

    std::vector<T> out(input.size(), static_cast<T>(0));
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            T running = static_cast<T>(0);
            if (!reverse) {
                for (int64_t a = 0; a < axis; ++a) {
                    int64_t idx = (o * axis + a) * inner + i;
                    if (exclusive) {
                        out[idx] = running;
                        running = static_cast<T>(running + input[idx]);
                    } else {
                        running = static_cast<T>(running + input[idx]);
                        out[idx] = running;
                    }
                }
            } else {
                for (int64_t a = axis - 1; a >= 0; --a) {
                    int64_t idx = (o * axis + a) * inner + i;
                    if (exclusive) {
                        out[idx] = running;
                        running = static_cast<T>(running + input[idx]);
                    } else {
                        running = static_cast<T>(running + input[idx]);
                        out[idx] = running;
                    }
                }
            }
        }
    }
    return out;
}

template <typename T>
bool CheckVectorClose(const std::vector<T>& actual, const std::vector<T>& expected, float tol)
{
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!ValueClose<T>(actual[i], expected[i], tol)) {
            LOG_PRINT("Mismatch at %zu\n", i);
            return false;
        }
    }
    return true;
}

bool RunStatusCase(const std::string& name, aclnnStatus expected, std::function<aclnnStatus()> fn)
{
    aclnnStatus ret = fn();
    bool ok = (ret == expected);
    LOG_PRINT("[%-46s] expected=%d actual=%d %s\n", name.c_str(), static_cast<int>(expected),
              static_cast<int>(ret), ok ? "PASS" : "FAIL");
    return ok;
}

bool RunStatusCaseAnyOf(const std::string& name, std::initializer_list<aclnnStatus> expectedList,
                        std::function<aclnnStatus()> fn)
{
    aclnnStatus ret = fn();
    bool ok = false;
    for (auto expected : expectedList) {
        if (ret == expected) {
            ok = true;
            break;
        }
    }
    LOG_PRINT("[%-46s] actual=%d %s\n", name.c_str(), static_cast<int>(ret), ok ? "PASS" : "FAIL");
    return ok;
}

bool RunCumsumWorkspaceStatusCase(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                                  aclDataType xType, aclDataType yType,
                                  std::initializer_list<aclnnStatus> expectedList)
{
    TensorHolder self;
    TensorHolder out;
    if (CreateAclTensor(shape, xType, nullptr, self) != ACL_SUCCESS ||
        CreateAclTensor(shape, yType, nullptr, out) != ACL_SUCCESS) {
        LOG_PRINT("[%s] tensor create failed.\n", name.c_str());
        return false;
    }
    return RunStatusCaseAnyOf(name, expectedList,
                              [&]() { return CallGetWorkspace(self.tensor, dim, yType, out.tensor); });
}

bool RunCumsumV2WorkspaceStatusCase(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                                    bool exclusive, bool reverse, aclDataType xType, aclDataType yType,
                                    std::initializer_list<aclnnStatus> expectedList)
{
    TensorHolder self;
    TensorHolder out;
    if (CreateAclTensor(shape, xType, nullptr, self) != ACL_SUCCESS ||
        CreateAclTensor(shape, yType, nullptr, out) != ACL_SUCCESS) {
        LOG_PRINT("[%s] tensor create failed.\n", name.c_str());
        return false;
    }
    return RunStatusCaseAnyOf(name, expectedList,
                              [&]() { return CallGetWorkspaceV2(self.tensor, dim, exclusive, reverse, out.tensor); });
}

bool RunCumsumWorkspaceStatusCaseWithDeviceMemoryPressure(const std::string& name, const std::vector<int64_t>& shape,
                                                          int64_t dim, aclDataType xType, aclDataType yType,
                                                          std::initializer_list<aclnnStatus> expectedList,
                                                          bool useV2, bool exclusive = false, bool reverse = false)
{
    TensorHolder self;
    TensorHolder out;
    if (CreateAclTensor(shape, xType, nullptr, self) != ACL_SUCCESS ||
        CreateAclTensor(shape, yType, nullptr, out) != ACL_SUCCESS) {
        LOG_PRINT("[%s] tensor create failed before pressure.\n", name.c_str());
        return false;
    }

    std::vector<void*> reserved;
    const size_t blockBytes = 32 * 1024 * 1024;
    for (int i = 0; i < 256; ++i) {
        void* ptr = nullptr;
        auto ret = aclrtMalloc(&ptr, blockBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS || ptr == nullptr) {
            break;
        }
        reserved.push_back(ptr);
    }

    bool ok = false;
    if (useV2) {
        ok = RunStatusCaseAnyOf(name, expectedList,
                                [&]() { return CallGetWorkspaceV2(self.tensor, dim, exclusive, reverse, out.tensor); });
    } else {
        ok = RunStatusCaseAnyOf(name, expectedList,
                                [&]() { return CallGetWorkspace(self.tensor, dim, yType, out.tensor); });
    }

    for (auto ptr : reserved) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    return ok;
}

template <typename T>
bool RunCumsumCase(const std::string& name, const std::vector<T>& input, const std::vector<int64_t>& shape, int64_t dim,
                   aclDataType dtype, aclrtStream stream)
{
    TensorHolder self;
    TensorHolder out;
    auto ret = CreateAclTensor(shape, dtype, input.data(), self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(shape, dtype, nullptr, out);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self.tensor, dim, dtype, out.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == STATUS_SUCCESS, LOG_PRINT("%s GetWorkspace failed ret=%d\n", name.c_str(), ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed ret=%d\n", ret); return false);
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
    }
    CHECK_RET(ret == STATUS_SUCCESS, LOG_PRINT("%s execute failed ret=%d\n", name.c_str(), ret); return false);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("stream sync failed ret=%d\n", ret); return false);

    std::vector<T> actual(input.size(), static_cast<T>(0));
    if (out.bytes > 0) {
        ret = aclrtMemcpy(actual.data(), out.bytes, out.deviceAddr, out.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("D2H failed ret=%d\n", ret); return false);
    }
    auto expected = CumsumRef(input, shape, dim, false, false);
    bool ok = CheckVectorClose(actual, expected, 1e-4f);
    LOG_PRINT("[%-46s] %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

bool RunCumsumV2Case(const std::string& name, const std::vector<float>& input, const std::vector<int64_t>& shape,
                     int64_t dim, bool exclusive, bool reverse, aclrtStream stream)
{
    TensorHolder self;
    TensorHolder out;
    auto ret = CreateAclTensor(shape, ACL_FLOAT, input.data(), self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(shape, ACL_FLOAT, nullptr, out);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, dim, exclusive, reverse, out.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == STATUS_SUCCESS, LOG_PRINT("%s GetWorkspace failed ret=%d\n", name.c_str(), ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed ret=%d\n", ret); return false);
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == STATUS_SUCCESS, LOG_PRINT("%s execute failed ret=%d\n", name.c_str(), ret); return false);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("stream sync failed ret=%d\n", ret); return false);

    std::vector<float> actual(input.size(), 0.f);
    if (out.bytes > 0) {
        ret = aclrtMemcpy(actual.data(), out.bytes, out.deviceAddr, out.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("D2H failed ret=%d\n", ret); return false);
    }
    auto expected = CumsumRef(input, shape, dim, exclusive, reverse);
    bool ok = CheckVectorClose(actual, expected, 1e-4f);
    LOG_PRINT("[%-46s] %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

uint16_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFU;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000U;
        uint32_t shifted = mant >> static_cast<uint32_t>(1 - exp + 13);
        uint32_t roundBit = (mant >> static_cast<uint32_t>(1 - exp + 12)) & 1U;
        return static_cast<uint16_t>(sign | shifted | roundBit);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    uint32_t rounded = mant + 0x1000U;
    if (rounded & 0x800000U) {
        rounded = 0;
        exp += 1;
        if (exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded >> 13));
}

float Fp16ToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exp = (value >> 10) & 0x1FU;
    uint32_t mant = value & 0x03FFU;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFU;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
T EncodeValue(double value)
{
    return static_cast<T>(value);
}

template <>
uint16_t EncodeValue<uint16_t>(double value)
{
    return FloatToFp16(static_cast<float>(value));
}

template <typename T>
double DecodeValue(T value)
{
    return static_cast<double>(value);
}

template <>
double DecodeValue<uint16_t>(uint16_t value)
{
    return static_cast<double>(Fp16ToFloat(value));
}

template <typename T>
std::vector<T> EncodeVector(const std::vector<double>& values)
{
    std::vector<T> encoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        encoded[i] = EncodeValue<T>(values[i]);
    }
    return encoded;
}

template <typename T>
std::vector<double> DecodeVector(const std::vector<T>& values)
{
    std::vector<double> decoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        decoded[i] = DecodeValue<T>(values[i]);
    }
    return decoded;
}

std::vector<uint16_t> EncodeBf16Vector(const std::vector<double>& values)
{
    std::vector<uint16_t> encoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        encoded[i] = FloatToBf16(static_cast<float>(values[i]));
    }
    return encoded;
}

std::vector<double> DecodeBf16Vector(const std::vector<uint16_t>& values)
{
    std::vector<double> decoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        decoded[i] = static_cast<double>(Bf16ToFloat(values[i]));
    }
    return decoded;
}

struct ErrorInfo {
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    int64_t maxErrorIndex = -1;
};

std::string PreviewVector(const std::vector<double>& values, size_t limit = 6)
{
    std::ostringstream oss;
    oss << std::setprecision(8) << "[";
    if (values.size() <= limit * 2) {
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << values[i];
        }
    } else {
        for (size_t i = 0; i < limit; ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << values[i];
        }
        oss << ", ..., ";
        for (size_t i = values.size() - limit; i < values.size(); ++i) {
            if (i != values.size() - limit) {
                oss << ", ";
            }
            oss << values[i];
        }
    }
    oss << "]";
    return oss.str();
}

ErrorInfo CalcError(const std::vector<double>& actual, const std::vector<double>& expected)
{
    ErrorInfo info;
    for (size_t i = 0; i < actual.size(); ++i) {
        double absError = std::fabs(actual[i] - expected[i]);
        double relError = absError / std::max(1.0, std::fabs(expected[i]));
        if (absError > info.maxAbsError) {
            info.maxAbsError = absError;
            info.maxRelError = relError;
            info.maxErrorIndex = static_cast<int64_t>(i);
        }
    }
    return info;
}

void PrintCaseResult(const std::string& name, const std::vector<double>& expected, const std::vector<double>& actual,
                     const ErrorInfo& error, bool pass, const std::string& note)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    LOG_PRINT("  Expected: %s\n", PreviewVector(expected).c_str());
    LOG_PRINT("  Actual:   %s\n", PreviewVector(actual).c_str());
    LOG_PRINT("  Max error: %.10g (at position %lld), max relative error: %.10g\n", error.maxAbsError,
              static_cast<long long>(error.maxErrorIndex), error.maxRelError);
    if (!note.empty()) {
        LOG_PRINT("  Note: %s\n", note.c_str());
    }
    LOG_PRINT("  [%s]\n\n", pass ? "PASS" : "FAIL");
}

bool CompareWithTolerance(const std::vector<double>& actual, const std::vector<double>& expected,
                          double atol, double rtol, bool exact, double& maxAbsError, int64_t& maxErrorIndex)
{
    if (actual.size() != expected.size()) {
        return false;
    }
    maxAbsError = 0.0;
    maxErrorIndex = -1;
    for (size_t i = 0; i < actual.size(); ++i) {
        double absError = std::fabs(actual[i] - expected[i]);
        if (absError > maxAbsError) {
            maxAbsError = absError;
            maxErrorIndex = static_cast<int64_t>(i);
        }
        if (exact) {
            if (actual[i] != expected[i]) {
                return false;
            }
            continue;
        }
        double tolerance = atol + rtol * std::fabs(expected[i]);
        if (absError > tolerance) {
            return false;
        }
    }
    return true;
}

template <typename InT, typename OutT>
bool RunPrecisionCumsumCase(const std::string& name, const std::vector<int64_t>& shape,
                            const std::vector<double>& inputValues, aclDataType inputType, aclDataType outputType,
                            int64_t dim, bool useV2, bool exclusive, bool reverse,
                            double atol, double rtol, bool exact,
                            const std::function<std::vector<InT>(const std::vector<double>&)>& encodeInput,
                            const std::function<std::vector<double>(const std::vector<InT>&)>& decodeInput,
                            const std::function<std::vector<double>(const std::vector<OutT>&)>& decodeOutput,
                            aclrtStream stream, bool passWhenUnsupported = false,
                            const std::string& note = "")
{
    auto inputHostData = encodeInput(inputValues);

    TensorHolder self;
    TensorHolder out;
    auto ret = CreateAclTensor(shape, inputType, inputHostData.data(), self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(shape, outputType, nullptr, out);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, dim, exclusive, reverse, out.tensor, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self.tensor, dim, outputType, out.tensor, &workspaceSize, &executor);
    }

    if (ret != STATUS_SUCCESS) {
        bool pass = passWhenUnsupported && (ret == STATUS_PARAM_INVALID);
        LOG_PRINT("Test case: %s\n", name.c_str());
        LOG_PRINT("  GetWorkspaceSize failed. ERROR: %d\n", static_cast<int>(ret));
        if (!note.empty()) {
            LOG_PRINT("  Note: %s\n", note.c_str());
        }
        LOG_PRINT("  [%s]\n\n", pass ? "PASS" : "FAIL");
        return pass;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] workspace malloc failed ret=%d\n", name.c_str(), ret); return false);
    }

    ret = useV2 ? aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream)
                : aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
    }
    CHECK_RET(ret == STATUS_SUCCESS, LOG_PRINT("[%s] execute failed ret=%d\n", name.c_str(), ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] stream sync failed ret=%d\n", name.c_str(), ret); return false);

    const int64_t elemCount = GetShapeSize(shape);
    std::vector<OutT> resultData(static_cast<size_t>(elemCount), static_cast<OutT>(0));
    if (out.bytes > 0) {
        ret = aclrtMemcpy(resultData.data(), out.bytes, out.deviceAddr, out.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] D2H failed ret=%d\n", name.c_str(), ret); return false);
    }

    const auto inputDecoded = decodeInput(inputHostData);
    const auto expected = CumsumRef<double>(inputDecoded, shape, dim, useV2 ? exclusive : false, useV2 ? reverse : false);
    const auto actual = decodeOutput(resultData);

    const auto error = CalcError(actual, expected);
    double maxAbsError = 0.0;
    int64_t maxErrorIndex = -1;
    bool ok = CompareWithTolerance(actual, expected, atol, rtol, exact, maxAbsError, maxErrorIndex);
    PrintCaseResult(name, expected, actual, error, ok, note);
    return ok;
}

template <typename T>
bool RunPrecisionTypedCase(const std::string& name, const std::vector<int64_t>& shape,
                           const std::vector<double>& inputValues, aclDataType dataType, int64_t dim,
                           bool useV2, bool exclusive, bool reverse,
                           double atol, double rtol, bool exact, aclrtStream stream,
                           const std::string& note = "")
{
    return RunPrecisionCumsumCase<T, T>(
        name, shape, inputValues, dataType, dataType, dim, useV2, exclusive, reverse, atol, rtol, exact,
        [](const std::vector<double>& values) { return EncodeVector<T>(values); },
        [](const std::vector<T>& values) { return DecodeVector<T>(values); },
        [](const std::vector<T>& values) { return DecodeVector<T>(values); },
        stream, false, note);
}

bool RunPrecisionBf16Case(const std::string& name, const std::vector<int64_t>& shape,
                          const std::vector<double>& inputValues, int64_t dim,
                          double atol, double rtol, aclrtStream stream,
                          const std::string& note = "")
{
    return RunPrecisionCumsumCase<uint16_t, uint16_t>(
        name, shape, inputValues, ACL_BF16, ACL_BF16, dim, false, false, false, atol, rtol, false,
        [](const std::vector<double>& values) { return EncodeBf16Vector(values); },
        [](const std::vector<uint16_t>& values) { return DecodeBf16Vector(values); },
        [](const std::vector<uint16_t>& values) { return DecodeBf16Vector(values); },
        stream, true, note);
}

std::vector<double> Repeat(double value, size_t count)
{
    return std::vector<double>(count, value);
}

std::vector<double> AlternatingMagnitude(size_t pairs)
{
    std::vector<double> values;
    values.reserve(pairs * 2);
    for (size_t i = 0; i < pairs; ++i) {
        values.push_back(1.0e8);
        values.push_back(1.0e-6);
    }
    return values;
}

std::vector<double> AlternatingSign(size_t count)
{
    std::vector<double> values(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        values[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
    return values;
}

std::vector<double> RepeatPattern(const std::vector<double>& pattern, size_t repeats)
{
    std::vector<double> values;
    values.reserve(pattern.size() * repeats);
    for (size_t i = 0; i < repeats; ++i) {
        values.insert(values.end(), pattern.begin(), pattern.end());
    }
    return values;
}

std::vector<double> SwallowRecoveryPattern(size_t repeats)
{
    return RepeatPattern({1.0e8, 1.0, -1.0e8}, repeats);
}

std::vector<double> FrontLoadedLargeThenOnes(double large, size_t ones)
{
    std::vector<double> values;
    values.reserve(ones + 1);
    values.push_back(large);
    for (size_t i = 0; i < ones; ++i) {
        values.push_back(1.0);
    }
    return values;
}

std::vector<double> BackLoadedLargeWithOnes(double large, size_t ones)
{
    std::vector<double> values(ones, 1.0);
    values.push_back(large);
    return values;
}

bool RunInt32WrapCumsumCase(const std::string& name, const std::vector<int64_t>& shape,
                            const std::vector<int32_t>& inputValues, int64_t dim, aclrtStream stream,
                            const std::string& note = "")
{
    TensorHolder self;
    TensorHolder out;
    auto ret = CreateAclTensor(shape, ACL_INT32, inputValues.data(), self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(shape, ACL_INT32, nullptr, out);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self.tensor, dim, ACL_INT32, out.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == STATUS_SUCCESS, LOG_PRINT("[%s] GetWorkspace ret=%d\n", name.c_str(), ret); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] workspace malloc ret=%d\n", name.c_str(), ret); return false);
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
    }
    CHECK_RET(ret == STATUS_SUCCESS, LOG_PRINT("[%s] execute ret=%d\n", name.c_str(), ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] stream sync ret=%d\n", name.c_str(), ret); return false);

    const int64_t elemCount = GetShapeSize(shape);
    std::vector<int32_t> resultData(static_cast<size_t>(elemCount), 0);
    if (out.bytes > 0) {
        ret = aclrtMemcpy(resultData.data(), out.bytes, out.deviceAddr, out.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] D2H ret=%d\n", name.c_str(), ret); return false);
    }

    std::vector<double> expected(static_cast<size_t>(elemCount), 0.0);
    if (!shape.empty()) {
        int64_t rank = static_cast<int64_t>(shape.size());
        int64_t realDim = dim;
        if (realDim < 0) {
            realDim += rank;
        }
        int64_t axis = shape[realDim];
        int64_t outer = 1;
        int64_t inner = 1;
        for (int64_t i = 0; i < realDim; ++i) {
            outer *= shape[i];
        }
        for (int64_t i = realDim + 1; i < rank; ++i) {
            inner *= shape[i];
        }
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t in = 0; in < inner; ++in) {
                uint32_t acc = 0;
                for (int64_t a = 0; a < axis; ++a) {
                    int64_t idx = o * axis * inner + a * inner + in;
                    acc += static_cast<uint32_t>(inputValues[static_cast<size_t>(idx)]);
                    expected[static_cast<size_t>(idx)] = static_cast<double>(static_cast<int32_t>(acc));
                }
            }
        }
    }

    auto actual = DecodeVector<int32_t>(resultData);
    const auto error = CalcError(actual, expected);
    double maxAbsError = 0.0;
    int64_t maxErrorIndex = -1;
    bool ok = CompareWithTolerance(actual, expected, 0.0, 0.0, true, maxAbsError, maxErrorIndex);
    PrintCaseResult(name, expected, actual, error, ok, note);
    return ok;
}

aclnnStatus CallGetWorkspace(const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
}

aclnnStatus CallGetWorkspaceV2(const aclTensor* self, int64_t dim, bool exclusive, bool reverse, aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int passCnt = 0;
    int totalCnt = 0;
    int caseNo = 0;
    auto Run = [&](const std::string& caseDesc, const std::function<bool()>& fn) {
        ++totalCnt;
        ++caseNo;
        LOG_PRINT("\nTest case %d: %s\n", caseNo, caseDesc.c_str());
        bool ok = fn();
        LOG_PRINT("  [%s]\n", ok ? "PASS" : "FAIL");
        if (ok) {
            ++passCnt;
        }
    };

    // [Case 1-8] 基础功能与属性组合
    Run("Basic Cumsum float32 dim=0", [&]() {
        return RunCumsumCase<float>("Cumsum_FLOAT32_dim0_2x2", {1.f, 2.f, 3.f, 4.f}, {2, 2}, 0, ACL_FLOAT, stream);
    });
    Run("Cumsum float32 negative axis", [&]() {
        return RunCumsumCase<float>("Cumsum_FLOAT32_negative_dim", {1.f, -2.f, 0.5f, 3.f, 4.f, -5.f},
                                    {2, 3}, -1, ACL_FLOAT, stream);
    });
    Run("Cumsum float32 long sequence", [&]() {
        std::vector<float> longSeq(2048, 1.f);
        return RunCumsumCase<float>("Cumsum_FLOAT32_long_sequence", longSeq, {1, 2048}, 1, ACL_FLOAT, stream);
    });
    Run("Cumsum int32 axis=1", [&]() {
        return RunCumsumCase<int32_t>("Cumsum_INT32_dim1", {1, 2, 3, 4, 5, 6}, {2, 3}, 1, ACL_INT32, stream);
    });
    Run("CumsumV2 EF_RF", [&]() {
        return RunCumsumV2Case("CumsumV2_exclusive_false_reverse_false", {1.f, 2.f, 3.f, 4.f}, {2, 2}, 1,
                               false, false, stream);
    });
    Run("CumsumV2 ET_RF", [&]() {
        return RunCumsumV2Case("CumsumV2_exclusive_true_reverse_false", {1.f, 2.f, 3.f, 4.f}, {2, 2}, 1,
                               true, false, stream);
    });
    Run("CumsumV2 EF_RT", [&]() {
        return RunCumsumV2Case("CumsumV2_exclusive_false_reverse_true", {1.f, 2.f, 3.f, 4.f}, {2, 2}, 1,
                               false, true, stream);
    });
    Run("CumsumV2 ET_RT", [&]() {
        return RunCumsumV2Case("CumsumV2_exclusive_true_reverse_true", {1.f, 2.f, 3.f, 4.f}, {2, 2}, 1,
                               true, true, stream);
    });

    // 精度导向用例（来自 test_aclnn_cumsum_precision.cpp）
    Run("Precision float32 decimal accumulation", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_0.1_len10000", {10000}, Repeat(0.1, 10000),
                                            ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream);
    });
    Run("Precision float32 mixed magnitude", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_mixed_magnitude", {2000}, AlternatingMagnitude(1000),
                                            ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream);
    });
    Run("Precision float16 decimal accumulation", [&]() {
        return RunPrecisionTypedCase<uint16_t>("Precision_float16_0.1_len2048", {2048}, Repeat(0.1, 2048),
                                               ACL_FLOAT16, 0, false, false, false, 1e-3, 1e-3, false, stream);
    });
    Run("Precision bf16 accumulation", [&]() {
        return RunPrecisionBf16Case("Precision_bf16_0.25_len256", {256}, Repeat(0.25, 256), 0, 8e-2, 8e-2, stream);
    });
    Run("Precision int32 exact", [&]() {
        return RunPrecisionTypedCase<int32_t>("Precision_int32_exact", {2, 4},
                                              {1, 2, 3, 4, -10, 20, -30, 40}, ACL_INT32, 1,
                                              false, false, false, 0.0, 0.0, true, stream);
    });
    Run("Precision int64 exact", [&]() {
        return RunPrecisionTypedCase<int64_t>("Precision_int64_exact", {5},
                                              {1000000000.0, -2.0, 3.0, -4.0, 5.0}, ACL_INT64, 0,
                                              false, false, false, 0.0, 0.0, true, stream);
    });
    Run("Precision dtype conversion int32->float", [&]() {
        return RunPrecisionCumsumCase<int32_t, float>(
            "Precision_int32_to_float32", {2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32, ACL_FLOAT,
            1, false, false, false, 1e-5, 1e-5, false,
            [](const std::vector<double>& values) { return EncodeVector<int32_t>(values); },
            [](const std::vector<int32_t>& values) { return DecodeVector<int32_t>(values); },
            [](const std::vector<float>& values) { return DecodeVector<float>(values); },
            stream, false);
    });
    Run("Precision V2 exclusive", [&]() {
        return RunPrecisionTypedCase<float>("Precision_v2_exclusive", {2, 4}, {1, 2, 3, 4, 10, 20, 30, 40},
                                            ACL_FLOAT, 1, true, true, false, 1e-5, 1e-5, false, stream);
    });
    Run("Precision V2 reverse", [&]() {
        return RunPrecisionTypedCase<float>("Precision_v2_reverse", {2, 4}, {1, 2, 3, 4, 10, 20, 30, 40},
                                            ACL_FLOAT, 1, true, false, true, 1e-5, 1e-5, false, stream);
    });
    Run("Precision V2 exclusive+reverse cancellation", [&]() {
        return RunPrecisionTypedCase<float>("Precision_v2_exclusive_reverse_cancellation", {8}, AlternatingSign(8),
                                            ACL_FLOAT, 0, true, true, true, 1e-5, 1e-5, false, stream);
    });

    Run("Precision float32 decimal accumulation variant (0.2)", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_0.2_len4096", {4096}, Repeat(0.2, 4096),
                                            ACL_FLOAT, 0, false, false, false, 5e-3, 5e-3, false, stream);
    });
    Run("Precision float32 decimal accumulation variant (0.01)", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_0.01_len8192", {8192}, Repeat(0.01, 8192),
                                            ACL_FLOAT, 0, false, false, false, 5e-3, 5e-3, false, stream);
    });
    Run("Precision float32 near-1 tiny differences", [&]() {
        std::vector<double> values(4096, 1.0);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = (i % 2 == 0) ? (1.0 + 1e-6) : (1.0 - 1e-6);
        }
        return RunPrecisionTypedCase<float>("Precision_float32_near_one_tiny_diff", {4096}, values,
                                            ACL_FLOAT, 0, false, false, false, 1e-4, 1e-4, false, stream);
    });
    Run("Precision float32 scale-disparity pattern", [&]() {
        std::vector<double> values;
        values.reserve(4096);
        for (int i = 0; i < 1024; ++i) {
            values.push_back(1.0e12);
            values.push_back(1.0);
            values.push_back(-1.0e12);
            values.push_back(1.0);
        }
        return RunPrecisionTypedCase<float>("Precision_float32_scale_disparity", {4096}, values,
                                            ACL_FLOAT, 0, false, false, false, 1e-2, 1e-2, false, stream);
    });
    Run("Precision float32 cancellation with tiny bias", [&]() {
        std::vector<double> values;
        values.reserve(4096);
        for (int i = 0; i < 2048; ++i) {
            values.push_back(1.0e4);
            values.push_back(-1.0e4 + 1.0e-3);
        }
        return RunPrecisionTypedCase<float>("Precision_float32_cancellation_tiny_bias", {4096}, values,
                                            ACL_FLOAT, 0, false, false, false, 1e-2, 1e-2, false, stream);
    });
    Run("Precision float16 near-1 tiny differences", [&]() {
        std::vector<double> values(2048, 1.0);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = (i % 2 == 0) ? (1.0 + 1e-3) : (1.0 - 1e-3);
        }
        return RunPrecisionTypedCase<uint16_t>("Precision_float16_near_one_tiny_diff", {2048}, values,
                                               ACL_FLOAT16, 0, false, false, false, 5e-2, 5e-2, false, stream);
    });
    Run("Precision float16 decimal accumulation variant (0.2)", [&]() {
        return RunPrecisionTypedCase<uint16_t>("Precision_float16_0.2_len2048", {2048}, Repeat(0.2, 2048),
                                               ACL_FLOAT16, 0, false, false, false, 5e-2, 5e-2, false, stream);
    });
    Run("Precision V2 exclusive decimal accumulation", [&]() {
        return RunPrecisionTypedCase<float>("Precision_v2_exclusive_decimal_0.1", {4096}, Repeat(0.1, 4096),
                                            ACL_FLOAT, 0, true, true, false, 5e-3, 5e-3, false, stream);
    });
    Run("Precision V2 reverse decimal accumulation", [&]() {
        return RunPrecisionTypedCase<float>("Precision_v2_reverse_decimal_0.1", {4096}, Repeat(0.1, 4096),
                                            ACL_FLOAT, 0, true, false, true, 5e-3, 5e-3, false, stream);
    });
    Run("Precision V2 exclusive+reverse scale-disparity", [&]() {
        std::vector<double> values;
        values.reserve(2048);
        for (int i = 0; i < 512; ++i) {
            values.push_back(1.0e8);
            values.push_back(1.0e-4);
            values.push_back(-1.0e8);
            values.push_back(-1.0e-4);
        }
        return RunPrecisionTypedCase<float>("Precision_v2_exclusive_reverse_scale_disparity", {2048}, values,
                                            ACL_FLOAT, 0, true, true, true, 1e-2, 1e-2, false, stream);
    });

    Run("Precision float32 subnormal accumulation", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_subnormal_1e40", {512}, Repeat(1.0e-40, 512),
                                            ACL_FLOAT, 0, false, false, false, 1e-44, 1e-5, false, stream);
    });
    Run("Precision float32 overflow accumulation", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_overflow_1e38", {4}, Repeat(1.0e38, 4),
                                            ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream);
    });
    Run("Precision float32 cancellation after swallow", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_swallow_recovery", {192}, SwallowRecoveryPattern(64),
                                            ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream);
    });
    Run("Precision float32 ULP plateau near 2^24", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_ulp_plateau_2pow24", {17},
                                            FrontLoadedLargeThenOnes(16777216.0, 16),
                                            ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream);
    });
    Run("Precision float32 order sensitivity near 2^24", [&]() {
        return RunPrecisionTypedCase<float>("Precision_float32_order_sensitivity_2pow24", {17},
                                            BackLoadedLargeWithOnes(16777216.0, 16),
                                            ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream);
    });
    Run("Precision float16 ULP plateau near 2048", [&]() {
        return RunPrecisionTypedCase<uint16_t>("Precision_float16_ulp_plateau_2048", {17},
                                               FrontLoadedLargeThenOnes(2048.0, 16),
                                               ACL_FLOAT16, 0, false, false, false, 1e-3, 1e-3, false, stream);
    });
    Run("Precision float16 overflow accumulation", [&]() {
        return RunPrecisionTypedCase<uint16_t>("Precision_float16_overflow_4000", {40}, Repeat(4000.0, 40),
                                               ACL_FLOAT16, 0, false, false, false, 1e-3, 1e-3, false, stream);
    });
    Run("Precision bf16 decimal accumulation", [&]() {
        return RunPrecisionBf16Case("Precision_bf16_decimal_0.1", {512}, Repeat(0.1, 512), 0, 1e-1, 1e-1, stream);
    });
    Run("Precision bf16 ULP plateau near 256", [&]() {
        return RunPrecisionBf16Case("Precision_bf16_ulp_plateau_256", {17}, FrontLoadedLargeThenOnes(256.0, 16),
                                    0, 1e-1, 1e-1, stream);
    });
    Run("Precision bf16 cancellation after swallow", [&]() {
        return RunPrecisionBf16Case("Precision_bf16_swallow_recovery", {96},
                                    RepeatPattern({256.0, 1.0, -256.0}, 32),
                                    0, 1e-1, 1e-1, stream);
    });
    Run("Precision int32 overflow wraparound", [&]() {
        return RunInt32WrapCumsumCase("Precision_int32_overflow_wraparound", {4}, {2147483647, 1, 1, 1},
                                      0, stream);
    });
    Run("Precision V2 reverse mixed magnitude", [&]() {
        return RunPrecisionTypedCase<float>("Precision_v2_reverse_swallow_recovery", {192}, SwallowRecoveryPattern(64),
                                            ACL_FLOAT, 0, true, false, true, 1e-5, 1e-5, false, stream);
    });
    Run("Precision V2 exclusive shifted plateau", [&]() {
        return RunPrecisionTypedCase<float>("Precision_v2_exclusive_plateau_2pow24", {17},
                                            FrontLoadedLargeThenOnes(16777216.0, 16),
                                            ACL_FLOAT, 0, true, true, false, 1e-5, 1e-5, false, stream);
    });

    const std::vector<aclDataType> documentedDtypes = {
        ACL_INT8,     ACL_INT16,     ACL_INT32,     ACL_INT64,      ACL_UINT8,
        ACL_UINT16,   ACL_UINT32,    ACL_UINT64,    ACL_FLOAT16,    ACL_FLOAT,
        ACL_DOUBLE,   ACL_COMPLEX64, ACL_COMPLEX128
    };
    const std::vector<int64_t> axisCases = {0, 1, -1};

    // [Case 42-80] 文档 dtype + axis 组合（Cumsum）
    for (auto dtype : documentedDtypes) {
        bool expectedLikelySupport =
            (dtype == ACL_INT8 || dtype == ACL_INT16 || dtype == ACL_INT32 || dtype == ACL_INT64 || dtype == ACL_UINT8 ||
             dtype == ACL_FLOAT16 || dtype == ACL_FLOAT || dtype == ACL_DOUBLE || dtype == ACL_COMPLEX64 ||
             dtype == ACL_COMPLEX128);

        for (auto dim : axisCases) {
            std::string name = std::string("Doc_Cumsum_dtype_") + GetDataTypeName(dtype) + "_axis_" +
                               std::to_string(static_cast<long long>(dim));
            Run(name, [&]() {
                if (expectedLikelySupport) {
                    return RunCumsumWorkspaceStatusCase(name, {2, 3}, dim, dtype, dtype,
                                                        {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
                }
                return RunCumsumWorkspaceStatusCase(name, {2, 3}, dim, dtype, dtype, {STATUS_PARAM_INVALID});
            });
        }
    }

    // [Case 81-93] 文档 dtype 组合（CumsumV2，EF_RF）
    for (auto dtype : documentedDtypes) {
        bool expectedLikelySupport =
            (dtype == ACL_INT8 || dtype == ACL_INT16 || dtype == ACL_INT32 || dtype == ACL_INT64 || dtype == ACL_UINT8 ||
             dtype == ACL_FLOAT16 || dtype == ACL_FLOAT || dtype == ACL_DOUBLE || dtype == ACL_COMPLEX64 ||
             dtype == ACL_COMPLEX128);

        std::string name = std::string("Doc_CumsumV2_dtype_") + GetDataTypeName(dtype) + "_EF_RF";
        Run(name, [&]() {
            if (expectedLikelySupport) {
                return RunCumsumV2WorkspaceStatusCase(name, {2, 3}, 1, false, false, dtype, dtype,
                                                      {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
            }
            return RunCumsumV2WorkspaceStatusCase(name, {2, 3}, 1, false, false, dtype, dtype,
                                                  {STATUS_PARAM_INVALID});
        });
    }

    const std::vector<std::pair<bool, bool>> attrPairs = {
        {false, false}, {true, false}, {false, true}, {true, true}
    };
    // [Case 94-97] 文档属性组合（exclusive/reverse）
    for (auto pair : attrPairs) {
        std::string name = std::string("Doc_CumsumV2_FLOAT_attr_") + (pair.first ? "ET" : "EF") + "_" +
                           (pair.second ? "RT" : "RF");
        Run(name, [&]() {
            return RunCumsumV2WorkspaceStatusCase(name, {2, 3}, 1, pair.first, pair.second, ACL_FLOAT, ACL_FLOAT,
                                                  {STATUS_SUCCESS});
        });
    }

    // arch35 host tiling 定向覆盖（float/int）
    struct TilingTargetCase {
        const char* name;
        std::vector<int64_t> shape;
        int64_t dim;
        aclDataType dtype;
        bool useV2;
        bool exclusive;
        bool reverse;
    };

    const std::vector<TilingTargetCase> arch35TilingCases = {
        // float tiling: N<CL / N>=CL, R全载/非全载, dtCast(fp16), axis正负与首末轴
        {"Arch35_float_small_n_small_r_axis1", {1, 2, 2}, 1, ACL_FLOAT, false, false, false},
        {"Arch35_float_small_n_large_r_axis1", {8, 64, 2}, 1, ACL_FLOAT, false, false, false},
        {"Arch35_float_large_n_mid_r_axis1", {64, 8, 64}, 1, ACL_FLOAT, false, false, false},
        {"Arch35_float_large_n_large_r_axis1", {2, 4096, 64}, 1, ACL_FLOAT, false, false, false},
        {"Arch35_float_borrow_n_axis1", {1, 4, 2048}, 1, ACL_FLOAT, false, false, false},
        {"Arch35_float_negative_axis", {4, 32, 17}, -1, ACL_FLOAT, false, false, false},
        {"Arch35_fp16_dtcast_axis1", {4, 512, 8}, 1, ACL_FLOAT16, false, false, false},
        {"Arch35_fp16_v2_attr_path", {2, 256, 16}, 1, ACL_FLOAT16, true, true, true},

        // int tiling: 覆盖axis分解与不同rightAxisLen场景，逼近11000/11001/11002路径
        {"Arch35_int32_axis0_large_right", {64, 4, 128}, 0, ACL_INT32, false, false, false},
        {"Arch35_int32_axis_last_small_right", {32, 64, 2}, -1, ACL_INT32, false, false, false},
        {"Arch35_int64_midaxis_group_like", {2, 2048, 2}, 1, ACL_INT64, false, false, false},
        {"Arch35_int8_midaxis_ra_heavy", {8, 128, 256}, 1, ACL_INT8, false, false, false},
        {"Arch35_uint8_midaxis_ra_light", {64, 32, 8}, 1, ACL_UINT8, false, false, false},
        {"Arch35_uint64_v2_attr_path", {4, 256, 16}, 1, ACL_UINT64, true, true, false},
    };

    for (const auto& tc : arch35TilingCases) {
        Run(tc.name, [&]() {
            if (tc.useV2) {
                return RunCumsumV2WorkspaceStatusCase(tc.name, tc.shape, tc.dim, tc.exclusive, tc.reverse, tc.dtype,
                                                      tc.dtype,
                                                      {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
            }
            return RunCumsumWorkspaceStatusCase(tc.name, tc.shape, tc.dim, tc.dtype, tc.dtype,
                                                {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
        });
    }

    // arch35 float tiling 分支命中覆盖（针对未覆盖行）
    const std::vector<TilingTargetCase> arch35FloatBranchHitCases = {
        // 目标: L438-L450 NGreaterClRFullLoad 中 N不能全载 分支
        {"Arch35_hit_NGreaterClRFullLoad_N_not_full", {64, 32, 4096}, 1, ACL_FLOAT, false, false, false},

        // 目标: L484-L500 NGreaterClRNotFullLoad 首分支(M > core/2)
        {"Arch35_hit_NGreaterClRNotFullLoad_M_enough", {40, 4096, 64}, 1, ACL_FLOAT, false, false, false},

        // 目标: L508-L525 NGreaterClRNotFullLoad 中 借N后够分核 分支
        {"Arch35_hit_NGreaterClRNotFullLoad_borrowN_enough", {1, 4096, 1024}, 1, ACL_FLOAT, false, false, false},

        // 目标: L565-L570 NGreaterClRNotFullLoadBorrowR 中 R不能全载 分支
        {"Arch35_hit_NGreaterClRNotFullLoadBorrowR_R_not_full", {1, 300000, 16}, 1, ACL_FLOAT, false, false, false},

        // 目标: L620-L679 CalcBorrowM + RNGreaterClBorrowM
        {"Arch35_hit_CalcBorrowM_RNGreaterClBorrowM", {96, 20000, 8}, 1, ACL_FLOAT, false, false, false},

        // 目标: L709-L862 RNGreaterClRNotFullLoad 家族函数入口(不借轴分支)
        {"Arch35_hit_RNGreaterClRNotFullLoad_notBorrowR", {48, 20000, 8}, 1, ACL_FLOAT, false, false, false},

        // 目标: L709-L862 RNGreaterClRNotFullLoad 家族函数入口(借R分支)
        {"Arch35_hit_RNGreaterClRNotFullLoad_borrowR", {1, 20000, 8}, 1, ACL_FLOAT, false, false, false},

        // 辅助: 负轴触发同路径，避免轴归一化导致分支偏差
        {"Arch35_hit_RNGreaterCl_negative_axis", {2, 30000, 8}, -2, ACL_FLOAT, false, false, false},
    };

    for (const auto& tc : arch35FloatBranchHitCases) {
        Run(tc.name, [&]() {
            return RunCumsumWorkspaceStatusCase(tc.name, tc.shape, tc.dim, tc.dtype, tc.dtype,
                                                {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
        });
    }

    // op_api 定向覆盖（aclnn_cumsum.cpp 未覆盖行）
    Run("OpApi_V2_empty_tensor_workspace_zero_path", [&]() {
        return RunCumsumV2WorkspaceStatusCase("OpApi_V2_empty_tensor_workspace_zero_path", {2, 0}, 1, false, false,
                                              ACL_FLOAT, ACL_FLOAT,
                                              {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });

    Run("OpApi_V2_dim_zero_int64_dim_tensor_path", [&]() {
        return RunCumsumV2WorkspaceStatusCase("OpApi_V2_dim_zero_int64_dim_tensor_path", {2, 3}, 0, true, false,
                                              ACL_FLOAT, ACL_FLOAT,
                                              {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });

    Run("OpApi_Cube_path_probe_shape_12800x512_fp16", [&]() {
        return RunCumsumWorkspaceStatusCase("OpApi_Cube_path_probe_shape_12800x512_fp16", {12800, 512}, 1,
                                            ACL_FLOAT16, ACL_FLOAT16,
                                            {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });

    Run("OpApi_platform_support_probe_v1", [&]() {
        return RunCumsumWorkspaceStatusCase("OpApi_platform_support_probe_v1", {2, 3}, 1, ACL_FLOAT, ACL_FLOAT,
                                            {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });

    Run("OpApi_platform_support_probe_v2", [&]() {
        return RunCumsumV2WorkspaceStatusCase("OpApi_platform_support_probe_v2", {2, 3}, 1, false, false, ACL_FLOAT,
                                              ACL_FLOAT,
                                              {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });

    // V2 dtype组合：针对 CheckDtypeValidWithoutDtype 中 out dtype 支持表分支展开
    const std::vector<aclDataType> v2OutDtypeProbes = {
        ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_INT32, ACL_INT64, ACL_INT8, ACL_UINT8,
        ACL_DOUBLE, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_COMPLEX64, ACL_COMPLEX128, ACL_BOOL
    };
    for (auto outType : v2OutDtypeProbes) {
        std::string name = std::string("OpApi_V2_out_dtype_probe_") + GetDataTypeName(outType);
        Run(name, [&]() {
            return RunCumsumV2WorkspaceStatusCase(name, {2, 3}, 1, false, false, ACL_FLOAT, outType,
                                                  {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
        });
    }

    // op_api/cumsum.cpp 定向覆盖（IsAiCoreSupport与AllocTensor失败探针）
    Run("OpApi_IsAiCoreSupport_branch_probe_float_v1", [&]() {
        return RunCumsumWorkspaceStatusCase("OpApi_IsAiCoreSupport_branch_probe_float_v1", {4, 8}, 1,
                                            ACL_FLOAT, ACL_FLOAT,
                                            {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });
    Run("OpApi_IsAiCoreSupport_branch_probe_float_v2", [&]() {
        return RunCumsumV2WorkspaceStatusCase("OpApi_IsAiCoreSupport_branch_probe_float_v2", {4, 8}, 1,
                                              false, false, ACL_FLOAT, ACL_FLOAT,
                                              {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });
    Run("OpApi_AllocOutFail_probe_v1", [&]() {
        return RunCumsumWorkspaceStatusCaseWithDeviceMemoryPressure(
            "OpApi_AllocOutFail_probe_v1", {2048, 2048}, 1, ACL_FLOAT, ACL_FLOAT,
            {STATUS_INNER_NULLPTR, STATUS_SUCCESS, STATUS_PARAM_INVALID}, false);
    });
    Run("OpApi_AllocOutFail_probe_v2", [&]() {
        return RunCumsumWorkspaceStatusCaseWithDeviceMemoryPressure(
            "OpApi_AllocOutFail_probe_v2", {2048, 2048}, 1, ACL_FLOAT, ACL_FLOAT,
            {STATUS_INNER_NULLPTR, STATUS_SUCCESS, STATUS_PARAM_INVALID}, true, false, false);
    });
    Run("OpApi_V1_dim_zero_int64_dim_tensor_path", [&]() {
        return RunCumsumWorkspaceStatusCase("OpApi_V1_dim_zero_int64_dim_tensor_path", {2, 3}, 0,
                                            ACL_FLOAT, ACL_FLOAT,
                                            {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
    });
    Run("OpApi_V1_internal_pipeline_pressure_probe", [&]() {
        return RunCumsumWorkspaceStatusCaseWithDeviceMemoryPressure(
            "OpApi_V1_internal_pipeline_pressure_probe", {4096, 1024}, 1, ACL_FLOAT, ACL_FLOAT,
            {STATUS_INNER_NULLPTR, STATUS_SUCCESS, STATUS_PARAM_INVALID}, false);
    });
    Run("OpApi_V2_internal_pipeline_pressure_probe", [&]() {
        return RunCumsumWorkspaceStatusCaseWithDeviceMemoryPressure(
            "OpApi_V2_internal_pipeline_pressure_probe", {4096, 1024}, 1, ACL_FLOAT, ACL_FLOAT,
            {STATUS_INNER_NULLPTR, STATUS_SUCCESS, STATUS_PARAM_INVALID}, true, true, true);
    });

    // arch35 int tiling 分支命中覆盖（cumsum_tiling_ascendc_int_arch35.cpp）
    const std::vector<TilingTargetCase> arch35IntBranchHitCases = {
        // 目标: L116-L118 AdjustTensor4TDRA 中 split cores on R 后直接return
        {"Arch35_int_hit_AdjustTensor4TDRA_splitR", {1, 30000, 64}, 1, ACL_INT32, false, false, false},

        // 目标: L155-L156 AdjustTensor4TDR 中 split cores on R
        {"Arch35_int_hit_AdjustTensor4TDR_splitR", {1, 60000, 1}, 1, ACL_INT32, false, false, false},

        // 目标: L177-L185 AdjustLARLpUnit 分支
        {"Arch35_int_hit_AdjustLARLpUnit", {256, 1024, 8}, 1, ACL_INT8, false, false, false},

        // 目标: L252/L257 CalcAxisWeight 可整除分支与返回
        {"Arch35_int_hit_CalcAxisWeight_divisible", {64, 4096, 64}, 1, ACL_INT32, false, false, false},

        // 目标: L295-L304 + L311 + L353 R轴分核组网路径(CUM_WITH_GROUP + SetScheduleMode)
        {"Arch35_int_hit_RBlockAxis_group_schedule", {1, 50000, 1}, 1, ACL_INT64, false, false, false},

        // 目标: L383-L407 PrintTilingData
        {"Arch35_int_hit_PrintTilingData", {32, 1024, 16}, 1, ACL_INT32, false, false, false},
    };

    for (const auto& tc : arch35IntBranchHitCases) {
        Run(tc.name, [&]() {
            return RunCumsumWorkspaceStatusCase(tc.name, tc.shape, tc.dim, tc.dtype, tc.dtype,
                                                {STATUS_SUCCESS, STATUS_PARAM_INVALID, STATUS_INNER_NULLPTR});
        });
    }

    // 异常与边界场景
    Run("abnormal null self", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<float> in(4, 1.f);
        if (CreateAclTensor({2, 2}, ACL_FLOAT, in.data(), self) != ACL_SUCCESS ||
            CreateAclTensor({2, 2}, ACL_FLOAT, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_abnormal_null_self", STATUS_PARAM_NULLPTR,
                             [&]() { return CallGetWorkspace(nullptr, 0, ACL_FLOAT, out.tensor); });
    });
    Run("abnormal null out", [&]() {
        TensorHolder self;
        std::vector<float> in(4, 1.f);
        if (CreateAclTensor({2, 2}, ACL_FLOAT, in.data(), self) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_abnormal_null_out", STATUS_PARAM_NULLPTR,
                             [&]() { return CallGetWorkspace(self.tensor, 0, ACL_FLOAT, nullptr); });
    });
    Run("abnormal shape mismatch", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<float> in(4, 1.f);
        if (CreateAclTensor({2, 2}, ACL_FLOAT, in.data(), self) != ACL_SUCCESS ||
            CreateAclTensor({2, 3}, ACL_FLOAT, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_abnormal_shape_mismatch", STATUS_PARAM_INVALID,
                             [&]() { return CallGetWorkspace(self.tensor, 0, ACL_FLOAT, out.tensor); });
    });
    Run("abnormal axis out of range", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<float> in(4, 1.f);
        if (CreateAclTensor({2, 2}, ACL_FLOAT, in.data(), self) != ACL_SUCCESS ||
            CreateAclTensor({2, 2}, ACL_FLOAT, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_abnormal_dim_out_of_range", STATUS_PARAM_INVALID,
                             [&]() { return CallGetWorkspace(self.tensor, 2, ACL_FLOAT, out.tensor); });
    });
    Run("abnormal dtype mismatch self/out", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<float> in(4, 1.f);
        if (CreateAclTensor({2, 2}, ACL_FLOAT, in.data(), self) != ACL_SUCCESS ||
            CreateAclTensor({2, 2}, ACL_FLOAT16, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_abnormal_dtype_not_match_out", STATUS_PARAM_INVALID,
                             [&]() { return CallGetWorkspace(self.tensor, 0, ACL_FLOAT, out.tensor); });
    });
    Run("abnormal bool unsupported", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<uint8_t> in(4, 1);
        if (CreateAclTensor({2, 2}, ACL_BOOL, in.data(), self) != ACL_SUCCESS ||
            CreateAclTensor({2, 2}, ACL_BOOL, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_abnormal_bool_not_supported", STATUS_PARAM_INVALID,
                             [&]() { return CallGetWorkspace(self.tensor, 0, ACL_BOOL, out.tensor); });
    });
    Run("scalar 0-dim tensor", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<float> scalar = {7.f};
        if (CreateAclTensor({}, ACL_FLOAT, scalar.data(), self) != ACL_SUCCESS ||
            CreateAclTensor({}, ACL_FLOAT, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_scalar_0dim", STATUS_SUCCESS,
                             [&]() { return CallGetWorkspace(self.tensor, 0, ACL_FLOAT, out.tensor); });
    });
    Run("empty tensor", [&]() {
        TensorHolder self;
        TensorHolder out;
        if (CreateAclTensor({2, 0}, ACL_FLOAT, nullptr, self) != ACL_SUCCESS ||
            CreateAclTensor({2, 0}, ACL_FLOAT, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_empty_tensor", STATUS_SUCCESS,
                             [&]() { return CallGetWorkspace(self.tensor, 0, ACL_FLOAT, out.tensor); });
    });
    Run("abnormal rank > 8", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<int64_t> shape10D(10, 2);
        std::vector<float> in(static_cast<size_t>(GetShapeSize(shape10D)), 1.f);
        if (CreateAclTensor(shape10D, ACL_FLOAT, in.data(), self) != ACL_SUCCESS ||
            CreateAclTensor(shape10D, ACL_FLOAT, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("Cumsum_abnormal_shape_rank_gt8", STATUS_PARAM_INVALID,
                             [&]() { return CallGetWorkspace(self.tensor, 0, ACL_FLOAT, out.tensor); });
    });
    Run("CumsumV2 abnormal self/out dtype mismatch", [&]() {
        TensorHolder self;
        TensorHolder out;
        std::vector<float> in(4, 1.f);
        if (CreateAclTensor({2, 2}, ACL_FLOAT, in.data(), self) != ACL_SUCCESS ||
            CreateAclTensor({2, 2}, ACL_INT32, nullptr, out) != ACL_SUCCESS) {
            return false;
        }
        return RunStatusCase("CumsumV2_abnormal_self_out_dtype_mismatch", STATUS_PARAM_INVALID,
                             [&]() { return CallGetWorkspaceV2(self.tensor, 0, false, false, out.tensor); });
    });

    int failCnt = totalCnt - passCnt;
    LOG_PRINT("\nSummary: %d passed, %d failed\n", passCnt, failCnt);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (passCnt == totalCnt) ? 0 : 1;
}
