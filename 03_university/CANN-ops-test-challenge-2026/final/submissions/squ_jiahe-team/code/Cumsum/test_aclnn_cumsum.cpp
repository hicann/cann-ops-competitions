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
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS ACL_SUCCESS
#endif

#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif

#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
        fflush(stdout);                 \
    } while (0)

namespace {

constexpr const char* kPrecisionSummaryLogPath = "/root/my_team/cumsum_precision_summary.log";

struct RuntimeContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
};

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
    size_t bytes = 0;
};

struct Tolerance {
    double atol = 0.0;
    double rtol = 0.0;
};

struct TestResult {
    std::string name;
    bool pass = false;
    bool isErrorCase = false;
    bool includeInSummary = false;
    std::string summaryTitle;
    std::string summaryNote;
    std::string expectedPreview;
    std::string actualPreview;
    double maxError = 0.0;
    size_t maxErrorIndex = 0;
};

enum class ApiKind {
    kCumsum,
    kCumsumV2,
};

struct SuccessCase {
    std::string name;
    ApiKind apiKind = ApiKind::kCumsum;
    std::vector<int64_t> shape;
    int64_t dim = 0;
    aclDataType selfType = ACL_FLOAT;
    aclDataType outType = ACL_FLOAT;
    bool exclusive = false;
    bool reverse = false;
    Tolerance tolerance{1e-5, 1e-5};
    std::vector<double> hostValues;
    std::string summaryTitle;
    std::string summaryNote;
    bool includeInSummary = false;
};

struct CompareStats {
    double maxDiff = 0.0;
    size_t maxIndex = 0;
    bool mismatchFound = false;
    size_t mismatchIndex = 0;
};

struct ErrorCase {
    std::string name;
    ApiKind apiKind = ApiKind::kCumsum;
    std::vector<int64_t> selfShape;
    std::vector<int64_t> outShape;
    int64_t dim = 0;
    aclDataType selfType = ACL_FLOAT;
    aclDataType outType = ACL_FLOAT;
    aclDataType dtype = ACL_FLOAT;
    bool exclusive = false;
    bool reverse = false;
    bool nullSelf = false;
    bool nullOut = false;
    aclnnStatus expectedStatus = ACLNN_SUCCESS;
    std::vector<double> selfValues;
    std::vector<double> outValues;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> BuildContiguousStrides(const std::vector<int64_t>& shape)
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

size_t GetTypeSize(aclDataType dataType)
{
    switch (dataType) {
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_DOUBLE:
            return sizeof(double);
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_INT8:
        case ACL_UINT8:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}

bool IsFloatType(aclDataType dataType)
{
    return dataType == ACL_FLOAT || dataType == ACL_DOUBLE || dataType == ACL_FLOAT16 || dataType == ACL_BF16;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000U;
    uint32_t mantissa = bits & 0x007FFFFFU;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x00800000U;
        const uint32_t shifted = mantissa >> static_cast<uint32_t>(1 - exponent);
        return static_cast<uint16_t>(sign | ((shifted + 0x00001000U) >> 13));
    }

    if (exponent >= 31) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        return static_cast<uint16_t>(sign | 0x7C00U | ((mantissa + 0x00001000U) >> 13));
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x00001000U) >> 13));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16;
    uint32_t exponent = (value >> 10) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
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
            mantissa &= 0x03FFU;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

double QuantizeToType(double value, aclDataType dataType)
{
    switch (dataType) {
        case ACL_FLOAT: {
            const float v = static_cast<float>(value);
            return static_cast<double>(v);
        }
        case ACL_DOUBLE:
            return value;
        case ACL_FLOAT16:
            return static_cast<double>(HalfToFloat(FloatToHalf(static_cast<float>(value))));
        case ACL_BF16:
            return static_cast<double>(BFloat16ToFloat(FloatToBFloat16(static_cast<float>(value))));
        case ACL_INT16:
            return static_cast<double>(static_cast<int16_t>(value));
        case ACL_INT32:
            return static_cast<double>(static_cast<int32_t>(value));
        case ACL_INT64:
            return static_cast<double>(static_cast<int64_t>(value));
        case ACL_INT8:
            return static_cast<double>(static_cast<int8_t>(value));
        case ACL_UINT8:
            return static_cast<double>(static_cast<uint8_t>(value));
        default:
            return value;
    }
}

void AppendEncodedValue(std::vector<uint8_t>* bytes, double value, aclDataType dataType)
{
    CHECK_RET(bytes != nullptr, return);

    switch (dataType) {
        case ACL_FLOAT: {
            const float v = static_cast<float>(value);
            const auto* ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes->insert(bytes->end(), ptr, ptr + sizeof(v));
            return;
        }
        case ACL_DOUBLE: {
            const double v = value;
            const auto* ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes->insert(bytes->end(), ptr, ptr + sizeof(v));
            return;
        }
        case ACL_FLOAT16: {
            const uint16_t v = FloatToHalf(static_cast<float>(value));
            const auto* ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes->insert(bytes->end(), ptr, ptr + sizeof(v));
            return;
        }
        case ACL_BF16: {
            const uint16_t v = FloatToBFloat16(static_cast<float>(value));
            const auto* ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes->insert(bytes->end(), ptr, ptr + sizeof(v));
            return;
        }
        case ACL_INT16: {
            const int16_t v = static_cast<int16_t>(value);
            const auto* ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes->insert(bytes->end(), ptr, ptr + sizeof(v));
            return;
        }
        case ACL_INT32: {
            const int32_t v = static_cast<int32_t>(value);
            const auto* ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes->insert(bytes->end(), ptr, ptr + sizeof(v));
            return;
        }
        case ACL_INT64: {
            const int64_t v = static_cast<int64_t>(value);
            const auto* ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes->insert(bytes->end(), ptr, ptr + sizeof(v));
            return;
        }
        case ACL_INT8: {
            const int8_t v = static_cast<int8_t>(value);
            bytes->push_back(static_cast<uint8_t>(v));
            return;
        }
        case ACL_UINT8: {
            const uint8_t v = static_cast<uint8_t>(value);
            bytes->push_back(v);
            return;
        }
        default:
            return;
    }
}

std::vector<uint8_t> EncodeValues(const std::vector<double>& values, aclDataType dataType)
{
    std::vector<uint8_t> bytes;
    bytes.reserve(values.size() * GetTypeSize(dataType));
    for (double value : values) {
        AppendEncodedValue(&bytes, value, dataType);
    }
    return bytes;
}

std::vector<double> DecodeValues(const std::vector<uint8_t>& bytes, aclDataType dataType)
{
    const size_t typeSize = GetTypeSize(dataType);
    std::vector<double> values;
    if (typeSize == 0) {
        return values;
    }

    const size_t count = bytes.size() / typeSize;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* ptr = bytes.data() + i * typeSize;
        switch (dataType) {
            case ACL_FLOAT: {
                float value = 0.0F;
                std::memcpy(&value, ptr, sizeof(value));
                values.push_back(static_cast<double>(value));
                break;
            }
            case ACL_DOUBLE: {
                double value = 0.0;
                std::memcpy(&value, ptr, sizeof(value));
                values.push_back(value);
                break;
            }
            case ACL_FLOAT16: {
                uint16_t value = 0;
                std::memcpy(&value, ptr, sizeof(value));
                values.push_back(static_cast<double>(HalfToFloat(value)));
                break;
            }
            case ACL_BF16: {
                uint16_t value = 0;
                std::memcpy(&value, ptr, sizeof(value));
                values.push_back(static_cast<double>(BFloat16ToFloat(value)));
                break;
            }
            case ACL_INT16: {
                int16_t value = 0;
                std::memcpy(&value, ptr, sizeof(value));
                values.push_back(static_cast<double>(value));
                break;
            }
            case ACL_INT32: {
                int32_t value = 0;
                std::memcpy(&value, ptr, sizeof(value));
                values.push_back(static_cast<double>(value));
                break;
            }
            case ACL_INT64: {
                int64_t value = 0;
                std::memcpy(&value, ptr, sizeof(value));
                values.push_back(static_cast<double>(value));
                break;
            }
            case ACL_INT8:
                values.push_back(static_cast<double>(static_cast<int8_t>(*ptr)));
                break;
            case ACL_UINT8:
                values.push_back(static_cast<double>(*ptr));
                break;
            default:
                break;
        }
    }
    return values;
}

std::vector<double> GenerateSequence(int64_t count, double start, double step)
{
    std::vector<double> values;
    values.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        values.push_back(start + step * static_cast<double>(i));
    }
    return values;
}

std::vector<double> GenerateAlternating(int64_t count, double positive, double negative)
{
    std::vector<double> values;
    values.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        values.push_back((i % 2 == 0) ? positive : negative);
    }
    return values;
}

std::vector<double> GenerateMixedMagnitude(int64_t count)
{
    std::vector<double> values;
    values.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        values.push_back((i % 2 == 0) ? 100000000.0 : 0.000001);
    }
    return values;
}

std::vector<double> GenerateConstant(int64_t count, double value)
{
    return std::vector<double>(static_cast<size_t>(count), value);
}

std::vector<double> GenerateTinyDecimal(int64_t count, double value)
{
    std::vector<double> values;
    values.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        values.push_back(value);
    }
    return values;
}

std::string FormatScalar(double value)
{
    std::ostringstream oss;
    const double absValue = std::fabs(value);
    if ((absValue >= 100000.0) || (absValue > 0.0 && absValue < 0.0001)) {
        oss << std::scientific << std::setprecision(6) << value;
    } else if (std::fabs(value - std::round(value)) < 1e-9) {
        oss << std::fixed << std::setprecision(1) << value;
    } else {
        oss << std::fixed << std::setprecision(6) << value;
    }
    return oss.str();
}

std::string FormatPreview(const std::vector<double>& values, size_t limit = 6)
{
    std::ostringstream oss;
    oss << "[";
    const size_t count = values.size();
    const size_t head = std::min(limit, count);
    for (size_t i = 0; i < head; ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << FormatScalar(values[i]);
    }
    if (count > limit) {
        oss << ", ..., " << FormatScalar(values.back());
    }
    oss << "]";
    return oss.str();
}

int NormalizeDim(const std::vector<int64_t>& shape, int64_t dim, int64_t* normalizedDim)
{
    CHECK_RET(normalizedDim != nullptr, return ACL_ERROR_INVALID_PARAM);
    int64_t rank = static_cast<int64_t>(shape.size());
    if (rank == 0) {
        rank = 1;
    }
    if (dim < -rank || dim >= rank) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *normalizedDim = dim >= 0 ? dim : dim + rank;
    return ACL_SUCCESS;
}

std::vector<double> CpuCumsum(
    const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim, bool exclusive, bool reverse,
    aclDataType selfType, aclDataType outType)
{
    std::vector<double> output(input.size(), 0.0);
    if (input.empty()) {
        return output;
    }

    if (shape.empty()) {
        if (exclusive) {
            output[0] = QuantizeToType(0.0, outType);
        } else {
            const double casted = QuantizeToType(QuantizeToType(input[0], selfType), outType);
            output[0] = QuantizeToType(casted, outType);
        }
        return output;
    }

    int64_t axis = 0;
    if (NormalizeDim(shape, dim, &axis) != ACL_SUCCESS) {
        return output;
    }

    const std::vector<int64_t> strides = BuildContiguousStrides(shape);
    const int64_t axisStride = strides[static_cast<size_t>(axis)];
    const int64_t axisLen = shape[static_cast<size_t>(axis)];
    const int64_t total = GetShapeSize(shape);

    std::vector<double> castInput(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        castInput[i] = QuantizeToType(QuantizeToType(input[i], selfType), outType);
    }

    for (int64_t linearIndex = 0; linearIndex < total; ++linearIndex) {
        const int64_t axisIndex = (linearIndex / axisStride) % axisLen;
        const int64_t baseIndex = linearIndex - axisIndex * axisStride;
        const int64_t begin = reverse ? axisIndex + (exclusive ? 1 : 0) : 0;
        const int64_t end = reverse ? axisLen : axisIndex + (exclusive ? 0 : 1);
        double sum = 0.0;
        for (int64_t offset = begin; offset < end; ++offset) {
            const size_t inputIndex = static_cast<size_t>(baseIndex + offset * axisStride);
            sum = QuantizeToType(sum + castInput[inputIndex], outType);
        }
        output[static_cast<size_t>(linearIndex)] = QuantizeToType(sum, outType);
    }

    return output;
}

bool CompareValues(
    const std::string& caseName, const std::vector<double>& actual, const std::vector<double>& expected,
    aclDataType dataType, const Tolerance& tolerance, CompareStats* stats)
{
    if (stats != nullptr) {
        *stats = CompareStats{};
    }
    if (actual.size() != expected.size()) {
        LOG_PRINT("[%s] [FAIL] size mismatch, actual=%zu expected=%zu\n",
            caseName.c_str(),
            actual.size(),
            expected.size());
        return false;
    }

    bool pass = true;
    double maxDiff = 0.0;
    size_t maxIndex = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(actual[i] - expected[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxIndex = i;
        }

        bool currentPass = false;
        if (IsFloatType(dataType)) {
            currentPass = diff <= tolerance.atol + tolerance.rtol * std::fabs(expected[i]);
        } else {
            currentPass = actual[i] == expected[i];
        }

        if (!currentPass) {
            if (stats != nullptr && !stats->mismatchFound) {
                stats->mismatchFound = true;
                stats->mismatchIndex = i;
            }
            pass = false;
        }
    }

    if (stats != nullptr) {
        stats->maxDiff = maxDiff;
        stats->maxIndex = maxIndex;
    }

    if (actual.empty()) {
        LOG_PRINT("[%s] [PASS] empty output verified\n", caseName.c_str());
        return true;
    }

    if (!pass) {
        const size_t mismatchIndex = (stats != nullptr && stats->mismatchFound) ? stats->mismatchIndex : maxIndex;
        LOG_PRINT("[%s] mismatch at %zu, actual=%.10f expected=%.10f diff=%.10f\n",
            caseName.c_str(),
            mismatchIndex,
            actual[mismatchIndex],
            expected[mismatchIndex],
            std::fabs(actual[mismatchIndex] - expected[mismatchIndex]));
    }

    LOG_PRINT("[%s] max_diff=%.10f at index=%zu [%s]\n",
        caseName.c_str(),
        maxDiff,
        maxIndex,
        pass ? "PASS" : "FAIL");
    return pass;
}

int Init(RuntimeContext* context)
{
    CHECK_RET(context != nullptr, return ACL_ERROR_INVALID_PARAM);

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSetDevice(context->deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);

    ret = aclrtCreateStream(&context->stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(RuntimeContext* context)
{
    if (context == nullptr) {
        return;
    }
    if (context->stream != nullptr) {
        aclrtDestroyStream(context->stream);
        context->stream = nullptr;
    }
    aclrtResetDevice(context->deviceId);
    aclFinalize();
}

int CreateAclTensorFromBytes(
    const std::vector<uint8_t>& hostBytes, const std::vector<int64_t>& shape, aclDataType dataType,
    TensorResource* resource)
{
    CHECK_RET(resource != nullptr, return ACL_ERROR_INVALID_PARAM);

    const size_t typeSize = GetTypeSize(dataType);
    CHECK_RET(typeSize > 0, LOG_PRINT("unsupported aclDataType=%d\n", static_cast<int>(dataType)); return ACL_ERROR_INVALID_PARAM);

    const size_t expectedBytes = static_cast<size_t>(GetShapeSize(shape)) * typeSize;
    CHECK_RET(hostBytes.size() == expectedBytes,
        LOG_PRINT("host bytes mismatch, expect=%zu actual=%zu\n", expectedBytes, hostBytes.size()); return ACL_ERROR_INVALID_PARAM);

    resource->bytes = expectedBytes;
    const size_t allocBytes = expectedBytes == 0 ? 1 : expectedBytes;
    auto ret = aclrtMalloc(&resource->deviceAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    if (expectedBytes > 0) {
        ret = aclrtMemcpy(resource->deviceAddr, expectedBytes, hostBytes.data(), expectedBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides = BuildContiguousStrides(shape);
    resource->tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        resource->deviceAddr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

void DestroyTensorResource(TensorResource* resource)
{
    if (resource == nullptr) {
        return;
    }
    if (resource->tensor != nullptr) {
        aclDestroyTensor(resource->tensor);
        resource->tensor = nullptr;
    }
    if (resource->deviceAddr != nullptr) {
        aclrtFree(resource->deviceAddr);
        resource->deviceAddr = nullptr;
    }
    resource->bytes = 0;
}

int CopyTensorToHostBytes(const TensorResource& resource, std::vector<uint8_t>* bytes)
{
    CHECK_RET(bytes != nullptr, return ACL_ERROR_INVALID_PARAM);
    bytes->assign(resource.bytes, 0);
    if (resource.bytes == 0) {
        return ACL_SUCCESS;
    }
    const auto ret = aclrtMemcpy(bytes->data(), resource.bytes, resource.deviceAddr, resource.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

TestResult RunSuccessCase(const RuntimeContext& context, const SuccessCase& testCase)
{
    LOG_PRINT("Running %s\n", testCase.name.c_str());

    TestResult result;
    result.name = testCase.name;
    result.pass = false;
    result.includeInSummary = testCase.includeInSummary;
    result.summaryTitle = testCase.summaryTitle;
    result.summaryNote = testCase.summaryNote;
    TensorResource self;
    TensorResource out;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    const int64_t elementCount = GetShapeSize(testCase.shape);
    CHECK_RET(static_cast<int64_t>(testCase.hostValues.size()) == elementCount,
        LOG_PRINT("[%s] [FAIL] host value count mismatch, expect=%ld actual=%zu\n",
            testCase.name.c_str(),
            elementCount,
            testCase.hostValues.size());
        return result);

    const std::vector<uint8_t> selfBytes = EncodeValues(testCase.hostValues, testCase.selfType);
    const std::vector<uint8_t> outBytes(static_cast<size_t>(elementCount) * GetTypeSize(testCase.outType), 0);

    int ret = CreateAclTensorFromBytes(selfBytes, testCase.shape, testCase.selfType, &self);
    CHECK_RET(ret == ACL_SUCCESS, return result);

    ret = CreateAclTensorFromBytes(outBytes, testCase.shape, testCase.outType, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorResource(&self); return result);

    aclnnStatus apiRet = ACLNN_SUCCESS;
    if (testCase.apiKind == ApiKind::kCumsum) {
        apiRet = aclnnCumsumGetWorkspaceSize(
            self.tensor, testCase.dim, testCase.outType, out.tensor, &workspaceSize, &executor);
    } else {
        apiRet = aclnnCumsumV2GetWorkspaceSize(
            self.tensor, testCase.dim, testCase.exclusive, testCase.reverse, out.tensor, &workspaceSize, &executor);
    }

    CHECK_RET(apiRet == ACLNN_SUCCESS,
        LOG_PRINT("[%s] [FAIL] GetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), apiRet);
        DestroyTensorResource(&self);
        DestroyTensorResource(&out);
        return result);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] [FAIL] workspace alloc failed. ERROR: %d\n", testCase.name.c_str(), ret);
            DestroyTensorResource(&self);
            DestroyTensorResource(&out);
            return result);
    }

    if (testCase.apiKind == ApiKind::kCumsum) {
        apiRet = aclnnCumsum(workspaceAddr, workspaceSize, executor, context.stream);
    } else {
        apiRet = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, context.stream);
    }
    CHECK_RET(apiRet == ACLNN_SUCCESS,
        LOG_PRINT("[%s] [FAIL] launch failed. ERROR: %d\n", testCase.name.c_str(), apiRet);
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        DestroyTensorResource(&self);
        DestroyTensorResource(&out);
        return result);

    ret = aclrtSynchronizeStream(context.stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[%s] [FAIL] stream sync failed. ERROR: %d\n", testCase.name.c_str(), ret);
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        DestroyTensorResource(&self);
        DestroyTensorResource(&out);
        return result);

    std::vector<uint8_t> actualBytes;
    ret = CopyTensorToHostBytes(out, &actualBytes);
    CHECK_RET(ret == ACL_SUCCESS,
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        DestroyTensorResource(&self);
        DestroyTensorResource(&out);
        return result);

    const std::vector<double> actualValues = DecodeValues(actualBytes, testCase.outType);
    const std::vector<double> expectedValues = CpuCumsum(testCase.hostValues,
        testCase.shape,
        testCase.dim,
        testCase.exclusive,
        testCase.reverse,
        testCase.selfType,
        testCase.outType);
    CompareStats stats;
    result.pass = CompareValues(testCase.name, actualValues, expectedValues, testCase.outType, testCase.tolerance, &stats);
    result.maxError = stats.maxDiff;
    result.maxErrorIndex = stats.maxIndex;
    result.expectedPreview = FormatPreview(expectedValues);
    result.actualPreview = FormatPreview(actualValues);

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return result;
}

TestResult RunErrorCase(const ErrorCase& testCase)
{
    LOG_PRINT("Running %s\n", testCase.name.c_str());

    TestResult result{testCase.name, false};
    result.isErrorCase = true;
    result.summaryTitle = testCase.name;
    TensorResource self;
    TensorResource out;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    if (!testCase.nullSelf) {
        const std::vector<uint8_t> selfBytes = EncodeValues(testCase.selfValues, testCase.selfType);
        int ret = CreateAclTensorFromBytes(selfBytes, testCase.selfShape, testCase.selfType, &self);
        CHECK_RET(ret == ACL_SUCCESS, return result);
        selfTensor = self.tensor;
    }

    if (!testCase.nullOut) {
        const std::vector<uint8_t> outBytes = EncodeValues(testCase.outValues, testCase.outType);
        int ret = CreateAclTensorFromBytes(outBytes, testCase.outShape, testCase.outType, &out);
        CHECK_RET(ret == ACL_SUCCESS,
            DestroyTensorResource(&self);
            return result);
        outTensor = out.tensor;
    }

    aclnnStatus apiRet = ACLNN_SUCCESS;
    if (testCase.apiKind == ApiKind::kCumsum) {
        apiRet = aclnnCumsumGetWorkspaceSize(
            selfTensor, testCase.dim, testCase.dtype, outTensor, &workspaceSize, &executor);
    } else {
        apiRet = aclnnCumsumV2GetWorkspaceSize(
            selfTensor, testCase.dim, testCase.exclusive, testCase.reverse, outTensor, &workspaceSize, &executor);
    }

    result.pass = (apiRet == testCase.expectedStatus);
    result.expectedPreview = "status=" + std::to_string(testCase.expectedStatus);
    result.actualPreview = "status=" + std::to_string(apiRet);
    LOG_PRINT("[%s] expect=%d actual=%d [%s]\n",
        testCase.name.c_str(),
        testCase.expectedStatus,
        apiRet,
        result.pass ? "PASS" : "FAIL");

    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return result;
}

std::vector<SuccessCase> BuildSuccessCases()
{
    std::vector<SuccessCase> cases;

    cases.push_back({"float32_dim0_basic",
        ApiKind::kCumsum,
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        {1.0, 2.0, 3.0, 4.0}});

    cases.push_back({"float16_to_float_cast_last_dim",
        ApiKind::kCumsum,
        {2, 3, 4},
        -1,
        ACL_FLOAT16,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(24, 1.5, -0.5)});

    cases.push_back({"bf16_host_path_axis0",
        ApiKind::kCumsum,
        {9, 1, 1, 8, 1, 17, 17},
        -7,
        ACL_BF16,
        ACL_BF16,
        false,
        false,
        {2e-2, 2e-2},
        GenerateAlternating(9 * 1 * 1 * 8 * 1 * 17 * 17, 1.0, -0.5)});

    cases.push_back({"double_aicpu_path_dim1",
        ApiKind::kCumsum,
        {3, 4},
        1,
        ACL_DOUBLE,
        ACL_DOUBLE,
        false,
        false,
        {1e-8, 1e-8},
        {0.25, -1.5, 2.0, 4.5, 1.25, 1.75, -0.5, 2.5, 3.0, 0.0, -2.0, 5.0}});

    cases.push_back({"int32_axis0",
        ApiKind::kCumsum,
        {4, 3, 2},
        0,
        ACL_INT32,
        ACL_INT32,
        false,
        false,
        {0.0, 0.0},
        GenerateSequence(24, -3.0, 1.0)});

    cases.push_back({"int64_last_dim",
        ApiKind::kCumsum,
        {3, 4, 5},
        2,
        ACL_INT64,
        ACL_INT64,
        false,
        false,
        {0.0, 0.0},
        GenerateSequence(60, 1.0, 1.0)});

    cases.push_back({"int16_axis0",
        ApiKind::kCumsum,
        {16, 8, 9, 16},
        0,
        ACL_INT16,
        ACL_INT16,
        false,
        false,
        {0.0, 0.0},
        GenerateAlternating(16 * 8 * 9 * 16, 7.0, -3.0)});

    cases.push_back({"int8_middle_axis_large_right",
        ApiKind::kCumsum,
        {5, 7, 80},
        1,
        ACL_INT8,
        ACL_INT8,
        false,
        false,
        {0.0, 0.0},
        GenerateAlternating(2800, 2.0, -1.0)});

    cases.push_back({"uint8_negative_dim",
        ApiKind::kCumsum,
        {5, 6, 7},
        -1,
        ACL_UINT8,
        ACL_UINT8,
        false,
        false,
        {0.0, 0.0},
        GenerateSequence(210, 1.0, 1.0)});

    cases.push_back({"scalar_0dim",
        ApiKind::kCumsum,
        {},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        {3.25}});

    cases.push_back({"empty_tensor",
        ApiKind::kCumsum,
        {2, 0, 3},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        {}});

    cases.push_back({"v2_empty_tensor",
        ApiKind::kCumsumV2,
        {0},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        true,
        true,
        {1e-5, 1e-5},
        {}});

    cases.push_back({"v2_scalar_dim0_reverse",
        ApiKind::kCumsumV2,
        {},
        0,
        ACL_FLOAT16,
        ACL_FLOAT16,
        false,
        true,
        {1e-3, 1e-3},
        {2.0}});

    cases.push_back({"float_host_axis0_large_n",
        ApiKind::kCumsum,
        {33, 513},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(33 * 513, 2.0, -1.0)});

    cases.push_back({"float_host_last_dim_r_full_load",
        ApiKind::kCumsum,
        {64, 256},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateSequence(64 * 256, -3.0, 0.125)});

    cases.push_back({"float_host_last_dim_r_not_full_load",
        ApiKind::kCumsum,
        {4, 4096},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(4 * 4096, 0.75, -0.25)});

    cases.push_back({"float_host_middle_axis_large_n",
        ApiKind::kCumsum,
        {2, 48, 768},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(2 * 48 * 768, 1.0, 0.25)});

    cases.push_back({"float_host_middle_axis_small_n",
        ApiKind::kCumsum,
        {3, 257, 7},
        1,
        ACL_FLOAT16,
        ACL_FLOAT16,
        false,
        false,
        {1e-3, 1e-3},
        GenerateAlternating(3 * 257 * 7, 0.5, -0.25)});

    cases.push_back({"cube_support_float32",
        ApiKind::kCumsum,
        {12800, 512},
        -1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(12800 * 512, 1.0, 0.5)});

    cases.push_back({"v2_float32_exclusive",
        ApiKind::kCumsumV2,
        {2, 3, 4},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        true,
        false,
        {1e-5, 1e-5},
        GenerateSequence(24, 1.0, 0.5)});

    cases.push_back({"v2_float16_reverse",
        ApiKind::kCumsumV2,
        {2, 3, 4},
        -2,
        ACL_FLOAT16,
        ACL_FLOAT16,
        false,
        true,
        {1e-3, 1e-3},
        GenerateAlternating(24, 0.25, 1.75)});

    cases.push_back({"v2_float32_exclusive_reverse",
        ApiKind::kCumsumV2,
        {4, 5},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        true,
        true,
        {1e-5, 1e-5},
        GenerateMixedMagnitude(20)});

    cases.push_back({"v2_float32_middle_axis_host",
        ApiKind::kCumsumV2,
        {8, 33, 65},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        true,
        false,
        {1e-5, 1e-5},
        GenerateSequence(8 * 33 * 65, -2.0, 0.25)});

    cases.push_back({"v2_bf16_last_dim_host_reverse",
        ApiKind::kCumsumV2,
        {5, 1, 5, 8, 9, 5, 1},
        4,
        ACL_BF16,
        ACL_BF16,
        false,
        true,
        {3e-2, 3e-2},
        GenerateAlternating(5 * 1 * 5 * 8 * 9 * 5 * 1, 1.0, 0.5)});

    cases.push_back({"v2_int32_axis0_exclusive_reverse",
        ApiKind::kCumsumV2,
        {9, 17, 9, 17},
        0,
        ACL_INT32,
        ACL_INT32,
        true,
        true,
        {0.0, 0.0},
        GenerateSequence(9 * 17 * 9 * 17, -8.0, 1.0)});

    cases.push_back({"v2_int8_last_dim_reverse",
        ApiKind::kCumsumV2,
        {1, 16, 1, 1, 8, 16, 15},
        6,
        ACL_INT8,
        ACL_INT8,
        false,
        true,
        {0.0, 0.0},
        GenerateAlternating(1 * 16 * 1 * 1 * 8 * 16 * 15, 3.0, -2.0)});

    cases.push_back({"v2_double_aicpu_exclusive_reverse",
        ApiKind::kCumsumV2,
        {5, 9},
        1,
        ACL_DOUBLE,
        ACL_DOUBLE,
        true,
        true,
        {1e-8, 1e-8},
        GenerateSequence(45, -1.5, 0.5)});

    cases.push_back({"float32_long_ones_precision",
        ApiKind::kCumsum,
        {16},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateConstant(16, 1.0),
        "Basic Cumsum (float32, length=16)",
        "Use a short float32 sequence to quickly verify basic accumulation correctness.",
        true});

    cases.push_back({"float16_long_ones_precision",
        ApiKind::kCumsum,
        {2, 8},
        1,
        ACL_FLOAT16,
        ACL_FLOAT16,
        false,
        false,
        {1e-3, 1e-3},
        GenerateConstant(16, 1.0),
        "Basic Cumsum (float16, shape=2x8)",
        "Check a small float16 last-dimension accumulation before testing long sequences.",
        false});

    cases.push_back({"float32_decimal_precision",
        ApiKind::kCumsum,
        {8},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateTinyDecimal(8, 0.1),
        "Decimal accumulation (float32, length=8)",
        "Use a short 0.1 sequence to verify decimal accumulation behavior with clear expected results.",
        false});

    cases.push_back({"float32_mixed_magnitude_precision",
        ApiKind::kCumsum,
        {8},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateMixedMagnitude(8),
        "Mixed magnitude (float32, length=8)",
        "Use a very short mixed-magnitude sequence so any zero output is immediately visible.",
        false});

    cases.push_back({"float32_cancellation_precision",
        ApiKind::kCumsum,
        {8},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(8, 1.0, -0.999999),
        "Alternating cancellation (float32, length=8)",
        "Keep the sequence short so cancellation behavior can be checked by direct inspection.",
        false});

    cases.push_back({"v2_exclusive_reverse_precision",
        ApiKind::kCumsumV2,
        {2, 4},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        true,
        true,
        {1e-5, 1e-5},
        GenerateSequence(8, 1.0, 1.0),
        "Exclusive + reverse mode (float32, shape=2x4)",
        "Use a tiny V2 case to confirm combined semantics before large-shape validation.",
        false});

    cases.push_back({"bf16_precision_comparison",
        ApiKind::kCumsum,
        {2, 8},
        0,
        ACL_BF16,
        ACL_BF16,
        false,
        false,
        {3e-2, 3e-2},
        GenerateTinyDecimal(16, 0.1),
        "Basic Cumsum (bf16, shape=2x8)",
        "Use a small bf16 case to check basic correctness before analyzing bf16 drift.",
        false});

    cases.push_back({"float_host_ngreater_rfull_m_ge_core_n_full_ub",
        ApiKind::kCumsum,
        {512, 128, 16},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(512 * 128 * 16, 1.0, -0.25)});

    cases.push_back({"float_host_ngreater_rfull_m_ge_core_n_split_ub",
        ApiKind::kCumsum,
        {32, 512, 128},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateSequence(32 * 512 * 128, -1.0, 0.03125)});

    cases.push_back({"float_host_ngreater_rnotfull_m_ge_core",
        ApiKind::kCumsum,
        {40, 4096, 16},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(40 * 4096 * 16, 0.5, -0.125)});

    cases.push_back({"float_host_ngreater_rnotfull_borrow_n",
        ApiKind::kCumsum,
        {1, 4096, 512},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(1 * 4096 * 512, 0.75, -0.5)});

    cases.push_back({"float_host_ngreater_rnotfull_borrow_r_full",
        ApiKind::kCumsum,
        {1, 4096, 128},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateSequence(1 * 4096 * 128, 0.0, 0.0625)});

    cases.push_back({"float_host_ngreater_rnotfull_borrow_r_notfull",
        ApiKind::kCumsum,
        {1, 8192, 128},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(1LL * 8192 * 128, 0.125, -0.0625)});

    cases.push_back({"float_host_rngreater_borrow_m",
        ApiKind::kCumsum,
        {160, 1024, 8},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(160 * 1024 * 8, 0.5, 0.25)});

    cases.push_back({"float_host_rngreater_notfull_notborrowr",
        ApiKind::kCumsum,
        {40, 4096, 8},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(40 * 4096 * 8, 1.0, -0.5)});

    cases.push_back({"float_host_rngreater_notfull_borrowr_twoway",
        ApiKind::kCumsum,
        {1, 32768, 8},
        1,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        {1e-5, 1e-5},
        GenerateAlternating(1LL * 32768 * 8, 0.25, -0.125)});

    return cases;
}

std::vector<ErrorCase> BuildErrorCases()
{
    std::vector<ErrorCase> cases;

    cases.push_back({"error_null_self",
        ApiKind::kCumsum,
        {2, 2},
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        true,
        false,
        ACLNN_ERR_PARAM_NULLPTR,
        {},
        {0.0, 0.0, 0.0, 0.0}});

    cases.push_back({"error_null_out",
        ApiKind::kCumsum,
        {2, 2},
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        false,
        true,
        ACLNN_ERR_PARAM_NULLPTR,
        {1.0, 2.0, 3.0, 4.0},
        {}});

    cases.push_back({"error_dtype_bool",
        ApiKind::kCumsum,
        {2, 2},
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_BOOL,
        false,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        {1.0, 2.0, 3.0, 4.0},
        {0.0, 0.0, 0.0, 0.0}});

    cases.push_back({"error_shape_mismatch",
        ApiKind::kCumsum,
        {2, 2},
        {2, 3},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        {1.0, 2.0, 3.0, 4.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

    cases.push_back({"error_dim_out_of_range",
        ApiKind::kCumsum,
        {2, 2},
        {2, 2},
        2,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        {1.0, 2.0, 3.0, 4.0},
        {0.0, 0.0, 0.0, 0.0}});

    cases.push_back({"error_rank_gt_8",
        ApiKind::kCumsum,
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        {1.0},
        {0.0}});

    cases.push_back({"error_v2_dtype_mismatch_self_out",
        ApiKind::kCumsumV2,
        {2, 2},
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_INT32,
        ACL_FLOAT,
        false,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        {1.0, 2.0, 3.0, 4.0},
        {0.0, 0.0, 0.0, 0.0}});

    cases.push_back({"error_v2_dim_out_of_range",
        ApiKind::kCumsumV2,
        {2, 2, 2},
        {2, 2, 2},
        -4,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        true,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        GenerateSequence(8, 1.0, 1.0),
        std::vector<double>(8, 0.0)});

    cases.push_back({"error_dtype_undefined",
        ApiKind::kCumsum,
        {2, 2},
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_DT_UNDEFINED,
        false,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        {1.0, 2.0, 3.0, 4.0},
        {0.0, 0.0, 0.0, 0.0}});

    cases.push_back({"error_v2_null_self",
        ApiKind::kCumsumV2,
        {2, 2},
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        true,
        true,
        false,
        ACLNN_ERR_PARAM_NULLPTR,
        {},
        {0.0, 0.0, 0.0, 0.0}});

    cases.push_back({"error_v2_null_out",
        ApiKind::kCumsumV2,
        {2, 2},
        {2, 2},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        true,
        false,
        false,
        true,
        ACLNN_ERR_PARAM_NULLPTR,
        {1.0, 2.0, 3.0, 4.0},
        {}});

    cases.push_back({"error_v2_rank_gt_8",
        ApiKind::kCumsumV2,
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        0,
        ACL_FLOAT,
        ACL_FLOAT,
        ACL_FLOAT,
        false,
        false,
        false,
        false,
        ACLNN_ERR_PARAM_INVALID,
        {1.0},
        {0.0}});

    return cases;
}

void PrintAllCaseResults(const std::vector<TestResult>& allResults)
{
    LOG_PRINT("\n========== Final Test Results ==========\n");
    int caseIndex = 1;
    for (const auto& result : allResults) {
        const std::string title = result.summaryTitle.empty() ? result.name : result.summaryTitle;
        LOG_PRINT("Test case %d: %s\n", caseIndex, title.c_str());
        LOG_PRINT("  Expected: %s\n", result.expectedPreview.empty() ? "N/A" : result.expectedPreview.c_str());
        LOG_PRINT("  Actual:   %s\n", result.actualPreview.empty() ? "N/A" : result.actualPreview.c_str());
        if (!result.isErrorCase) {
            LOG_PRINT("  Max error: %.10f (at position %zu)\n", result.maxError, result.maxErrorIndex);
        } else {
            LOG_PRINT("  Max error: N/A\n");
        }
        LOG_PRINT("  [%s]\n\n", result.pass ? "PASS" : "FAIL");
        ++caseIndex;
    }
}

void PrintDetailedSummary(const std::vector<TestResult>& successResults)
{
    int summaryCount = 0;
    int summaryPassed = 0;
    for (const auto& result : successResults) {
        if (!result.includeInSummary) {
            continue;
        }
        ++summaryCount;
        if (result.pass) {
            ++summaryPassed;
        }
    }

    LOG_PRINT("\n========== Precision Test Summary ==========\n");
    int caseIndex = 1;
    for (const auto& result : successResults) {
        if (!result.includeInSummary) {
            continue;
        }
        const std::string title = result.summaryTitle.empty() ? result.name : result.summaryTitle;
        LOG_PRINT("Test case %d: %s\n", caseIndex, title.c_str());
        LOG_PRINT("  Expected: %s\n", result.expectedPreview.c_str());
        LOG_PRINT("  Actual:   %s\n", result.actualPreview.c_str());
        LOG_PRINT("  Max error: %.10f (at position %zu)\n", result.maxError, result.maxErrorIndex);
        if (!result.summaryNote.empty()) {
            LOG_PRINT("  Note: %s\n", result.summaryNote.c_str());
        }
        LOG_PRINT("  [%s]\n\n", result.pass ? "PASS" : "FAIL");
        ++caseIndex;
    }
    LOG_PRINT("Precision Summary: %d passed, %d failed\n", summaryPassed, summaryCount - summaryPassed);
}

void WritePrecisionSummaryToFile(const std::vector<TestResult>& successResults)
{
    FILE* file = std::fopen(kPrecisionSummaryLogPath, "w");
    if (file == nullptr) {
        LOG_PRINT("[WARN] failed to open precision summary log: %s\n", kPrecisionSummaryLogPath);
        return;
    }

    int summaryCount = 0;
    int summaryPassed = 0;
    std::fprintf(file, "========== Precision Test Summary ==========\n");
    int caseIndex = 1;
    for (const auto& result : successResults) {
        if (!result.includeInSummary) {
            continue;
        }
        ++summaryCount;
        if (result.pass) {
            ++summaryPassed;
        }
        const std::string title = result.summaryTitle.empty() ? result.name : result.summaryTitle;
        std::fprintf(file, "Test case %d: %s\n", caseIndex, title.c_str());
        std::fprintf(file, "  Expected: %s\n", result.expectedPreview.c_str());
        std::fprintf(file, "  Actual:   %s\n", result.actualPreview.c_str());
        std::fprintf(file, "  Max error: %.10f (at position %zu)\n", result.maxError, result.maxErrorIndex);
        if (!result.summaryNote.empty()) {
            std::fprintf(file, "  Note: %s\n", result.summaryNote.c_str());
        }
        std::fprintf(file, "  [%s]\n\n", result.pass ? "PASS" : "FAIL");
        ++caseIndex;
    }
    std::fprintf(file, "Precision Summary: %d passed, %d failed\n", summaryPassed, summaryCount - summaryPassed);
    std::fclose(file);
    LOG_PRINT("Precision summary saved to %s\n", kPrecisionSummaryLogPath);
}

}  // namespace

int main()
{
    RuntimeContext context;
    int ret = Init(&context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    int passed = 0;
    int failed = 0;

    std::vector<TestResult> allResults;
    std::vector<TestResult> successResults;
    const std::vector<SuccessCase> successCases = BuildSuccessCases();
    allResults.reserve(successCases.size() + BuildErrorCases().size());
    successResults.reserve(successCases.size());
    for (const auto& testCase : successCases) {
        TestResult result = RunSuccessCase(context, testCase);
        successResults.push_back(result);
        allResults.push_back(result);
        if (result.pass) {
            ++passed;
        } else {
            ++failed;
        }
    }

    const std::vector<ErrorCase> errorCases = BuildErrorCases();
    for (const auto& testCase : errorCases) {
        TestResult result = RunErrorCase(testCase);
        allResults.push_back(result);
        if (result.pass) {
            ++passed;
        } else {
            ++failed;
        }
    }

    PrintAllCaseResults(allResults);
    PrintDetailedSummary(successResults);
    WritePrecisionSummaryToFile(successResults);
    LOG_PRINT("Summary: %d passed, %d failed\n", passed, failed);
    Finalize(&context);
    return failed == 0 ? 0 : 1;
}
