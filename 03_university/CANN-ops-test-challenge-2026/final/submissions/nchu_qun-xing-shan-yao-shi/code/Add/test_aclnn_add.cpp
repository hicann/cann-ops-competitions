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
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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
        fflush(stdout);                 \
    } while (0)

namespace {

struct TensorSpec {
    std::vector<int64_t> shape;
    aclDataType dtype;
    std::vector<double> values;
};

struct ScalarSpec {
    aclDataType dtype;
    double value;
};

struct TensorHolder {
    aclTensor* tensor = nullptr;
    void* deviceAddr = nullptr;
    std::vector<int64_t> shape;
    aclDataType dtype = ACL_FLOAT;
};

struct CompareReport {
    bool ok = true;
    int64_t badIndex = -1;
    double expected = 0.0;
    double actual = 0.0;
    double absError = 0.0;
    double relError = 0.0;
};

struct TestStats {
    int passed = 0;
    int failed = 0;
    int total = 0;
};

using ExecuteFunc = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

constexpr aclnnStatus ACLNN_STATUS_PARAM_NULLPTR = static_cast<aclnnStatus>(161001);
constexpr aclnnStatus ACLNN_STATUS_PARAM_INVALID = static_cast<aclnnStatus>(161002);

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        shapeSize *= shape[i];
    }
    return shapeSize;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

const char* DTypeName(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return "FLOAT32";
        case ACL_FLOAT16:
            return "FLOAT16";
        case ACL_BF16:
            return "BFLOAT16";
        case ACL_INT32:
            return "INT32";
        case ACL_INT64:
            return "INT64";
        case ACL_INT8:
            return "INT8";
        case ACL_UINT8:
            return "UINT8";
        case ACL_UINT32:
            return "UINT32";
        case ACL_BOOL:
            return "BOOL";
        case ACL_DOUBLE:
            return "DOUBLE";
        default:
            return "UNKNOWN";
    }
}

size_t DTypeSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_INT32:
        case ACL_UINT32:
            return sizeof(uint32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_INT8:
        case ACL_UINT8:
        case ACL_BOOL:
            return sizeof(uint8_t);
        case ACL_DOUBLE:
            return sizeof(double);
        default:
            return sizeof(uint8_t);
    }
}

uint32_t FloatToBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float BitsToFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t FloatToHalfBits(float value)
{
    uint32_t bits = FloatToBits(value);
    uint32_t sign = (bits >> 16) & 0x8000U;
    uint32_t exponent = (bits >> 23) & 0xFFU;
    uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        uint32_t nanMantissa = mantissa >> 13;
        return static_cast<uint16_t>(sign | 0x7C00U | nanMantissa | 1U);
    }

    int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000U;
        uint32_t shift = static_cast<uint32_t>(14 - halfExponent);
        uint32_t rounded = (mantissa + (1U << (shift - 1))) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t roundedMantissa = mantissa + 0x1000U;
    if (roundedMantissa & 0x800000U) {
        roundedMantissa = 0;
        ++halfExponent;
        if (halfExponent >= 0x1F) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExponent) << 10) | (roundedMantissa >> 13));
}

float HalfBitsToFloat(uint16_t half)
{
    uint32_t sign = (static_cast<uint32_t>(half & 0x8000U)) << 16;
    uint32_t exponent = (half >> 10) & 0x1FU;
    uint32_t mantissa = half & 0x03FFU;
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
            uint32_t floatExponent = exponent + (127 - 15);
            bits = sign | (floatExponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        uint32_t floatExponent = exponent + (127 - 15);
        bits = sign | (floatExponent << 23) | (mantissa << 13);
    }

    return BitsToFloat(bits);
}

uint16_t FloatToBfloat16Bits(float value)
{
    uint32_t bits = FloatToBits(value);
    uint32_t absBits = bits & 0x7FFFFFFFU;
    if (absBits > 0x7F800000U) {
        uint16_t nanBits = static_cast<uint16_t>(bits >> 16);
        return static_cast<uint16_t>(nanBits | 0x0040U);
    }
    uint32_t lsb = (bits >> 16) & 1U;
    uint32_t roundingBias = 0x7FFFU + lsb;
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

float Bfloat16BitsToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    return BitsToFloat(bits);
}

int32_t WrapInt32(double value)
{
    int64_t rounded = static_cast<int64_t>(value);
    uint32_t wrapped = static_cast<uint32_t>(rounded);
    int32_t result = 0;
    std::memcpy(&result, &wrapped, sizeof(result));
    return result;
}

int8_t WrapInt8(double value)
{
    int64_t rounded = static_cast<int64_t>(value);
    uint8_t wrapped = static_cast<uint8_t>(rounded);
    int8_t result = 0;
    std::memcpy(&result, &wrapped, sizeof(result));
    return result;
}

uint8_t WrapUint8(double value)
{
    int64_t rounded = static_cast<int64_t>(value);
    return static_cast<uint8_t>(rounded);
}

template <typename T>
void AppendBytes(std::vector<uint8_t>& bytes, const T& value)
{
    const uint8_t* begin = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(T));
}

double CastToStorage(double value, aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT: {
            float v = static_cast<float>(value);
            return static_cast<double>(v);
        }
        case ACL_FLOAT16: {
            float v = HalfBitsToFloat(FloatToHalfBits(static_cast<float>(value)));
            return static_cast<double>(v);
        }
        case ACL_BF16: {
            float v = Bfloat16BitsToFloat(FloatToBfloat16Bits(static_cast<float>(value)));
            return static_cast<double>(v);
        }
        case ACL_DOUBLE:
            return value;
        case ACL_INT32:
            return static_cast<double>(WrapInt32(value));
        case ACL_INT64:
            return static_cast<double>(static_cast<int64_t>(value));
        case ACL_INT8:
            return static_cast<double>(WrapInt8(value));
        case ACL_UINT8:
            return static_cast<double>(WrapUint8(value));
        case ACL_UINT32:
            return static_cast<double>(static_cast<uint32_t>(static_cast<uint64_t>(value)));
        case ACL_BOOL:
            return value != 0.0 ? 1.0 : 0.0;
        default:
            return value;
    }
}

std::vector<uint8_t> MakeTensorBytes(const TensorSpec& spec)
{
    std::vector<uint8_t> bytes;
    int64_t elementCount = GetShapeSize(spec.shape);
    bytes.reserve(static_cast<size_t>(std::max<int64_t>(elementCount, 0)) * DTypeSize(spec.dtype));
    for (int64_t i = 0; i < elementCount; ++i) {
        double source = spec.values.empty() ? 0.0 : spec.values[static_cast<size_t>(i) % spec.values.size()];
        switch (spec.dtype) {
            case ACL_FLOAT: {
                float v = static_cast<float>(source);
                AppendBytes(bytes, v);
                break;
            }
            case ACL_FLOAT16: {
                uint16_t v = FloatToHalfBits(static_cast<float>(source));
                AppendBytes(bytes, v);
                break;
            }
            case ACL_BF16: {
                uint16_t v = FloatToBfloat16Bits(static_cast<float>(source));
                AppendBytes(bytes, v);
                break;
            }
            case ACL_DOUBLE: {
                double v = source;
                AppendBytes(bytes, v);
                break;
            }
            case ACL_INT32: {
                int32_t v = static_cast<int32_t>(source);
                AppendBytes(bytes, v);
                break;
            }
            case ACL_INT64: {
                int64_t v = static_cast<int64_t>(source);
                AppendBytes(bytes, v);
                break;
            }
            case ACL_INT8: {
                int8_t v = static_cast<int8_t>(source);
                AppendBytes(bytes, v);
                break;
            }
            case ACL_UINT8: {
                uint8_t v = static_cast<uint8_t>(source);
                AppendBytes(bytes, v);
                break;
            }
            case ACL_UINT32: {
                uint32_t v = static_cast<uint32_t>(source);
                AppendBytes(bytes, v);
                break;
            }
            case ACL_BOOL: {
                uint8_t v = source != 0.0 ? 1U : 0U;
                AppendBytes(bytes, v);
                break;
            }
            default: {
                uint8_t v = 0U;
                AppendBytes(bytes, v);
                break;
            }
        }
    }
    return bytes;
}

double ReadValue(const uint8_t* data, aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT: {
            float v = 0.0f;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_FLOAT16: {
            uint16_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(HalfBitsToFloat(v));
        }
        case ACL_BF16: {
            uint16_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(Bfloat16BitsToFloat(v));
        }
        case ACL_DOUBLE: {
            double v = 0.0;
            std::memcpy(&v, data, sizeof(v));
            return v;
        }
        case ACL_INT32: {
            int32_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT64: {
            int64_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT8: {
            int8_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_UINT8: {
            uint8_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_UINT32: {
            uint32_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_BOOL: {
            uint8_t v = 0;
            std::memcpy(&v, data, sizeof(v));
            return v == 0 ? 0.0 : 1.0;
        }
        default:
            return 0.0;
    }
}

std::vector<double> DecodeTensorValues(const std::vector<uint8_t>& bytes, aclDataType dtype)
{
    std::vector<double> values;
    size_t typeSize = DTypeSize(dtype);
    if (typeSize == 0) {
        return values;
    }
    size_t count = bytes.size() / typeSize;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        values.push_back(ReadValue(bytes.data() + i * typeSize, dtype));
    }
    return values;
}

std::vector<double> StoredValues(const TensorSpec& spec)
{
    return DecodeTensorValues(MakeTensorBytes(spec), spec.dtype);
}

double StoredScalarValue(const ScalarSpec& spec)
{
    TensorSpec scalarAsTensor{{1}, spec.dtype, {spec.value}};
    std::vector<double> values = StoredValues(scalarAsTensor);
    return values.empty() ? 0.0 : values[0];
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

int CreateAclTensor(const TensorSpec& spec, TensorHolder& holder)
{
    std::vector<uint8_t> hostBytes = MakeTensorBytes(spec);
    int64_t elementCount = GetShapeSize(spec.shape);
    size_t dataSize = static_cast<size_t>(std::max<int64_t>(elementCount, 0)) * DTypeSize(spec.dtype);
    size_t allocSize = std::max<size_t>(dataSize, 1U);

    void* deviceAddr = nullptr;
    auto ret = aclrtMalloc(&deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    if (dataSize > 0) {
        ret = aclrtMemcpy(deviceAddr, dataSize, hostBytes.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); aclrtFree(deviceAddr); return ret);
    }

    std::vector<int64_t> strides = MakeStrides(spec.shape);
    aclTensor* tensor = aclCreateTensor(
        spec.shape.data(), spec.shape.size(), spec.dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        spec.shape.data(), spec.shape.size(), deviceAddr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed for dtype %s\n", DTypeName(spec.dtype)); aclrtFree(deviceAddr); return -1);

    holder.tensor = tensor;
    holder.deviceAddr = deviceAddr;
    holder.shape = spec.shape;
    holder.dtype = spec.dtype;
    return ACL_SUCCESS;
}

void DestroyTensor(TensorHolder& holder)
{
    if (holder.tensor != nullptr) {
        aclDestroyTensor(holder.tensor);
        holder.tensor = nullptr;
    }
    if (holder.deviceAddr != nullptr) {
        aclrtFree(holder.deviceAddr);
        holder.deviceAddr = nullptr;
    }
}

aclScalar* CreateAclScalar(const ScalarSpec& spec)
{
    switch (spec.dtype) {
        case ACL_FLOAT: {
            float v = static_cast<float>(spec.value);
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_FLOAT16: {
            uint16_t v = FloatToHalfBits(static_cast<float>(spec.value));
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_BF16: {
            uint16_t v = FloatToBfloat16Bits(static_cast<float>(spec.value));
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_DOUBLE: {
            double v = spec.value;
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_INT32: {
            int32_t v = static_cast<int32_t>(spec.value);
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_INT64: {
            int64_t v = static_cast<int64_t>(spec.value);
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_INT8: {
            int8_t v = static_cast<int8_t>(spec.value);
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_UINT8: {
            uint8_t v = static_cast<uint8_t>(spec.value);
            return aclCreateScalar(&v, spec.dtype);
        }
        case ACL_BOOL: {
            bool v = spec.value != 0.0;
            return aclCreateScalar(&v, spec.dtype);
        }
        default: {
            float v = static_cast<float>(spec.value);
            return aclCreateScalar(&v, spec.dtype);
        }
    }
}

bool CopyTensorToHost(const TensorHolder& holder, std::vector<uint8_t>& hostBytes)
{
    int64_t elementCount = GetShapeSize(holder.shape);
    size_t dataSize = static_cast<size_t>(std::max<int64_t>(elementCount, 0)) * DTypeSize(holder.dtype);
    hostBytes.assign(dataSize, 0);
    if (dataSize == 0) {
        return true;
    }
    auto ret = aclrtMemcpy(hostBytes.data(), dataSize, holder.deviceAddr, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return false);
    return true;
}

std::vector<int64_t> LinearToIndices(int64_t linear, const std::vector<int64_t>& shape)
{
    std::vector<int64_t> indices(shape.size(), 0);
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
        int64_t extent = shape[static_cast<size_t>(dim)];
        if (extent == 0) {
            indices[static_cast<size_t>(dim)] = 0;
            continue;
        }
        indices[static_cast<size_t>(dim)] = linear % extent;
        linear /= extent;
    }
    return indices;
}

int64_t BroadcastOffset(const std::vector<int64_t>& outIndices, const std::vector<int64_t>& outShape,
    const std::vector<int64_t>& inShape)
{
    (void)outShape;
    if (inShape.empty()) {
        return 0;
    }
    std::vector<int64_t> strides = MakeStrides(inShape);
    int64_t offset = 0;
    int64_t dimOffset = static_cast<int64_t>(outIndices.size()) - static_cast<int64_t>(inShape.size());
    for (size_t i = 0; i < inShape.size(); ++i) {
        int64_t outDim = static_cast<int64_t>(i) + dimOffset;
        int64_t outIndex = outDim < 0 ? 0 : outIndices[static_cast<size_t>(outDim)];
        int64_t inIndex = inShape[i] == 1 ? 0 : outIndex;
        offset += inIndex * strides[i];
    }
    return offset;
}

std::vector<double> ComputeTensorTensorExpected(const TensorSpec& selfSpec, const TensorSpec& otherSpec,
    const ScalarSpec& alphaSpec, const TensorSpec& outSpec)
{
    std::vector<double> selfValues = StoredValues(selfSpec);
    std::vector<double> otherValues = StoredValues(otherSpec);
    double alpha = StoredScalarValue(alphaSpec);
    int64_t outCount = GetShapeSize(outSpec.shape);
    std::vector<double> expected;
    expected.reserve(static_cast<size_t>(std::max<int64_t>(outCount, 0)));

    for (int64_t i = 0; i < outCount; ++i) {
        std::vector<int64_t> indices = LinearToIndices(i, outSpec.shape);
        int64_t selfOffset = BroadcastOffset(indices, outSpec.shape, selfSpec.shape);
        int64_t otherOffset = BroadcastOffset(indices, outSpec.shape, otherSpec.shape);
        double raw = selfValues[static_cast<size_t>(selfOffset)] + alpha * otherValues[static_cast<size_t>(otherOffset)];
        expected.push_back(CastToStorage(raw, outSpec.dtype));
    }
    return expected;
}

std::vector<double> ComputeAddsExpected(const TensorSpec& selfSpec, const ScalarSpec& otherSpec,
    const ScalarSpec& alphaSpec, const TensorSpec& outSpec)
{
    std::vector<double> selfValues = StoredValues(selfSpec);
    double other = StoredScalarValue(otherSpec);
    double alpha = StoredScalarValue(alphaSpec);
    int64_t outCount = GetShapeSize(outSpec.shape);
    std::vector<double> expected;
    expected.reserve(static_cast<size_t>(std::max<int64_t>(outCount, 0)));
    bool boolScalarSpecial = selfSpec.dtype == ACL_BOOL && otherSpec.dtype == ACL_BOOL && alphaSpec.dtype == ACL_BOOL &&
                             outSpec.dtype != ACL_BOOL && other != 0.0 && alpha != 0.0;

    for (int64_t i = 0; i < outCount; ++i) {
        double raw = selfValues[static_cast<size_t>(i)] + alpha * other;
        if (boolScalarSpecial) {
            raw = raw != 0.0 ? 1.0 : 0.0;
        }
        expected.push_back(CastToStorage(raw, outSpec.dtype));
    }
    return expected;
}

std::vector<double> ComputeV3Expected(const ScalarSpec& selfSpec, const TensorSpec& otherSpec,
    const ScalarSpec& alphaSpec, const TensorSpec& outSpec)
{
    double self = StoredScalarValue(selfSpec);
    std::vector<double> otherValues = StoredValues(otherSpec);
    double alpha = StoredScalarValue(alphaSpec);
    int64_t outCount = GetShapeSize(outSpec.shape);
    std::vector<double> expected;
    expected.reserve(static_cast<size_t>(std::max<int64_t>(outCount, 0)));

    for (int64_t i = 0; i < outCount; ++i) {
        double raw = self + alpha * otherValues[static_cast<size_t>(i)];
        expected.push_back(CastToStorage(raw, outSpec.dtype));
    }
    return expected;
}

void GetTolerance(aclDataType dtype, double& atol, double& rtol)
{
    switch (dtype) {
        case ACL_FLOAT:
            atol = 1e-6;
            rtol = 1e-6;
            break;
        case ACL_FLOAT16:
            atol = 1e-4;
            rtol = 1e-4;
            break;
        case ACL_BF16:
            atol = 1e-2;
            rtol = 1e-2;
            break;
        case ACL_DOUBLE:
            atol = 1e-12;
            rtol = 1e-12;
            break;
        default:
            atol = 0.0;
            rtol = 0.0;
            break;
    }
}

bool AlmostEqual(double expected, double actual, double atol, double rtol)
{
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return std::isinf(actual) && (std::signbit(expected) == std::signbit(actual));
    }
    if (std::isnan(actual) || std::isinf(actual)) {
        return false;
    }
    return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

CompareReport CompareVectors(const std::vector<double>& expected, const std::vector<double>& actual, aclDataType dtype)
{
    CompareReport report;
    if (expected.size() != actual.size()) {
        report.ok = false;
        report.badIndex = -2;
        return report;
    }

    double atol = 0.0;
    double rtol = 0.0;
    GetTolerance(dtype, atol, rtol);
    for (size_t i = 0; i < expected.size(); ++i) {
        bool equal = (atol == 0.0 && rtol == 0.0) ? (expected[i] == actual[i]) : AlmostEqual(expected[i], actual[i], atol, rtol);
        if (!equal) {
            report.ok = false;
            report.badIndex = static_cast<int64_t>(i);
            report.expected = expected[i];
            report.actual = actual[i];
            if (!std::isnan(expected[i]) && !std::isnan(actual[i])) {
                report.absError = std::fabs(actual[i] - expected[i]);
                report.relError = expected[i] != 0.0 ? report.absError / std::fabs(expected[i]) : report.absError;
            }
            return report;
        }
    }
    return report;
}

void PrintVectorSample(const char* label, const std::vector<double>& values)
{
    LOG_PRINT("  %s: [", label);
    size_t limit = std::min<size_t>(values.size(), 6U);
    for (size_t i = 0; i < limit; ++i) {
        LOG_PRINT("%s%.9g", i == 0 ? "" : ", ", values[i]);
    }
    if (values.size() > limit) {
        LOG_PRINT(", ...");
    }
    LOG_PRINT("]\n");
}

bool ExecutePrepared(aclrtStream stream, ExecuteFunc executeFunc, uint64_t workspaceSize, aclOpExecutor* executor)
{
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }

    auto ret = executeFunc(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("execute failed. ERROR: %d\n", ret); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);
    return true;
}

void RecordResult(TestStats& stats, const std::string& name, bool ok)
{
    ++stats.total;
    if (ok) {
        ++stats.passed;
        LOG_PRINT("  [PASS]\n");
    } else {
        ++stats.failed;
        LOG_PRINT("  [FAIL]\n");
    }
    LOG_PRINT("\n");
}

bool RunAclnnAddCase(aclrtStream stream, const std::string& name, const TensorSpec& selfSpec,
    const TensorSpec& otherSpec, const ScalarSpec& alphaSpec, const TensorSpec& outSpec)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    aclScalar* alpha = nullptr;

    bool ok = CreateAclTensor(selfSpec, self) == ACL_SUCCESS && CreateAclTensor(otherSpec, other) == ACL_SUCCESS &&
              CreateAclTensor(outSpec, out) == ACL_SUCCESS;
    if (ok) {
        alpha = CreateAclScalar(alphaSpec);
        ok = alpha != nullptr;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (ok) {
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);
            ok = false;
        }
    }
    if (ok) {
        ok = ExecutePrepared(stream, aclnnAdd, workspaceSize, executor);
    }

    std::vector<double> expected;
    std::vector<double> actual;
    if (ok) {
        expected = ComputeTensorTensorExpected(selfSpec, otherSpec, alphaSpec, outSpec);
        std::vector<uint8_t> resultBytes;
        ok = CopyTensorToHost(out, resultBytes);
        actual = DecodeTensorValues(resultBytes, outSpec.dtype);
    }
    if (ok) {
        PrintVectorSample("Expected", expected);
        PrintVectorSample("Actual  ", actual);
        CompareReport report = CompareVectors(expected, actual, outSpec.dtype);
        if (!report.ok) {
            LOG_PRINT("  First mismatch index: %lld expected=%.17g actual=%.17g abs=%.6e rel=%.6e\n",
                static_cast<long long>(report.badIndex), report.expected, report.actual, report.absError, report.relError);
            ok = false;
        }
    }

    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(self);
    DestroyTensor(other);
    DestroyTensor(out);
    return ok;
}

bool RunAclnnAddsCase(aclrtStream stream, const std::string& name, const TensorSpec& selfSpec,
    const ScalarSpec& otherSpec, const ScalarSpec& alphaSpec, const TensorSpec& outSpec)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    TensorHolder self;
    TensorHolder out;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;

    bool ok = CreateAclTensor(selfSpec, self) == ACL_SUCCESS && CreateAclTensor(outSpec, out) == ACL_SUCCESS;
    if (ok) {
        other = CreateAclScalar(otherSpec);
        alpha = CreateAclScalar(alphaSpec);
        ok = other != nullptr && alpha != nullptr;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (ok) {
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", ret);
            ok = false;
        }
    }
    if (ok) {
        ok = ExecutePrepared(stream, aclnnAdds, workspaceSize, executor);
    }

    std::vector<double> expected;
    std::vector<double> actual;
    if (ok) {
        expected = ComputeAddsExpected(selfSpec, otherSpec, alphaSpec, outSpec);
        std::vector<uint8_t> resultBytes;
        ok = CopyTensorToHost(out, resultBytes);
        actual = DecodeTensorValues(resultBytes, outSpec.dtype);
    }
    if (ok) {
        PrintVectorSample("Expected", expected);
        PrintVectorSample("Actual  ", actual);
        CompareReport report = CompareVectors(expected, actual, outSpec.dtype);
        if (!report.ok) {
            LOG_PRINT("  First mismatch index: %lld expected=%.17g actual=%.17g abs=%.6e rel=%.6e\n",
                static_cast<long long>(report.badIndex), report.expected, report.actual, report.absError, report.relError);
            ok = false;
        }
    }

    if (other != nullptr) {
        aclDestroyScalar(other);
    }
    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(self);
    DestroyTensor(out);
    return ok;
}

bool RunAclnnInplaceAddCase(aclrtStream stream, const std::string& name, const TensorSpec& selfSpec,
    const TensorSpec& otherSpec, const ScalarSpec& alphaSpec)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    TensorHolder self;
    TensorHolder other;
    aclScalar* alpha = nullptr;

    bool ok = CreateAclTensor(selfSpec, self) == ACL_SUCCESS && CreateAclTensor(otherSpec, other) == ACL_SUCCESS;
    if (ok) {
        alpha = CreateAclScalar(alphaSpec);
        ok = alpha != nullptr;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (ok) {
        auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret);
            ok = false;
        }
    }
    if (ok) {
        ok = ExecutePrepared(stream, aclnnInplaceAdd, workspaceSize, executor);
    }

    std::vector<double> expected;
    std::vector<double> actual;
    if (ok) {
        TensorSpec outSpec{selfSpec.shape, selfSpec.dtype, {}};
        expected = ComputeTensorTensorExpected(selfSpec, otherSpec, alphaSpec, outSpec);
        std::vector<uint8_t> resultBytes;
        ok = CopyTensorToHost(self, resultBytes);
        actual = DecodeTensorValues(resultBytes, selfSpec.dtype);
    }
    if (ok) {
        PrintVectorSample("Expected", expected);
        PrintVectorSample("Actual  ", actual);
        CompareReport report = CompareVectors(expected, actual, selfSpec.dtype);
        if (!report.ok) {
            LOG_PRINT("  First mismatch index: %lld expected=%.17g actual=%.17g abs=%.6e rel=%.6e\n",
                static_cast<long long>(report.badIndex), report.expected, report.actual, report.absError, report.relError);
            ok = false;
        }
    }

    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(self);
    DestroyTensor(other);
    return ok;
}

bool RunAclnnInplaceAddsCase(aclrtStream stream, const std::string& name, const TensorSpec& selfSpec,
    const ScalarSpec& otherSpec, const ScalarSpec& alphaSpec)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    TensorHolder self;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;

    bool ok = CreateAclTensor(selfSpec, self) == ACL_SUCCESS;
    if (ok) {
        other = CreateAclScalar(otherSpec);
        alpha = CreateAclScalar(alphaSpec);
        ok = other != nullptr && alpha != nullptr;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (ok) {
        auto ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other, alpha, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n", ret);
            ok = false;
        }
    }
    if (ok) {
        ok = ExecutePrepared(stream, aclnnInplaceAdds, workspaceSize, executor);
    }

    std::vector<double> expected;
    std::vector<double> actual;
    if (ok) {
        TensorSpec outSpec{selfSpec.shape, selfSpec.dtype, {}};
        expected = ComputeAddsExpected(selfSpec, otherSpec, alphaSpec, outSpec);
        std::vector<uint8_t> resultBytes;
        ok = CopyTensorToHost(self, resultBytes);
        actual = DecodeTensorValues(resultBytes, selfSpec.dtype);
    }
    if (ok) {
        PrintVectorSample("Expected", expected);
        PrintVectorSample("Actual  ", actual);
        CompareReport report = CompareVectors(expected, actual, selfSpec.dtype);
        if (!report.ok) {
            LOG_PRINT("  First mismatch index: %lld expected=%.17g actual=%.17g abs=%.6e rel=%.6e\n",
                static_cast<long long>(report.badIndex), report.expected, report.actual, report.absError, report.relError);
            ok = false;
        }
    }

    if (other != nullptr) {
        aclDestroyScalar(other);
    }
    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(self);
    return ok;
}

bool RunAclnnAddV3Case(aclrtStream stream, const std::string& name, const ScalarSpec& selfSpec,
    const TensorSpec& otherSpec, const ScalarSpec& alphaSpec, const TensorSpec& outSpec)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    TensorHolder other;
    TensorHolder out;
    aclScalar* self = nullptr;
    aclScalar* alpha = nullptr;

    bool ok = CreateAclTensor(otherSpec, other) == ACL_SUCCESS && CreateAclTensor(outSpec, out) == ACL_SUCCESS;
    if (ok) {
        self = CreateAclScalar(selfSpec);
        alpha = CreateAclScalar(alphaSpec);
        ok = self != nullptr && alpha != nullptr;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (ok) {
        auto ret = aclnnAddV3GetWorkspaceSize(self, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", ret);
            ok = false;
        }
    }
    if (ok) {
        ok = ExecutePrepared(stream, aclnnAddV3, workspaceSize, executor);
    }

    std::vector<double> expected;
    std::vector<double> actual;
    if (ok) {
        expected = ComputeV3Expected(selfSpec, otherSpec, alphaSpec, outSpec);
        std::vector<uint8_t> resultBytes;
        ok = CopyTensorToHost(out, resultBytes);
        actual = DecodeTensorValues(resultBytes, outSpec.dtype);
    }
    if (ok) {
        PrintVectorSample("Expected", expected);
        PrintVectorSample("Actual  ", actual);
        CompareReport report = CompareVectors(expected, actual, outSpec.dtype);
        if (!report.ok) {
            LOG_PRINT("  First mismatch index: %lld expected=%.17g actual=%.17g abs=%.6e rel=%.6e\n",
                static_cast<long long>(report.badIndex), report.expected, report.actual, report.absError, report.relError);
            ok = false;
        }
    }

    if (self != nullptr) {
        aclDestroyScalar(self);
    }
    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(other);
    DestroyTensor(out);
    return ok;
}

bool RunAclnnInplaceAddV3Case(aclrtStream stream, const std::string& name, const ScalarSpec& selfSpec,
    const TensorSpec& otherSpec, const ScalarSpec& alphaSpec)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    TensorHolder other;
    aclScalar* self = nullptr;
    aclScalar* alpha = nullptr;

    bool ok = CreateAclTensor(otherSpec, other) == ACL_SUCCESS;
    if (ok) {
        self = CreateAclScalar(selfSpec);
        alpha = CreateAclScalar(alphaSpec);
        ok = self != nullptr && alpha != nullptr;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (ok) {
        auto ret = aclnnInplaceAddV3GetWorkspaceSize(self, other.tensor, alpha, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnInplaceAddV3GetWorkspaceSize failed. ERROR: %d\n", ret);
            ok = false;
        }
    }
    if (ok) {
        ok = ExecutePrepared(stream, aclnnInplaceAddV3, workspaceSize, executor);
    }

    std::vector<double> expected;
    std::vector<double> actual;
    if (ok) {
        TensorSpec outSpec{otherSpec.shape, otherSpec.dtype, {}};
        expected = ComputeV3Expected(selfSpec, otherSpec, alphaSpec, outSpec);
        std::vector<uint8_t> resultBytes;
        ok = CopyTensorToHost(other, resultBytes);
        actual = DecodeTensorValues(resultBytes, otherSpec.dtype);
    }
    if (ok) {
        PrintVectorSample("Expected", expected);
        PrintVectorSample("Actual  ", actual);
        CompareReport report = CompareVectors(expected, actual, otherSpec.dtype);
        if (!report.ok) {
            LOG_PRINT("  First mismatch index: %lld expected=%.17g actual=%.17g abs=%.6e rel=%.6e\n",
                static_cast<long long>(report.badIndex), report.expected, report.actual, report.absError, report.relError);
            ok = false;
        }
    }

    if (self != nullptr) {
        aclDestroyScalar(self);
    }
    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(other);
    return ok;
}

bool RunAclnnAddWorkspaceOnlyCase(const std::string& name, const TensorSpec& selfSpec,
    const TensorSpec& otherSpec, const ScalarSpec& alphaSpec, const TensorSpec& outSpec)
{
    LOG_PRINT("Test case: %s\n", name.c_str());
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    aclScalar* alpha = nullptr;

    bool ok = CreateAclTensor(selfSpec, self) == ACL_SUCCESS && CreateAclTensor(otherSpec, other) == ACL_SUCCESS &&
              CreateAclTensor(outSpec, out) == ACL_SUCCESS;
    if (ok) {
        alpha = CreateAclScalar(alphaSpec);
        ok = alpha != nullptr;
    }

    if (ok) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);
            ok = false;
        } else {
            LOG_PRINT("  WorkspaceSize: %llu\n", static_cast<unsigned long long>(workspaceSize));
        }
    }

    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(self);
    DestroyTensor(other);
    DestroyTensor(out);
    return ok;
}

bool ExpectStatus(const char* label, aclnnStatus actual, aclnnStatus expected)
{
    if (actual != expected) {
        LOG_PRINT("  %s expected status %d, got %d\n", label, expected, actual);
        return false;
    }
    return true;
}

bool RunInvalidInputCases()
{
    LOG_PRINT("Test case: Invalid input validation\n");
    bool ok = true;
    TensorHolder floatTensor;
    TensorHolder floatOut;
    TensorHolder floatOther;
    TensorHolder uint32Tensor;
    TensorHolder uint32Out;
    TensorHolder badBroadcastOther;
    TensorHolder badOut;
    TensorHolder rank9Tensor;
    TensorHolder rank9Out;
    TensorHolder int32Tensor;
    TensorHolder int32Out;
    TensorHolder v3Uint8Tensor;
    TensorHolder v3Uint8Out;
    aclScalar* alphaFloat = nullptr;
    aclScalar* alphaInt = nullptr;
    aclScalar* selfScalar = nullptr;

    ok = ok && CreateAclTensor({{2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}}, floatTensor) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_FLOAT, {0, 0, 0, 0, 0, 0}}, floatOut) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_FLOAT, {1, 1, 1, 1, 1, 1}}, floatOther) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_UINT32, {1, 2, 3, 4, 5, 6}}, uint32Tensor) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_UINT32, {0, 0, 0, 0, 0, 0}}, uint32Out) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{4, 3}, ACL_FLOAT, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}, badBroadcastOther) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{3, 2}, ACL_FLOAT, {0, 0, 0, 0, 0, 0}}, badOut) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, {1}}, rank9Tensor) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, {0}}, rank9Out) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_INT32, {1, 2, 3, 4, 5, 6}}, int32Tensor) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_INT32, {0, 0, 0, 0, 0, 0}}, int32Out) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_UINT8, {1, 2, 3, 4, 5, 6}}, v3Uint8Tensor) == ACL_SUCCESS;
    ok = ok && CreateAclTensor({{2, 3}, ACL_UINT8, {0, 0, 0, 0, 0, 0}}, v3Uint8Out) == ACL_SUCCESS;
    alphaFloat = CreateAclScalar({ACL_FLOAT, 1.0});
    alphaInt = CreateAclScalar({ACL_INT32, 1.0});
    selfScalar = CreateAclScalar({ACL_FLOAT, 1.0});
    ok = ok && alphaFloat != nullptr && alphaInt != nullptr && selfScalar != nullptr;

    if (ok) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ok = ok && ExpectStatus(
            "aclnnAdd nullptr self",
            aclnnAddGetWorkspaceSize(nullptr, floatOther.tensor, alphaFloat, floatOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnAdd nullptr other",
            aclnnAddGetWorkspaceSize(floatTensor.tensor, nullptr, alphaFloat, floatOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnAdd nullptr alpha",
            aclnnAddGetWorkspaceSize(floatTensor.tensor, floatOther.tensor, nullptr, floatOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnAdd nullptr out",
            aclnnAddGetWorkspaceSize(floatTensor.tensor, floatOther.tensor, alphaFloat, nullptr, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnAdd UINT32 unsupported",
            aclnnAddGetWorkspaceSize(uint32Tensor.tensor, uint32Tensor.tensor, alphaInt, uint32Out.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnAdd bad broadcast",
            aclnnAddGetWorkspaceSize(floatTensor.tensor, badBroadcastOther.tensor, alphaFloat, floatOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnAdd bad output shape",
            aclnnAddGetWorkspaceSize(floatTensor.tensor, floatOther.tensor, alphaFloat, badOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnAdd rank > 8",
            aclnnAddGetWorkspaceSize(rank9Tensor.tensor, rank9Tensor.tensor, alphaFloat, rank9Out.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnAdd unsupported mixed output",
            aclnnAddGetWorkspaceSize(int32Tensor.tensor, floatTensor.tensor, alphaFloat, int32Out.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnAdds bad output shape",
            aclnnAddsGetWorkspaceSize(floatTensor.tensor, alphaFloat, alphaFloat, badOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnInplaceAdd nullptr self",
            aclnnInplaceAddGetWorkspaceSize(nullptr, floatOther.tensor, alphaFloat, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnInplaceAdd bad broadcast",
            aclnnInplaceAddGetWorkspaceSize(floatTensor.tensor, badBroadcastOther.tensor, alphaFloat, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnInplaceAdds nullptr alpha",
            aclnnInplaceAddsGetWorkspaceSize(floatTensor.tensor, alphaFloat, nullptr, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnAddV3 nullptr self",
            aclnnAddV3GetWorkspaceSize(nullptr, floatTensor.tensor, alphaFloat, floatOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnAddV3 unsupported UINT8 other",
            aclnnAddV3GetWorkspaceSize(selfScalar, v3Uint8Tensor.tensor, alphaFloat, v3Uint8Out.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnAddV3 bad output shape",
            aclnnAddV3GetWorkspaceSize(selfScalar, floatTensor.tensor, alphaFloat, badOut.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnAddV3 rank > 8",
            aclnnAddV3GetWorkspaceSize(selfScalar, rank9Tensor.tensor, alphaFloat, rank9Out.tensor, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
        ok = ok && ExpectStatus(
            "aclnnInplaceAddV3 nullptr other",
            aclnnInplaceAddV3GetWorkspaceSize(selfScalar, nullptr, alphaFloat, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_NULLPTR);
        ok = ok && ExpectStatus(
            "aclnnInplaceAddV3 rank > 8",
            aclnnInplaceAddV3GetWorkspaceSize(selfScalar, rank9Tensor.tensor, alphaFloat, &workspaceSize, &executor),
            ACLNN_STATUS_PARAM_INVALID);
    }

    if (alphaFloat != nullptr) {
        aclDestroyScalar(alphaFloat);
    }
    if (alphaInt != nullptr) {
        aclDestroyScalar(alphaInt);
    }
    if (selfScalar != nullptr) {
        aclDestroyScalar(selfScalar);
    }
    DestroyTensor(floatTensor);
    DestroyTensor(floatOut);
    DestroyTensor(floatOther);
    DestroyTensor(uint32Tensor);
    DestroyTensor(uint32Out);
    DestroyTensor(badBroadcastOther);
    DestroyTensor(badOut);
    DestroyTensor(rank9Tensor);
    DestroyTensor(rank9Out);
    DestroyTensor(int32Tensor);
    DestroyTensor(int32Out);
    DestroyTensor(v3Uint8Tensor);
    DestroyTensor(v3Uint8Out);
    return ok;
}

std::vector<double> MakePattern(int64_t count, double scale, double bias)
{
    std::vector<double> values;
    values.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        int64_t base = (i % 97) - 48;
        values.push_back(static_cast<double>(base) * scale + bias);
    }
    return values;
}

void RunAndRecord(TestStats& stats, const std::string& name, bool ok)
{
    (void)name;
    RecordResult(stats, name, ok);
}

} // namespace

int main()
{
    // Keep example output focused on test results; CANN INFO logs can otherwise hide PASS/FAIL lines.
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 1);
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "0", 1);

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    TestStats stats;

    RunAndRecord(stats, "Add float32 alpha=1 zero-sum",
        RunAclnnAddCase(stream, "Add float32 alpha=1 zero-sum",
            {{4, 2}, ACL_FLOAT, {1.0, -2.0, 3.5, -4.5, 5.25, -6.25, 7.5, -8.5}},
            {{4, 2}, ACL_FLOAT, {-1.0, 2.0, -3.5, 4.5, -5.25, 6.25, -7.5, 8.5}},
            {ACL_FLOAT, 1.0},
            {{4, 2}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add float32 broadcast alpha=0.5",
        RunAclnnAddCase(stream, "Add float32 broadcast alpha=0.5",
            {{4, 1}, ACL_FLOAT, {1.0, 2.0, 3.0, 4.0}},
            {{1, 5}, ACL_FLOAT, {-2.0, -1.0, 0.0, 1.0, 2.0}},
            {ACL_FLOAT, 0.5},
            {{4, 5}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add float32 alpha=0 self only",
        RunAclnnAddCase(stream, "Add float32 alpha=0 self only",
            {{2, 3}, ACL_FLOAT, {9.0, -8.0, 7.0, -6.0, 5.0, -4.0}},
            {{2, 3}, ACL_FLOAT, {100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
            {ACL_FLOAT, 0.0},
            {{2, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add empty int32 tensor",
        RunAclnnAddCase(stream, "Add empty int32 tensor",
            {{2, 0, 3}, ACL_INT32, {}},
            {{1, 0, 3}, ACL_INT32, {}},
            {ACL_INT32, 1.0},
            {{2, 0, 3}, ACL_INT32, {}}));

    RunAndRecord(stats, "Add float32 large shape negative alpha",
        RunAclnnAddCase(stream, "Add float32 large shape negative alpha",
            {{1024, 128}, ACL_FLOAT, MakePattern(1024 * 128, 0.25, 0.0)},
            {{1024, 128}, ACL_FLOAT, MakePattern(1024 * 128, 0.125, 1.0)},
            {ACL_FLOAT, -1.0},
            {{1024, 128}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add float16 alpha=2.5",
        RunAclnnAddCase(stream, "Add float16 alpha=2.5",
            {{2, 4}, ACL_FLOAT16, {1.0, 2.0, -3.0, 4.0, 0.5, -0.5, 8.0, -8.0}},
            {{2, 4}, ACL_FLOAT16, {0.5, -1.0, 2.0, -2.0, 1.5, 2.5, -4.0, 4.0}},
            {ACL_FLOAT, 2.5},
            {{2, 4}, ACL_FLOAT16, {}}));

    RunAndRecord(stats, "Add bfloat16 alpha=1 zero-sum",
        RunAclnnAddCase(stream, "Add bfloat16 alpha=1 zero-sum",
            {{2, 4}, ACL_BF16, {1.0, -2.0, 3.5, -4.5, 16.0, -32.0, 0.25, -0.125}},
            {{2, 4}, ACL_BF16, {-1.0, 2.0, -3.5, 4.5, -16.0, 32.0, -0.25, 0.125}},
            {ACL_FLOAT, 1.0},
            {{2, 4}, ACL_BF16, {}}));

    RunAndRecord(stats, "Add mixed float16 + float32 alpha=1 zero-sum",
        RunAclnnAddCase(stream, "Add mixed float16 + float32 alpha=1 zero-sum",
            {{2, 3}, ACL_FLOAT16, {1.0, -2.0, 3.0, -4.0, 5.0, -6.0}},
            {{2, 3}, ACL_FLOAT, {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}},
            {ACL_FLOAT, 1.0},
            {{2, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add mixed float32 + bfloat16 alpha=1 zero-sum",
        RunAclnnAddCase(stream, "Add mixed float32 + bfloat16 alpha=1 zero-sum",
            {{2, 3}, ACL_FLOAT, {1.0, -2.0, 3.0, -4.0, 5.0, -6.0}},
            {{2, 3}, ACL_BF16, {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}},
            {ACL_FLOAT, 1.0},
            {{2, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add mixed float16 + float32 alpha=0.5",
        RunAclnnAddCase(stream, "Add mixed float16 + float32 alpha=0.5",
            {{2, 3}, ACL_FLOAT16, {1.0, -2.0, 3.0, -4.0, 5.0, -6.0}},
            {{2, 3}, ACL_FLOAT, {0.25, 0.5, -0.75, 1.0, -1.25, 1.5}},
            {ACL_FLOAT, 0.5},
            {{2, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add int32 alpha=2",
        RunAclnnAddCase(stream, "Add int32 alpha=2",
            {{4}, ACL_INT32, {100.0, -100.0, 7.0, -7.0}},
            {{4}, ACL_INT32, {10.0, -10.0, 3.0, -3.0}},
            {ACL_INT32, 2.0},
            {{4}, ACL_INT32, {}}));

    RunAndRecord(stats, "Add int32 alpha=1 zero-sum",
        RunAclnnAddCase(stream, "Add int32 alpha=1 zero-sum",
            {{4}, ACL_INT32, {100.0, -100.0, 7.0, -7.0}},
            {{4}, ACL_INT32, {-100.0, 100.0, -7.0, 7.0}},
            {ACL_INT32, 1.0},
            {{4}, ACL_INT32, {}}));

    RunAndRecord(stats, "Add int8 alpha=2",
        RunAclnnAddCase(stream, "Add int8 alpha=2",
            {{6}, ACL_INT8, {12.0, 7.0, -8.0, -10.0, 5.0, -5.0}},
            {{6}, ACL_INT8, {3.0, 1.0, -1.0, -5.0, 8.0, -9.0}},
            {ACL_INT8, 2.0},
            {{6}, ACL_INT8, {}}));

    RunAndRecord(stats, "Add int8 alpha=1 zero-sum",
        RunAclnnAddCase(stream, "Add int8 alpha=1 zero-sum",
            {{6}, ACL_INT8, {12.0, 7.0, -8.0, -10.0, 5.0, -5.0}},
            {{6}, ACL_INT8, {-12.0, -7.0, 8.0, 10.0, -5.0, 5.0}},
            {ACL_INT8, 1.0},
            {{6}, ACL_INT8, {}}));

    RunAndRecord(stats, "Add uint8 alpha=2",
        RunAclnnAddCase(stream, "Add uint8 alpha=2",
            {{6}, ACL_UINT8, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}},
            {{6}, ACL_UINT8, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
            {ACL_INT32, 2.0},
            {{6}, ACL_UINT8, {}}));

    RunAndRecord(stats, "Add uint8 alpha=1 zero",
        RunAclnnAddCase(stream, "Add uint8 alpha=1 zero",
            {{6}, ACL_UINT8, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{6}, ACL_UINT8, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {ACL_UINT8, 1.0},
            {{6}, ACL_UINT8, {}}));

    RunAndRecord(stats, "Add int64 alpha=2",
        RunAclnnAddCase(stream, "Add int64 alpha=2",
            {{4}, ACL_INT64, {1000.0, -1000.0, 7.0, -7.0}},
            {{4}, ACL_INT64, {10.0, -10.0, 3.0, -3.0}},
            {ACL_INT64, 2.0},
            {{4}, ACL_INT64, {}}));

    RunAndRecord(stats, "Add double AICPU route workspace-only",
        RunAclnnAddWorkspaceOnlyCase("Add double AICPU route workspace-only",
            {{4}, ACL_DOUBLE, {1.0, -2.0, 3.0, -4.0}},
            {{4}, ACL_DOUBLE, {-2.0, 4.0, -6.0, 8.0}},
            {ACL_DOUBLE, 0.5},
            {{4}, ACL_DOUBLE, {}}));

    RunAndRecord(stats, "Add bool alpha=1 zero",
        RunAclnnAddCase(stream, "Add bool alpha=1 zero",
            {{6}, ACL_BOOL, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{6}, ACL_BOOL, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {ACL_BOOL, 1.0},
            {{6}, ACL_BOOL, {}}));

    RunAndRecord(stats, "Add rank-8 tensor",
        RunAclnnAddCase(stream, "Add rank-8 tensor",
            {{1, 2, 1, 2, 1, 2, 1, 2}, ACL_FLOAT, {1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0}},
            {{1, 2, 1, 2, 1, 2, 1, 2}, ACL_FLOAT, {-1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0}},
            {ACL_FLOAT, 1.0},
            {{1, 2, 1, 2, 1, 2, 1, 2}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Add float32 special values and precision probes",
        RunAclnnAddCase(stream, "Add float32 special values and precision probes",
            {{7}, ACL_FLOAT,
                {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(), std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::denorm_min(), 1.0e10, 1.00000011920928955078125}},
            {{7}, ACL_FLOAT,
                {1.0, -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                    std::numeric_limits<float>::max(), 0.0, 1.0e-5, -1.0}},
            {ACL_FLOAT, 0.5},
            {{7}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Adds float32 scalar negative alpha",
        RunAclnnAddsCase(stream, "Adds float32 scalar negative alpha",
            {{2, 3}, ACL_FLOAT, {1.0, 2.0, -3.0, 4.0, -5.0, 6.0}},
            {ACL_FLOAT, 2.0},
            {ACL_FLOAT, -1.5},
            {{2, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "Adds bool scalar false alpha=2",
        RunAclnnAddsCase(stream, "Adds bool scalar false alpha=2",
            {{6}, ACL_BOOL, {1.0, 0.0, 1.0, 0.0, 0.0, 1.0}},
            {ACL_BOOL, 0.0},
            {ACL_INT32, 2.0},
            {{6}, ACL_BOOL, {}}));

    RunAndRecord(stats, "Adds empty int32 tensor",
        RunAclnnAddsCase(stream, "Adds empty int32 tensor",
            {{2, 0, 3}, ACL_INT32, {}},
            {ACL_INT32, 2.0},
            {ACL_INT32, 2.0},
            {{2, 0, 3}, ACL_INT32, {}}));

    RunAndRecord(stats, "InplaceAdd float32 broadcast negative alpha",
        RunAclnnInplaceAddCase(stream, "InplaceAdd float32 broadcast negative alpha",
            {{4, 5}, ACL_FLOAT, MakePattern(20, 0.5, 0.0)},
            {{1, 5}, ACL_FLOAT, {1.0, 2.0, 3.0, 4.0, 5.0}},
            {ACL_FLOAT, -1.0}));

    RunAndRecord(stats, "InplaceAdds int32 alpha=0",
        RunAclnnInplaceAddsCase(stream, "InplaceAdds int32 alpha=0",
            {{2, 3}, ACL_INT32, {1.0, -2.0, 3.0, -4.0, 5.0, -6.0}},
            {ACL_INT32, 99.0},
            {ACL_INT32, 0.0}));

    RunAndRecord(stats, "AddV3 float32 alpha=1 zero-sum",
        RunAclnnAddV3Case(stream, "AddV3 float32 alpha=1 zero-sum",
            {ACL_FLOAT, 10.0},
            {{2, 3}, ACL_FLOAT, {-10.0, -10.0, -10.0, -10.0, -10.0, -10.0}},
            {ACL_FLOAT, 1.0},
            {{2, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "AddV3 empty float32 tensor",
        RunAclnnAddV3Case(stream, "AddV3 empty float32 tensor",
            {ACL_FLOAT, 10.0},
            {{0, 3}, ACL_FLOAT, {}},
            {ACL_FLOAT, 1.0},
            {{0, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "AddV3 float32 alpha=0.5",
        RunAclnnAddV3Case(stream, "AddV3 float32 alpha=0.5",
            {ACL_FLOAT, -1.0},
            {{2, 3}, ACL_FLOAT, {2.0, 4.0, 6.0, 8.0, -10.0, -12.0}},
            {ACL_FLOAT, 0.5},
            {{2, 3}, ACL_FLOAT, {}}));

    RunAndRecord(stats, "AddV3 int32 axpy path",
        RunAclnnAddV3Case(stream, "AddV3 int32 axpy path",
            {ACL_INT32, 5.0},
            {{5}, ACL_INT32, {10.0, -20.0, 30.0, -40.0, 50.0}},
            {ACL_INT32, 2.0},
            {{5}, ACL_INT32, {}}));

    RunAndRecord(stats, "AddV3 int8 mul-add path zero-sum",
        RunAclnnAddV3Case(stream, "AddV3 int8 mul-add path zero-sum",
            {ACL_INT8, 4.0},
            {{5}, ACL_INT8, {-2.0, -2.0, -2.0, -2.0, -2.0}},
            {ACL_INT8, 2.0},
            {{5}, ACL_INT8, {}}));

    RunAndRecord(stats, "AddV3 bfloat16 path",
        RunAclnnAddV3Case(stream, "AddV3 bfloat16 path",
            {ACL_BF16, 1.0},
            {{4}, ACL_BF16, {2.0, -4.0, 8.0, -16.0}},
            {ACL_FLOAT, 2.0},
            {{4}, ACL_BF16, {}}));

    RunAndRecord(stats, "InplaceAddV3 float16 negative alpha",
        RunAclnnInplaceAddV3Case(stream, "InplaceAddV3 float16 negative alpha",
            {ACL_FLOAT16, 1.0},
            {{4}, ACL_FLOAT16, {1.0, 2.0, 3.0, 4.0}},
            {ACL_FLOAT, -1.0}));

    RunAndRecord(stats, "Invalid input validation", RunInvalidInputCases());

    LOG_PRINT("Summary: %d passed, %d failed, %d total\n", stats.passed, stats.failed, stats.total);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return stats.failed == 0 ? 0 : 1;
}
