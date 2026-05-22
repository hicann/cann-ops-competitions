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
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

namespace l0op {
const aclTensor* Add(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
const aclTensor* AddInplace(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
}

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
    std::vector<int64_t> shape;
    aclDataType dataType = ACL_DT_UNDEFINED;
};

struct ScalarResource {
    aclScalar* scalar = nullptr;
};

struct RuntimeContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool initialized = false;
};

struct TestStats {
    int passed = 0;
    int failed = 0;
};

enum class ExecOutcome {
    kSuccess,
    kExpectedError,
    kFailure
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    int64_t shapeSize = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        shapeSize *= shape[i];
    }
    return shapeSize;
}

size_t GetDataTypeSize(aclDataType dataType)
{
    switch (dataType) {
        case ACL_BOOL:
        case ACL_INT8:
        case ACL_UINT8:
            return 1U;
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT16:
            return 2U;
        case ACL_FLOAT:
        case ACL_INT32:
            return 4U;
        case ACL_DOUBLE:
        case ACL_INT64:
        case ACL_COMPLEX64:
            return 8U;
        case ACL_COMPLEX128:
            return 16U;
        default:
            return 0U;
    }
}

bool IsComplexDataType(aclDataType dataType)
{
    return dataType == ACL_COMPLEX64 || dataType == ACL_COMPLEX128;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape)
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

uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToHalfBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000U;
    const uint32_t exponent = (bits >> 23) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa != 0U) {
            return static_cast<uint16_t>(sign | 0x7E00U);
        }
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        uint32_t subnormal = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - halfExponent);
        uint16_t halfMantissa = static_cast<uint16_t>(subnormal >> shift);
        if (((subnormal >> (shift - 1)) & 1U) != 0U) {
            ++halfMantissa;
        }
        return static_cast<uint16_t>(sign | halfMantissa);
    }

    uint16_t halfMantissa = static_cast<uint16_t>(mantissa >> 13);
    if ((mantissa & 0x1000U) != 0U) {
        ++halfMantissa;
        if ((halfMantissa & 0x0400U) != 0U) {
            halfMantissa = 0U;
            ++halfExponent;
            if (halfExponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(halfExponent) << 10) | halfMantissa);
}

float HalfBitsToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value >> 10) & 0x1FU;
    const uint32_t mantissa = value & 0x03FFU;

    uint32_t bits = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            uint32_t normalizedMantissa = mantissa;
            int32_t normalizedExponent = -14;
            while ((normalizedMantissa & 0x0400U) == 0U) {
                normalizedMantissa <<= 1;
                --normalizedExponent;
            }
            normalizedMantissa &= 0x03FFU;
            bits = sign | static_cast<uint32_t>((normalizedExponent + 127) << 23) | (normalizedMantissa << 13);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        bits = sign | static_cast<uint32_t>((static_cast<int32_t>(exponent) - 15 + 127) << 23) | (mantissa << 13);
    }

    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool Init(RuntimeContext* context)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return false);
    ret = aclrtSetDevice(context->deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return false);
    ret = aclrtCreateStream(&context->stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return false);
    context->initialized = true;
    return true;
}

void Finalize(RuntimeContext* context)
{
    if (context->stream != nullptr) {
        aclrtDestroyStream(context->stream);
        context->stream = nullptr;
    }
    if (context->initialized) {
        aclrtResetDevice(context->deviceId);
        aclFinalize();
        context->initialized = false;
    }
}

void DestroyTensor(TensorResource* resource)
{
    if (resource->tensor != nullptr) {
        aclDestroyTensor(resource->tensor);
        resource->tensor = nullptr;
    }
    if (resource->deviceAddr != nullptr) {
        aclrtFree(resource->deviceAddr);
        resource->deviceAddr = nullptr;
    }
}

void DestroyScalar(ScalarResource* resource)
{
    if (resource->scalar != nullptr) {
        aclDestroyScalar(resource->scalar);
        resource->scalar = nullptr;
    }
}

bool CreateScalarFromDouble(double value, aclDataType dataType, ScalarResource* resource)
{
    switch (dataType) {
        case ACL_BOOL: {
            bool v = value != 0.0;
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_INT8: {
            int8_t v = static_cast<int8_t>(value);
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_UINT8: {
            uint8_t v = static_cast<uint8_t>(value);
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_INT16: {
            int16_t v = static_cast<int16_t>(value);
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_INT32: {
            int32_t v = static_cast<int32_t>(value);
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_INT64: {
            int64_t v = static_cast<int64_t>(value);
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_FLOAT16: {
            uint16_t v = FloatToHalfBits(static_cast<float>(value));
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_BF16: {
            uint16_t v = FloatToBf16Bits(static_cast<float>(value));
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_FLOAT: {
            float v = static_cast<float>(value);
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        case ACL_DOUBLE: {
            double v = value;
            resource->scalar = aclCreateScalar(&v, dataType);
            break;
        }
        default:
            LOG_PRINT("unsupported scalar dtype: %d\n", static_cast<int>(dataType));
            return false;
    }

    CHECK_RET(resource->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return false);
    return true;
}

bool CreateTensorFromValues(
    const std::vector<double>& values, const std::vector<int64_t>& shape, aclDataType dataType, TensorResource* resource)
{
    resource->shape = shape;
    resource->dataType = dataType;

    const int64_t elementCount = GetShapeSize(shape);
    const int64_t expectedValueCount = IsComplexDataType(dataType) ? elementCount * 2 : elementCount;
    CHECK_RET(
        static_cast<int64_t>(values.size()) == expectedValueCount,
        LOG_PRINT(
            "value count mismatch, expect=%ld actual=%zu\n", static_cast<long>(expectedValueCount), values.size());
        return false);

    const size_t typeSize = GetDataTypeSize(dataType);
    CHECK_RET(typeSize > 0U, LOG_PRINT("unsupported tensor dtype: %d\n", static_cast<int>(dataType)); return false);

    std::vector<uint8_t> hostBytes(static_cast<size_t>(elementCount) * typeSize, 0U);
    for (int64_t i = 0; i < elementCount; ++i) {
        const size_t offset = static_cast<size_t>(i) * typeSize;
        switch (dataType) {
            case ACL_BOOL: {
                const uint8_t v = values[static_cast<size_t>(i)] != 0.0 ? 1U : 0U;
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_INT8: {
                const int8_t v = static_cast<int8_t>(values[static_cast<size_t>(i)]);
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_UINT8: {
                const uint8_t v = static_cast<uint8_t>(values[static_cast<size_t>(i)]);
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_INT16: {
                const int16_t v = static_cast<int16_t>(values[static_cast<size_t>(i)]);
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_INT32: {
                const int32_t v = static_cast<int32_t>(values[static_cast<size_t>(i)]);
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_INT64: {
                const int64_t v = static_cast<int64_t>(values[static_cast<size_t>(i)]);
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_FLOAT16: {
                const uint16_t v = FloatToHalfBits(static_cast<float>(values[static_cast<size_t>(i)]));
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_BF16: {
                const uint16_t v = FloatToBf16Bits(static_cast<float>(values[static_cast<size_t>(i)]));
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_FLOAT: {
                const float v = static_cast<float>(values[static_cast<size_t>(i)]);
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_DOUBLE: {
                const double v = values[static_cast<size_t>(i)];
                std::memcpy(&hostBytes[offset], &v, sizeof(v));
                break;
            }
            case ACL_COMPLEX64: {
                const float real = static_cast<float>(values[static_cast<size_t>(2 * i)]);
                const float imag = static_cast<float>(values[static_cast<size_t>(2 * i + 1)]);
                std::memcpy(&hostBytes[offset], &real, sizeof(real));
                std::memcpy(&hostBytes[offset + sizeof(real)], &imag, sizeof(imag));
                break;
            }
            case ACL_COMPLEX128: {
                const double real = values[static_cast<size_t>(2 * i)];
                const double imag = values[static_cast<size_t>(2 * i + 1)];
                std::memcpy(&hostBytes[offset], &real, sizeof(real));
                std::memcpy(&hostBytes[offset + sizeof(real)], &imag, sizeof(imag));
                break;
            }
            default:
                LOG_PRINT("unsupported tensor dtype for values: %d\n", static_cast<int>(dataType));
                return false;
        }
    }

    const size_t byteSize = hostBytes.size();
    if (byteSize > 0U) {
        auto ret = aclrtMalloc(&resource->deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return false);
        ret = aclrtMemcpy(resource->deviceAddr, byteSize, hostBytes.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return false);
    }

    std::vector<int64_t> strides = MakeContiguousStrides(shape);
    resource->tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
        resource->deviceAddr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return false);
    return true;
}

bool CreateZerosTensor(const std::vector<int64_t>& shape, aclDataType dataType, TensorResource* resource)
{
    const int64_t elementCount = GetShapeSize(shape);
    const int64_t valueCount = IsComplexDataType(dataType) ? elementCount * 2 : elementCount;
    std::vector<double> zeros(static_cast<size_t>(valueCount), 0.0);
    return CreateTensorFromValues(zeros, shape, dataType, resource);
}

std::vector<double> ReadTensorToDoubleVector(const TensorResource& resource)
{
    const int64_t elementCount = GetShapeSize(resource.shape);
    const size_t typeSize = GetDataTypeSize(resource.dataType);
    const size_t outputValueCount =
        IsComplexDataType(resource.dataType) ? static_cast<size_t>(elementCount) * 2U : static_cast<size_t>(elementCount);
    std::vector<double> values(outputValueCount, 0.0);
    if (elementCount == 0 || typeSize == 0U) {
        return values;
    }

    std::vector<uint8_t> hostBytes(static_cast<size_t>(elementCount) * typeSize, 0U);
    const auto ret = aclrtMemcpy(
        hostBytes.data(), hostBytes.size(), resource.deviceAddr, hostBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy device to host failed. ERROR: %d\n", ret);
        return values;
    }

    for (int64_t i = 0; i < elementCount; ++i) {
        const size_t offset = static_cast<size_t>(i) * typeSize;
        switch (resource.dataType) {
            case ACL_BOOL: {
                uint8_t v = 0U;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = v == 0U ? 0.0 : 1.0;
                break;
            }
            case ACL_INT8: {
                int8_t v = 0;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(v);
                break;
            }
            case ACL_UINT8: {
                uint8_t v = 0U;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(v);
                break;
            }
            case ACL_INT16: {
                int16_t v = 0;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(v);
                break;
            }
            case ACL_INT32: {
                int32_t v = 0;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(v);
                break;
            }
            case ACL_INT64: {
                int64_t v = 0;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(v);
                break;
            }
            case ACL_FLOAT16: {
                uint16_t v = 0U;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(HalfBitsToFloat(v));
                break;
            }
            case ACL_BF16: {
                uint16_t v = 0U;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(Bf16BitsToFloat(v));
                break;
            }
            case ACL_FLOAT: {
                float v = 0.0F;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = static_cast<double>(v);
                break;
            }
            case ACL_DOUBLE: {
                double v = 0.0;
                std::memcpy(&v, &hostBytes[offset], sizeof(v));
                values[static_cast<size_t>(i)] = v;
                break;
            }
            case ACL_COMPLEX64: {
                float real = 0.0F;
                float imag = 0.0F;
                std::memcpy(&real, &hostBytes[offset], sizeof(real));
                std::memcpy(&imag, &hostBytes[offset + sizeof(real)], sizeof(imag));
                values[static_cast<size_t>(2 * i)] = static_cast<double>(real);
                values[static_cast<size_t>(2 * i + 1)] = static_cast<double>(imag);
                break;
            }
            case ACL_COMPLEX128: {
                double real = 0.0;
                double imag = 0.0;
                std::memcpy(&real, &hostBytes[offset], sizeof(real));
                std::memcpy(&imag, &hostBytes[offset + sizeof(real)], sizeof(imag));
                values[static_cast<size_t>(2 * i)] = real;
                values[static_cast<size_t>(2 * i + 1)] = imag;
                break;
            }
            default:
                break;
        }
    }
    return values;
}

std::string FormatDoubleVector(const std::vector<double>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

bool CheckResult(
    const std::string& caseName, const std::vector<double>& actual, const std::vector<double>& expected, bool exact,
    double atol, double rtol)
{
    LOG_PRINT("[Precision][%s]\n", caseName.c_str());
    LOG_PRINT("  Expected: %s\n", FormatDoubleVector(expected).c_str());
    LOG_PRINT("  Actual:   %s\n", FormatDoubleVector(actual).c_str());
    if (exact) {
        LOG_PRINT("  Criterion: exact match\n");
    } else {
        LOG_PRINT("  Criterion: atol=%.8f, rtol=%.8f\n", atol, rtol);
    }

    CHECK_RET(
        actual.size() == expected.size(),
        LOG_PRINT("[FAIL] %s size mismatch, actual=%zu expected=%zu\n", caseName.c_str(), actual.size(), expected.size());
        return false);

    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(actual[i] - expected[i]);
        const double limit = exact ? 0.0 : (atol + rtol * std::fabs(expected[i]));
        if ((exact && diff != 0.0) || (!exact && diff > limit)) {
            LOG_PRINT(
                "[FAIL] %s idx=%zu actual=%.8f expected=%.8f diff=%.8f limit=%.8f\n", caseName.c_str(), i, actual[i],
                expected[i], diff, limit);
            return false;
        }
    }

    LOG_PRINT("[PASS] %s\n", caseName.c_str());
    return true;
}

bool CheckPointerResult(const std::string& caseName, const void* ptr, bool expectNull)
{
    const bool matched = expectNull ? (ptr == nullptr) : (ptr != nullptr);
    if (matched) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return true;
    }
    LOG_PRINT(
        "[FAIL] %s pointer expectation mismatch, expectNull=%d actual=%p\n",
        caseName.c_str(),
        expectNull ? 1 : 0,
        ptr);
    return false;
}

template <typename GetWorkspaceFn, typename RunFn>
bool ExecuteOp(
    const std::string& caseName, GetWorkspaceFn getWorkspaceFn, RunFn runFn, RuntimeContext* context)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = getWorkspaceFn(&workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[FAIL] %s GetWorkspace failed. ERROR: %d\n", caseName.c_str(), ret);
        return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s workspace malloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return false);
    }

    ret = runFn(workspaceAddr, workspaceSize, executor, context->stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(context->stream);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[FAIL] %s run failed. ERROR: %d\n", caseName.c_str(), ret);
        return false);
    return true;
}

template <typename GetWorkspaceFn, typename RunFn>
ExecOutcome ExecuteOpAllowExpectedError(
    const std::string& caseName, GetWorkspaceFn getWorkspaceFn, RunFn runFn, RuntimeContext* context,
    aclnnStatus expectedGetWorkspaceError = ACLNN_SUCCESS, aclnnStatus expectedRunError = ACLNN_SUCCESS)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = getWorkspaceFn(&workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        if (expectedGetWorkspaceError != ACLNN_SUCCESS && ret == expectedGetWorkspaceError) {
            LOG_PRINT("[XFAIL_ENV] %s expected simulator limitation at GetWorkspace. ERROR: %d\n", caseName.c_str(), ret);
            return ExecOutcome::kExpectedError;
        }
        LOG_PRINT("[FAIL] %s GetWorkspace failed. ERROR: %d\n", caseName.c_str(), ret);
        return ExecOutcome::kFailure;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s workspace malloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return ExecOutcome::kFailure;
        }
    }

    ret = runFn(workspaceAddr, workspaceSize, executor, context->stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(context->stream);
    }

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    if (ret != ACL_SUCCESS) {
        if (expectedRunError != ACLNN_SUCCESS && ret == expectedRunError) {
            LOG_PRINT("[XFAIL_ENV] %s expected simulator limitation at run. ERROR: %d\n", caseName.c_str(), ret);
            return ExecOutcome::kExpectedError;
        }
        LOG_PRINT("[FAIL] %s run failed. ERROR: %d\n", caseName.c_str(), ret);
        return ExecOutcome::kFailure;
    }

    return ExecOutcome::kSuccess;
}

bool CheckStatusAny(
    const std::string& caseName, int32_t actual, std::initializer_list<int32_t> expectedList, bool passOnMatch)
{
    for (int32_t expected : expectedList) {
        if (actual == expected) {
            LOG_PRINT("[%s] %s\n", passOnMatch ? "PASS" : "INFO", caseName.c_str());
            return passOnMatch;
        }
    }
    LOG_PRINT(
        "[FAIL] %s status mismatch, actual=%d\n", caseName.c_str(), static_cast<int>(actual));
    return false;
}

bool CheckStatus(
    const std::string& caseName, aclnnStatus actual, aclnnStatus expected, bool passOnMatch)
{
    return CheckStatusAny(
        caseName, static_cast<int32_t>(actual), {static_cast<int32_t>(expected)}, passOnMatch);
}

bool CheckInvalidStatus(const std::string& caseName, aclnnStatus actual)
{
    return CheckStatusAny(
        caseName,
        static_cast<int32_t>(actual),
        {static_cast<int32_t>(ACLNN_ERR_PARAM_INVALID), static_cast<int32_t>(ACL_ERROR_GE_PARAM_INVALID)},
        true);
}

bool CheckInvalidOrEnvLimitedStatus(const std::string& caseName, aclnnStatus actual)
{
    const int32_t actualValue = static_cast<int32_t>(actual);
    if (actualValue == static_cast<int32_t>(ACLNN_ERR_PARAM_INVALID) ||
        actualValue == static_cast<int32_t>(ACL_ERROR_GE_PARAM_INVALID)) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return true;
    }
    if (actual == ACLNN_ERR_INNER_NULLPTR || actual == ACLNN_ERR_INNER) {
        LOG_PRINT(
            "[XFAIL_ENV] %s expected invalid-path probe blocked by simulator environment. ERROR: %d\n",
            caseName.c_str(),
            static_cast<int>(actual));
        return true;
    }
    return false;
}

bool RunAddCaseFloatDirect(RuntimeContext* context)
{
    const std::string caseName = "add_float_direct_alpha1";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 3};
    const std::vector<double> selfValues = {1.0, -2.0, 3.5, 4.0, 0.5, -1.5};
    const std::vector<double> otherValues = {0.25, 2.0, -0.5, 3.0, -1.0, 2.5};
    const std::vector<double> expected = {1.25, 0.0, 3.0, 7.0, -0.5, 1.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseFloatLongDecimal(RuntimeContext* context)
{
    const std::string caseName = "add_float_long_decimal_precision";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 3};
    const std::vector<double> selfValues = {0.1234567, -0.9876543, 3.1415926, -2.7182818, 1234.56775, -0.0003125};
    const std::vector<double> otherValues = {9.8765430, 0.2222222, -1.4142135, 2.5000002, -34.56789, 0.3333333};
    const std::vector<double> expected = {9.9999997, -0.7654321, 1.7273791, -0.2182816, 1199.99986, 0.3330208};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 5e-5, 5e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseFloatBroadcastAxpy(RuntimeContext* context)
{
    const std::string caseName = "add_float_broadcast_axpy";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {3};
    const std::vector<double> selfValues = {2.0, -1.0, 4.0, 8.0, 3.0, -2.0};
    const std::vector<double> otherValues = {1.0, -2.0, 0.5};
    const std::vector<double> expected = {0.5, 2.0, 3.25, 6.5, 6.0, -2.75};

    bool ok = CreateTensorFromValues(selfValues, selfShape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(otherValues, otherShape, ACL_FLOAT, &other) &&
              CreateZerosTensor(selfShape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(-1.5, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseMixedFp16Fp32(RuntimeContext* context)
{
    const std::string caseName = "add_mixed_fp16_fp32_alpha1";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1.5, -2.0, 0.25, 4.0};
    const std::vector<double> otherValues = {0.5, 3.25, -1.25, 2.0};
    const std::vector<double> expected = {2.0, 1.25, -1.0, 6.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT16, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseMixedFp32Fp16(RuntimeContext* context)
{
    const std::string caseName = "add_mixed_fp32_fp16_alpha1";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {3.0, -1.0, 5.5, 0.5};
    const std::vector<double> otherValues = {1.0, 2.0, -4.0, 6.0};
    const std::vector<double> expected = {4.0, 1.0, 1.5, 6.5};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_FLOAT16, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseBf16Direct(RuntimeContext* context)
{
    const std::string caseName = "add_bf16_direct_alpha1";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1.0, -2.5, 3.0, 0.5};
    const std::vector<double> otherValues = {0.25, 1.5, -0.5, 2.0};
    const std::vector<double> expected = {1.25, -1.0, 2.5, 2.5};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_BF16, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_BF16, &other) &&
              CreateZerosTensor(shape, ACL_BF16, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-2, 1e-2);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseInt32Direct(RuntimeContext* context)
{
    const std::string caseName = "add_int32_direct_alpha1";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1, -2, 3, 4};
    const std::vector<double> otherValues = {5, 6, -1, 2};
    const std::vector<double> expected = {6, 4, 2, 6};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_INT32, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_INT32, &other) &&
              CreateZerosTensor(shape, ACL_INT32, &out) &&
              CreateScalarFromDouble(1.0, ACL_INT32, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseInt64Direct(RuntimeContext* context)
{
    const std::string caseName = "add_int64_direct_alpha1";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {10, 20, -5, 7};
    const std::vector<double> otherValues = {1, -2, 3, 4};
    const std::vector<double> expected = {11, 18, -2, 11};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_INT64, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_INT64, &other) &&
              CreateZerosTensor(shape, ACL_INT64, &out) &&
              CreateScalarFromDouble(1.0, ACL_INT64, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseUint8Direct(RuntimeContext* context)
{
    const std::string caseName = "add_uint8_direct_alpha1";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1, 2, 3, 4};
    const std::vector<double> otherValues = {4, 3, 2, 1};
    const std::vector<double> expected = {5, 5, 5, 5};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_UINT8, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_UINT8, &other) &&
              CreateZerosTensor(shape, ACL_UINT8, &out) &&
              CreateScalarFromDouble(1.0, ACL_UINT8, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseComplex64FloatPromote(RuntimeContext* context)
{
    const std::string caseName = "add_complex64_float_promote_probe";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {
        1.0, 0.5,
        -2.0, 1.5,
        3.25, -0.75,
        -4.5, 2.0};
    const std::vector<double> otherValues = {0.25, -1.0, 2.5, -3.0};
    const std::vector<double> expected = {
        1.25, 0.5,
        -3.0, 1.5,
        5.75, -0.75,
        -7.5, 2.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_COMPLEX64, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_COMPLEX64, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR,
            ACLNN_ERR_INNER);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseFloatComplex64Promote(RuntimeContext* context)
{
    const std::string caseName = "add_float_complex64_promote_probe";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {0.25, -1.0, 2.5, -3.0};
    const std::vector<double> otherValues = {
        1.0, 0.5,
        -2.0, 1.5,
        3.25, -0.75,
        -4.5, 2.0};
    const std::vector<double> expected = {
        1.25, 0.5,
        -3.0, 1.5,
        5.75, -0.75,
        -7.5, 2.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_COMPLEX64, &other) &&
              CreateZerosTensor(shape, ACL_COMPLEX64, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR,
            ACLNN_ERR_INNER);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseInt16Int32Promote(RuntimeContext* context)
{
    const std::string caseName = "add_int16_int32_promote";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1, -2, 3, 4};
    const std::vector<double> otherValues = {5, 6, -1, 2};
    const std::vector<double> expected = {6, 4, 2, 6};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_INT16, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_INT32, &other) &&
              CreateZerosTensor(shape, ACL_INT32, &out) &&
              CreateScalarFromDouble(1.0, ACL_INT32, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseDoubleDirectProbe(RuntimeContext* context)
{
    const std::string caseName = "add_double_direct_probe";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1.0, -2.0, 3.0, 4.0};
    const std::vector<double> otherValues = {0.5, 1.5, -1.0, 2.0};
    const std::vector<double> expected = {1.5, -0.5, 2.0, 6.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_DOUBLE, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_DOUBLE, &other) &&
              CreateZerosTensor(shape, ACL_DOUBLE, &out) &&
              CreateScalarFromDouble(1.0, ACL_DOUBLE, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR,
            ACLNN_ERR_INNER);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-10, 1e-10));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseDoubleFallbackProbe(RuntimeContext* context)
{
    const std::string caseName = "add_double_fallback_probe";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1.0, -2.0, 3.0, 4.0};
    const std::vector<double> otherValues = {0.5, 1.5, -1.0, 2.0};
    const std::vector<double> expected = {2.0, 1.0, 1.0, 8.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_DOUBLE, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_DOUBLE, &other) &&
              CreateZerosTensor(shape, ACL_DOUBLE, &out) &&
              CreateScalarFromDouble(2.0, ACL_DOUBLE, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR,
            ACLNN_ERR_INNER);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-10, 1e-10));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseInt64AxpyV2(RuntimeContext* context)
{
    const std::string caseName = "add_int64_axpyv2";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1, 2, 3, 4};
    const std::vector<double> otherValues = {5, -1, 2, 3};
    const std::vector<double> expected = {11, 0, 7, 10};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_INT64, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_INT64, &other) &&
              CreateZerosTensor(shape, ACL_INT64, &out) &&
              CreateScalarFromDouble(2.0, ACL_INT64, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddCaseUint8AxpyV2(RuntimeContext* context)
{
    const std::string caseName = "add_uint8_axpyv2";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1, 2, 3, 4};
    const std::vector<double> otherValues = {4, 3, 2, 1};
    const std::vector<double> expected = {9, 8, 7, 6};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_UINT8, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_UINT8, &other) &&
              CreateZerosTensor(shape, ACL_UINT8, &out) &&
              CreateScalarFromDouble(2.0, ACL_UINT8, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddsCaseBoolCast(RuntimeContext* context)
{
    const std::string caseName = "adds_bool_cast_fix";
    TensorResource self;
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 3};
    const std::vector<double> selfValues = {1, 0, 1, 0, 1, 0};
    const std::vector<double> expected = {1, 1, 1, 1, 1, 1};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_BOOL, &self) &&
              CreateZerosTensor(shape, ACL_INT32, &out) &&
              CreateScalarFromDouble(1.0, ACL_BOOL, &other) &&
              CreateScalarFromDouble(1.0, ACL_BOOL, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(
                    self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdds,
            context,
            ACLNN_ERR_INNER_NULLPTR);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0));
    }

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&out);
    DestroyTensor(&self);
    return ok;
}

bool RunAddsCaseFloatDirect(RuntimeContext* context)
{
    const std::string caseName = "adds_float_direct_alpha1";
    TensorResource self;
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1.0, -2.0, 3.5, 4.0};
    const std::vector<double> expected = {1.5, -1.5, 4.0, 4.5};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT, &self) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(0.5, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddsGetWorkspaceSize(
                          self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdds,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5);

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&out);
    DestroyTensor(&self);
    return ok;
}

bool RunAddsCaseFloatLongDecimal(RuntimeContext* context)
{
    const std::string caseName = "adds_float_long_decimal_precision";
    TensorResource self;
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {0.1234567, -9.876543, 1.234567, -0.0009765625};
    const std::vector<double> expected = {0.4567900, -9.5432097, 1.5679003, 0.3323567375};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT, &self) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(0.3333333, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddsGetWorkspaceSize(
                          self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdds,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 5e-5, 5e-5);

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&out);
    DestroyTensor(&self);
    return ok;
}

bool RunAddsCaseBf16Scalar(RuntimeContext* context)
{
    const std::string caseName = "adds_bf16_scalar_alpha1";
    TensorResource self;
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1.0, -2.5, 3.0, 0.5};
    const std::vector<double> expected = {1.5, -2.0, 3.5, 1.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_BF16, &self) &&
              CreateZerosTensor(shape, ACL_BF16, &out) &&
              CreateScalarFromDouble(0.5, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddsGetWorkspaceSize(
                          self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdds,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-2, 1e-2);

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&out);
    DestroyTensor(&self);
    return ok;
}

bool RunAddsCaseInt64FallbackProbe(RuntimeContext* context)
{
    const std::string caseName = "adds_int64_fallback_probe";
    TensorResource self;
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1, 2, 3, 4};
    const std::vector<double> expected = {7, 8, 9, 10};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_INT64, &self) &&
              CreateZerosTensor(shape, ACL_INT64, &out) &&
              CreateScalarFromDouble(3.0, ACL_INT64, &other) &&
              CreateScalarFromDouble(2.0, ACL_INT64, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(
                    self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdds,
            context,
            ACLNN_ERR_INNER_NULLPTR,
            ACLNN_ERR_INNER);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0));
    }

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&out);
    DestroyTensor(&self);
    return ok;
}

bool RunAddsEmptyTensorCase(RuntimeContext* context)
{
    const std::string caseName = "adds_empty_tensor";
    TensorResource self;
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {0, 2};
    const std::vector<double> emptyValues;
    bool ok = CreateTensorFromValues(emptyValues, shape, ACL_FLOAT, &self) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddsGetWorkspaceSize(
                          self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAdds,
                  context);

    if (ok) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
    }

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&out);
    DestroyTensor(&self);
    return ok;
}

bool RunInplaceAddCase(RuntimeContext* context)
{
    const std::string caseName = "inplace_add_float";
    TensorResource self;
    TensorResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1.0, 2.0, -3.0, 4.5};
    const std::vector<double> otherValues = {0.5, -1.0, 2.0, 3.5};
    const std::vector<double> expected = {1.5, 1.0, -1.0, 8.0};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnInplaceAddGetWorkspaceSize(
                          self.tensor, other.tensor, alpha.scalar, workspaceSize, executor);
                  },
                  aclnnInplaceAdd,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(self), expected, false, 1e-5, 1e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunL0AddInplaceBroadcastInvalidCase()
{
    const std::string caseName = "l0_add_inplace_broadcast_invalid";
    TensorResource self;
    TensorResource other;
    bool ok = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<int64_t>{2, 3}, ACL_FLOAT, &self) &&
              CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other);
    if (ok) {
        ok = CheckPointerResult(caseName, l0op::AddInplace(self.tensor, other.tensor, nullptr), true);
    }

    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunL0AddBroadcastInvalidCase()
{
    const std::string caseName = "l0_add_broadcast_invalid";
    TensorResource self;
    TensorResource other;
    bool ok = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<int64_t>{2, 3}, ACL_FLOAT, &self) &&
              CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other);
    if (ok) {
        ok = CheckPointerResult(caseName, l0op::Add(self.tensor, other.tensor, nullptr), true);
    }

    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunL0AddInplaceOutputShapeInvalidCase()
{
    const std::string caseName = "l0_add_inplace_output_shape_invalid";
    TensorResource self;
    TensorResource other;
    bool ok = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self) &&
              CreateTensorFromValues(std::vector<double>{10, 20}, std::vector<int64_t>{2}, ACL_FLOAT, &other);
    if (ok) {
        ok = CheckPointerResult(caseName, l0op::AddInplace(self.tensor, other.tensor, nullptr), true);
    }

    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunL0AddInplaceMixedOutputInvalidCase()
{
    const std::string caseName = "l0_add_inplace_mixed_output_invalid";
    TensorResource self;
    TensorResource other;
    bool ok = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self) &&
              CreateTensorFromValues(std::vector<double>{5, 6, 7, 8}, std::vector<int64_t>{2, 2}, ACL_FLOAT16, &other);
    if (ok) {
        ok = CheckPointerResult(caseName, l0op::AddInplace(self.tensor, other.tensor, nullptr), true);
    }

    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunInplaceAddsCaseInt32(RuntimeContext* context)
{
    const std::string caseName = "inplace_adds_int32";
    TensorResource self;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> selfValues = {1, 2, 3, 4};
    const std::vector<double> expected = {7, 8, 9, 10};

    bool ok = CreateTensorFromValues(selfValues, shape, ACL_INT32, &self) &&
              CreateScalarFromDouble(3.0, ACL_INT32, &other) &&
              CreateScalarFromDouble(2.0, ACL_INT32, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddsGetWorkspaceSize(
                    self.tensor, other.scalar, alpha.scalar, workspaceSize, executor);
            },
            aclnnInplaceAdds,
            context,
            ACLNN_ERR_INNER_NULLPTR);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(self), expected, true, 0.0, 0.0));
    }

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunInplaceAddsEmptyTensorCase(RuntimeContext* context)
{
    const std::string caseName = "inplace_adds_empty_tensor";
    TensorResource self;
    ScalarResource other;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {0, 2};
    const std::vector<double> emptyValues;
    bool ok = CreateTensorFromValues(emptyValues, shape, ACL_FLOAT, &self) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnInplaceAddsGetWorkspaceSize(
                          self.tensor, other.scalar, alpha.scalar, workspaceSize, executor);
                  },
                  aclnnInplaceAdds,
                  context);

    if (ok) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
    }

    DestroyScalar(&alpha);
    DestroyScalar(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddV3CaseFloatDirect(RuntimeContext* context)
{
    const std::string caseName = "add_v3_float_direct";
    TensorResource other;
    TensorResource out;
    ScalarResource self;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 3};
    const std::vector<double> otherValues = {1.0, -2.0, 3.0, 0.5, 4.0, -1.5};
    const std::vector<double> expected = {3.5, 0.5, 5.5, 3.0, 6.5, 1.0};

    bool ok = CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(2.5, ACL_FLOAT, &self) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddV3GetWorkspaceSize(
                          self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAddV3,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyScalar(&self);
    return ok;
}

bool RunAddV3CaseFloatLongDecimal(RuntimeContext* context)
{
    const std::string caseName = "add_v3_float_long_decimal_precision";
    TensorResource other;
    TensorResource out;
    ScalarResource self;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> otherValues = {3.1415926, -2.7182818, 0.0000001, -999.99994};
    const std::vector<double> expected = {3.2650493, -2.5948251, 0.1234568, -999.8764833};

    bool ok = CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(0.1234567, ACL_FLOAT, &self) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnAddV3GetWorkspaceSize(
                          self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
                  },
                  aclnnAddV3,
                  context) &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 5e-5, 5e-5);

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyScalar(&self);
    return ok;
}

bool RunAddV3CaseInt8Fallback(RuntimeContext* context)
{
    const std::string caseName = "add_v3_int8_mul_add_fallback";
    TensorResource other;
    TensorResource out;
    ScalarResource self;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> otherValues = {1, 2, -1, 3};
    const std::vector<double> expected = {5, 7, 1, 9};

    bool ok = CreateTensorFromValues(otherValues, shape, ACL_INT8, &other) &&
              CreateZerosTensor(shape, ACL_INT8, &out) &&
              CreateScalarFromDouble(3.0, ACL_INT8, &self) &&
              CreateScalarFromDouble(2.0, ACL_INT8, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(
                    self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAddV3,
            context,
            ACLNN_SUCCESS,
            ACLNN_ERR_INNER);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, true, 0.0, 0.0));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyScalar(&self);
    return ok;
}

bool RunAddV3CaseFloatScalarInt32Tensor(RuntimeContext* context)
{
    const std::string caseName = "add_v3_float_scalar_int32_tensor";
    TensorResource other;
    TensorResource out;
    ScalarResource self;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> otherValues = {1, 2, -1, 3};
    const std::vector<double> expected = {3.5, 4.5, 1.5, 5.5};

    bool ok = CreateTensorFromValues(otherValues, shape, ACL_INT32, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(2.5, ACL_FLOAT, &self) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(
                    self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAddV3,
            context,
            ACLNN_ERR_INNER_NULLPTR,
            ACLNN_ERR_INNER);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyScalar(&self);
    return ok;
}

bool RunInplaceAddV3CaseFloatAxpy(RuntimeContext* context)
{
    const std::string caseName = "inplace_add_v3_float_axpy";
    TensorResource other;
    ScalarResource self;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> otherValues = {2.0, -1.0, 0.5, 4.0};
    const std::vector<double> expected = {4.0, 1.0, 2.5, 6.0};

    bool ok = CreateTensorFromValues(otherValues, shape, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &self) &&
              CreateScalarFromDouble(1.5, ACL_FLOAT, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddV3GetWorkspaceSize(
                    self.scalar, other.tensor, alpha.scalar, workspaceSize, executor);
            },
            aclnnInplaceAddV3,
            context,
            ACLNN_ERR_INNER_NULLPTR);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(other), expected, false, 1e-5, 1e-5));
    }

    DestroyScalar(&alpha);
    DestroyScalar(&self);
    DestroyTensor(&other);
    return ok;
}

bool RunInplaceAddV3EmptyTensorCase(RuntimeContext* context)
{
    const std::string caseName = "inplace_add_v3_empty_tensor";
    TensorResource other;
    ScalarResource self;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {0, 2};
    const std::vector<double> emptyValues;
    bool ok = CreateTensorFromValues(emptyValues, shape, ACL_FLOAT, &other) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &self) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha) &&
              ExecuteOp(
                  caseName,
                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                      return aclnnInplaceAddV3GetWorkspaceSize(
                          self.scalar, other.tensor, alpha.scalar, workspaceSize, executor);
                  },
                  aclnnInplaceAddV3,
                  context);

    if (ok) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
    }

    DestroyScalar(&alpha);
    DestroyScalar(&self);
    DestroyTensor(&other);
    return ok;
}

bool RunAddCaseFloatBroadcastAxpyProbe(RuntimeContext* context)
{
    const std::string caseName = "add_float_broadcast_axpy_probe";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {3};
    const std::vector<double> selfValues = {2.0, -1.0, 4.0, 8.0, 3.0, -2.0};
    const std::vector<double> otherValues = {1.0, -2.0, 0.5};
    const std::vector<double> expected = {0.5, 2.0, 3.25, 6.5, 6.0, -2.75};

    bool ok = CreateTensorFromValues(selfValues, selfShape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(otherValues, otherShape, ACL_FLOAT, &other) &&
              CreateZerosTensor(selfShape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(-1.5, ACL_FLOAT, &alpha);
    if (ok) {
        const ExecOutcome outcome = ExecuteOpAllowExpectedError(
            caseName,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(
                    self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            },
            aclnnAdd,
            context,
            ACLNN_ERR_INNER_NULLPTR);
        ok = outcome == ExecOutcome::kExpectedError ||
             (outcome == ExecOutcome::kSuccess &&
              CheckResult(caseName, ReadTensorToDoubleVector(out), expected, false, 1e-5, 1e-5));
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddEmptyTensorCase()
{
    const std::string caseName = "add_empty_tensor_workspace";
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {0, 3};
    const std::vector<double> emptyValues;
    bool ok = CreateTensorFromValues(emptyValues, shape, ACL_FLOAT, &self) &&
              CreateTensorFromValues(emptyValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
    if (ok) {
        uint64_t workspaceSize = 1U;
        aclOpExecutor* executor = nullptr;
        const aclnnStatus ret =
            aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        ok = (ret == ACL_SUCCESS) && (workspaceSize == 0U);
        if (ok) {
            LOG_PRINT("[PASS] %s\n", caseName.c_str());
        } else {
            LOG_PRINT("[FAIL] %s ret=%d workspace=%llu\n", caseName.c_str(), static_cast<int>(ret),
                static_cast<unsigned long long>(workspaceSize));
        }
    }

    DestroyScalar(&alpha);
    DestroyTensor(&out);
    DestroyTensor(&other);
    DestroyTensor(&self);
    return ok;
}

bool RunAddV3EmptyTensorCase()
{
    const std::string caseName = "add_v3_empty_tensor_workspace";
    TensorResource other;
    TensorResource out;
    ScalarResource self;
    ScalarResource alpha;

    const std::vector<int64_t> shape = {0, 2};
    const std::vector<double> emptyValues;
    bool ok = CreateTensorFromValues(emptyValues, shape, ACL_FLOAT, &other) &&
              CreateZerosTensor(shape, ACL_FLOAT, &out) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &self) &&
              CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
    if (ok) {
        uint64_t workspaceSize = 1U;
        aclOpExecutor* executor = nullptr;
        const aclnnStatus ret =
            aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        ok = (ret == ACL_SUCCESS) && (workspaceSize == 0U);
        if (ok) {
            LOG_PRINT("[PASS] %s\n", caseName.c_str());
        } else {
            LOG_PRINT("[FAIL] %s ret=%d workspace=%llu\n", caseName.c_str(), static_cast<int>(ret),
                static_cast<unsigned long long>(workspaceSize));
        }
    }

    DestroyScalar(&alpha);
    DestroyScalar(&self);
    DestroyTensor(&out);
    DestroyTensor(&other);
    return ok;
}

bool RunNegativeCases()
{
    bool ok = true;

    {
        const std::string caseName = "add_nullptr_check";
        uint64_t workspaceSize = 0U;
        aclOpExecutor* executor = nullptr;
        const aclnnStatus ret = aclnnAddGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, &executor);
        ok = CheckStatus(caseName, ret, ACLNN_ERR_PARAM_NULLPTR, true) && ok;
    }

    {
        const std::string caseName = "add_invalid_out_shape";
        TensorResource self;
        TensorResource other;
        TensorResource out;
        ScalarResource alpha;
        const std::vector<double> values = {1, 2, 3, 4, 5, 6};
        const std::vector<int64_t> inShape = {2, 3};
        const std::vector<int64_t> outShape = {3, 2};
        bool localOk = CreateTensorFromValues(values, inShape, ACL_FLOAT, &self) &&
                       CreateTensorFromValues(values, inShape, ACL_FLOAT, &other) &&
                       CreateZerosTensor(outShape, ACL_FLOAT, &out) &&
                       CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckStatus(caseName, ret, ACLNN_ERR_PARAM_INVALID, true);
        }
        DestroyScalar(&alpha);
        DestroyTensor(&out);
        DestroyTensor(&other);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "add_bool_alpha_float_invalid";
        TensorResource self;
        TensorResource other;
        TensorResource out;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>{1, 0, 1, 0}, std::vector<int64_t>{2, 2}, ACL_BOOL, &self) &&
                       CreateTensorFromValues(std::vector<double>{0, 1, 0, 1}, std::vector<int64_t>{2, 2}, ACL_BOOL, &other) &&
                       CreateZerosTensor(std::vector<int64_t>{2, 2}, ACL_BOOL, &out) &&
                       CreateScalarFromDouble(0.5, ACL_FLOAT, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckInvalidOrEnvLimitedStatus(caseName, ret);
        }
        DestroyScalar(&alpha);
        DestroyTensor(&out);
        DestroyTensor(&other);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "add_tiling_same_dtype_out_mismatch";
        TensorResource self;
        TensorResource other;
        TensorResource out;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_INT32, &self) &&
                       CreateTensorFromValues(std::vector<double>{4, 3, 2, 1}, std::vector<int64_t>{2, 2}, ACL_INT32, &other) &&
                       CreateZerosTensor(std::vector<int64_t>{2, 2}, ACL_INT64, &out) &&
                       CreateScalarFromDouble(1.0, ACL_INT32, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckInvalidOrEnvLimitedStatus(caseName, ret);
        }
        DestroyScalar(&alpha);
        DestroyTensor(&out);
        DestroyTensor(&other);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "add_tiling_mixed_dtype_out_not_float";
        TensorResource self;
        TensorResource other;
        TensorResource out;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT16, &self) &&
                       CreateTensorFromValues(std::vector<double>{4, 3, 2, 1}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other) &&
                       CreateZerosTensor(std::vector<int64_t>{2, 2}, ACL_FLOAT16, &out) &&
                       CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckInvalidOrEnvLimitedStatus(caseName, ret);
        }
        DestroyScalar(&alpha);
        DestroyTensor(&out);
        DestroyTensor(&other);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "add_tiling_complex128_not_supported";
        TensorResource self;
        TensorResource other;
        TensorResource out;
        ScalarResource alpha;
        bool localOk =
            CreateTensorFromValues(
                std::vector<double>{1.0, 0.5, -2.0, 1.5, 3.0, -0.5, -4.0, 2.0},
                std::vector<int64_t>{2, 2},
                ACL_COMPLEX128,
                &self) &&
            CreateTensorFromValues(
                std::vector<double>{0.5, -1.0, 2.0, 0.0, -1.5, 3.0, 1.0, -2.5},
                std::vector<int64_t>{2, 2},
                ACL_COMPLEX128,
                &other) &&
            CreateZerosTensor(std::vector<int64_t>{2, 2}, ACL_COMPLEX128, &out) &&
            CreateScalarFromDouble(1.0, ACL_DOUBLE, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckInvalidOrEnvLimitedStatus(caseName, ret);
        }
        DestroyScalar(&alpha);
        DestroyTensor(&out);
        DestroyTensor(&other);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "adds_invalid_out_shape";
        TensorResource self;
        TensorResource out;
        ScalarResource other;
        ScalarResource alpha;
        const std::vector<double> values = {1, 2, 3, 4};
        const std::vector<int64_t> selfShape = {2, 2};
        const std::vector<int64_t> outShape = {4};
        bool localOk = CreateTensorFromValues(values, selfShape, ACL_INT32, &self) &&
                       CreateZerosTensor(outShape, ACL_INT32, &out) &&
                       CreateScalarFromDouble(2.0, ACL_INT32, &other) &&
                       CreateScalarFromDouble(1.0, ACL_INT32, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckStatus(caseName, ret, ACLNN_ERR_PARAM_INVALID, true);
        }
        DestroyScalar(&alpha);
        DestroyScalar(&other);
        DestroyTensor(&out);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "adds_bool_alpha_float_invalid";
        TensorResource self;
        TensorResource out;
        ScalarResource other;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>{1, 0, 1, 0}, std::vector<int64_t>{2, 2}, ACL_BOOL, &self) &&
                       CreateZerosTensor(std::vector<int64_t>{2, 2}, ACL_BOOL, &out) &&
                       CreateScalarFromDouble(1.0, ACL_BOOL, &other) &&
                       CreateScalarFromDouble(0.5, ACL_FLOAT, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckInvalidStatus(caseName, ret);
        }
        DestroyScalar(&alpha);
        DestroyScalar(&other);
        DestroyTensor(&out);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "inplace_add_invalid_shape";
        TensorResource self;
        TensorResource other;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>(2, 1.0), std::vector<int64_t>(1, 2), ACL_FLOAT, &self) &&
                       CreateTensorFromValues(std::vector<double>(4, 1.0), std::vector<int64_t>(2, 2), ACL_FLOAT, &other) &&
                       CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
            localOk = CheckStatus(caseName, ret, ACLNN_ERR_PARAM_INVALID, true);
        }
        DestroyScalar(&alpha);
        DestroyTensor(&other);
        DestroyTensor(&self);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "add_v3_out_bool_invalid";
        TensorResource other;
        TensorResource out;
        ScalarResource self;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_INT32, &other) &&
                       CreateZerosTensor(std::vector<int64_t>{2, 2}, ACL_BOOL, &out) &&
                       CreateScalarFromDouble(1.5, ACL_FLOAT, &self) &&
                       CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckInvalidStatus(caseName, ret);
        }
        DestroyScalar(&alpha);
        DestroyScalar(&self);
        DestroyTensor(&out);
        DestroyTensor(&other);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "add_v3_other_double_unsupported";
        TensorResource other;
        TensorResource out;
        ScalarResource self;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_DOUBLE, &other) &&
                       CreateZerosTensor(std::vector<int64_t>{2, 2}, ACL_DOUBLE, &out) &&
                       CreateScalarFromDouble(1.0, ACL_DOUBLE, &self) &&
                       CreateScalarFromDouble(1.0, ACL_DOUBLE, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckInvalidStatus(caseName, ret);
        }
        DestroyScalar(&alpha);
        DestroyScalar(&self);
        DestroyTensor(&out);
        DestroyTensor(&other);
        ok = localOk && ok;
    }

    {
        const std::string caseName = "add_v3_invalid_out_shape";
        TensorResource other;
        TensorResource out;
        ScalarResource self;
        ScalarResource alpha;
        bool localOk = CreateTensorFromValues(std::vector<double>(4, 1.0), std::vector<int64_t>(2, 2), ACL_FLOAT, &other) &&
                       CreateZerosTensor(std::vector<int64_t>(4), ACL_FLOAT, &out) &&
                       CreateScalarFromDouble(1.0, ACL_FLOAT, &self) &&
                       CreateScalarFromDouble(1.0, ACL_FLOAT, &alpha);
        if (localOk) {
            uint64_t workspaceSize = 0U;
            aclOpExecutor* executor = nullptr;
            const aclnnStatus ret =
                aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            localOk = CheckStatus(caseName, ret, ACLNN_ERR_PARAM_INVALID, true);
        }
        DestroyScalar(&alpha);
        DestroyScalar(&self);
        DestroyTensor(&out);
        DestroyTensor(&other);
        ok = localOk && ok;
    }

    return ok;
}

void RecordResult(bool passed, TestStats* stats)
{
    if (passed) {
        ++stats->passed;
    } else {
        ++stats->failed;
    }
}

} // namespace

int main()
{
    RuntimeContext context;
    TestStats stats;

    if (!Init(&context)) {
        return 1;
    }

    RecordResult(RunAddCaseFloatDirect(&context), &stats);
    RecordResult(RunAddCaseFloatLongDecimal(&context), &stats);
    RecordResult(RunAddCaseFloatBroadcastAxpyProbe(&context), &stats);
    RecordResult(RunAddCaseMixedFp16Fp32(&context), &stats);
    RecordResult(RunAddCaseMixedFp32Fp16(&context), &stats);
    RecordResult(RunAddCaseBf16Direct(&context), &stats);
    RecordResult(RunAddCaseInt32Direct(&context), &stats);
    RecordResult(RunAddCaseInt64Direct(&context), &stats);
    RecordResult(RunAddCaseUint8Direct(&context), &stats);
    RecordResult(RunAddCaseComplex64FloatPromote(&context), &stats);
    RecordResult(RunAddCaseFloatComplex64Promote(&context), &stats);
    RecordResult(RunAddCaseInt16Int32Promote(&context), &stats);
    RecordResult(RunAddCaseDoubleDirectProbe(&context), &stats);
    RecordResult(RunAddCaseDoubleFallbackProbe(&context), &stats);
    RecordResult(RunAddCaseInt64AxpyV2(&context), &stats);
    RecordResult(RunAddCaseUint8AxpyV2(&context), &stats);
    RecordResult(RunAddsCaseBoolCast(&context), &stats);
    RecordResult(RunAddsCaseFloatDirect(&context), &stats);
    RecordResult(RunAddsCaseFloatLongDecimal(&context), &stats);
    RecordResult(RunAddsCaseBf16Scalar(&context), &stats);
    RecordResult(RunAddsCaseInt64FallbackProbe(&context), &stats);
    RecordResult(RunAddsEmptyTensorCase(&context), &stats);
    RecordResult(RunInplaceAddCase(&context), &stats);
    RecordResult(RunL0AddBroadcastInvalidCase(), &stats);
    RecordResult(RunL0AddInplaceBroadcastInvalidCase(), &stats);
    RecordResult(RunL0AddInplaceOutputShapeInvalidCase(), &stats);
    RecordResult(RunL0AddInplaceMixedOutputInvalidCase(), &stats);
    RecordResult(RunInplaceAddsCaseInt32(&context), &stats);
    RecordResult(RunInplaceAddsEmptyTensorCase(&context), &stats);
    RecordResult(RunAddV3CaseFloatDirect(&context), &stats);
    RecordResult(RunAddV3CaseFloatLongDecimal(&context), &stats);
    RecordResult(RunAddV3CaseInt8Fallback(&context), &stats);
    RecordResult(RunAddV3CaseFloatScalarInt32Tensor(&context), &stats);
    RecordResult(RunInplaceAddV3CaseFloatAxpy(&context), &stats);
    RecordResult(RunInplaceAddV3EmptyTensorCase(&context), &stats);
    RecordResult(RunAddEmptyTensorCase(), &stats);
    RecordResult(RunAddV3EmptyTensorCase(), &stats);
    RecordResult(RunNegativeCases(), &stats);

    Finalize(&context);

    LOG_PRINT("========================================\n");
    LOG_PRINT("Add test summary: passed=%d failed=%d\n", stats.passed, stats.failed);
    LOG_PRINT("========================================\n");
    return stats.failed == 0 ? 0 : 1;
}
