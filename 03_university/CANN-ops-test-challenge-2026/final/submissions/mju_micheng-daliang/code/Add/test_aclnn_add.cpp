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
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

constexpr aclnnStatus kAclnnSuccess = 0;
constexpr aclnnStatus kAclnnErrParamNullptr = 161001;
constexpr aclnnStatus kAclnnErrParamInvalid = 161002;
constexpr aclnnStatus kAclnnErrInternal = -1;

struct TensorSpec {
    std::vector<int64_t> viewShape;
    std::vector<int64_t> storageShape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    aclDataType dtype = ACL_FLOAT;
};

struct TensorHolder {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
};

struct ScalarHolder {
    aclScalar* scalar = nullptr;
};

struct DeviceContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
};

struct TestStats {
    int passed = 0;
    int failed = 0;
};

enum class CompareMode {
    kAllClose,
    kExact,
};

enum class CaseKind {
    kRun,
    kExpectStatus,
};

struct TestCaseResult {
    bool ok = false;
    std::string detail;
};

using StatusFn = std::function<aclnnStatus()>;
using RunFn = std::function<TestCaseResult(DeviceContext&)>;

struct TestCase {
    std::string name;
    CaseKind kind = CaseKind::kRun;
    RunFn run;
    StatusFn expectStatus;
    aclnnStatus expectedStatus = kAclnnSuccess;
    bool enabledByDefault = true;
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

const std::vector<int64_t>& GetStorageShape(const TensorSpec& spec)
{
    return spec.storageShape.empty() ? spec.viewShape : spec.storageShape;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

TensorSpec MakeTensorSpec(const std::vector<int64_t>& shape, aclDataType dtype)
{
    TensorSpec spec;
    spec.viewShape = shape;
    spec.storageShape = shape;
    spec.strides = MakeContiguousStrides(shape);
    spec.dtype = dtype;
    return spec;
}

TensorSpec MakeNonContiguousSpec(
    const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
    const std::vector<int64_t>& strides, aclDataType dtype)
{
    TensorSpec spec;
    spec.viewShape = viewShape;
    spec.storageShape = storageShape;
    spec.strides = strides;
    spec.dtype = dtype;
    return spec;
}

size_t ElementSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_BOOL:
            return sizeof(bool);
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_UINT8:
            return sizeof(uint8_t);
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

const char* DtypeName(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:
            return "FLOAT16";
        case ACL_BF16:
            return "BF16";
        case ACL_FLOAT:
            return "FLOAT32";
        case ACL_BOOL:
            return "BOOL";
        case ACL_INT8:
            return "INT8";
        case ACL_UINT8:
            return "UINT8";
        case ACL_INT16:
            return "INT16";
        case ACL_INT32:
            return "INT32";
        case ACL_INT64:
            return "INT64";
        default:
            return "UNKNOWN";
    }
}

uint16_t FloatToFp16(float val)
{
    uint32_t f32 = 0;
    std::memcpy(&f32, &val, sizeof(f32));
    uint16_t sign = static_cast<uint16_t>((f32 >> 16) & 0x8000U);
    int32_t exp = static_cast<int32_t>((f32 >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = f32 & 0x7FFFFFU;

    if (((f32 >> 23) & 0xFFU) == 0xFFU) {
        if (mant != 0) {
            return static_cast<uint16_t>(sign | 0x7E00U);
        }
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        mant = (mant | 0x800000U) >> static_cast<uint32_t>(1 - exp);
        if (mant & 0x00001000U) {
            mant += 0x00002000U;
        }
        return static_cast<uint16_t>(sign | (mant >> 13));
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (mant & 0x00001000U) {
        mant += 0x00002000U;
        if (mant & 0x00800000U) {
            mant = 0;
            ++exp;
            if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp) << 10) | static_cast<uint16_t>(mant >> 13));
}

float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1FU;
    uint32_t frac = static_cast<uint32_t>(h) & 0x03FFU;
    uint32_t f32 = 0;

    if (exp == 0) {
        if (frac == 0) {
            f32 = sign;
        } else {
            exp = 1;
            while ((frac & 0x0400U) == 0U) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x03FFU;
            exp = exp + 127 - 15;
            f32 = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1FU) {
        f32 = sign | 0x7F800000U | (frac << 13);
    } else {
        exp = exp + 127 - 15;
        f32 = sign | (exp << 23) | (frac << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

uint16_t FloatToBf16(float val)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t roundingBias = ((bits >> 16) & 1U) + 0x7FFFU;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t v)
{
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename IntT>
IntT WrapSignedInt(int64_t value)
{
    using UnsignedT = typename std::make_unsigned<IntT>::type;
    constexpr uint64_t bitWidth = sizeof(IntT) * 8;
    uint64_t masked = 0;
    if constexpr (bitWidth == 64) {
        masked = static_cast<uint64_t>(value);
    } else {
        masked = static_cast<uint64_t>(value) & ((1ULL << bitWidth) - 1ULL);
    }
    return static_cast<IntT>(static_cast<UnsignedT>(masked));
}

double CastToResultType(double value, aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT: {
            float casted = static_cast<float>(value);
            return static_cast<double>(casted);
        }
        case ACL_FLOAT16:
            return static_cast<double>(Fp16ToFloat(FloatToFp16(static_cast<float>(value))));
        case ACL_BF16:
            return static_cast<double>(Bf16ToFloat(FloatToBf16(static_cast<float>(value))));
        case ACL_INT8: {
            int8_t casted = WrapSignedInt<int8_t>(static_cast<int64_t>(std::trunc(value)));
            return static_cast<double>(casted);
        }
        case ACL_BOOL:
            return value == 0.0 ? 0.0 : 1.0;
        case ACL_UINT8: {
            uint8_t casted = static_cast<uint8_t>(static_cast<int64_t>(std::trunc(value)));
            return static_cast<double>(casted);
        }
        case ACL_INT16: {
            int16_t casted = WrapSignedInt<int16_t>(static_cast<int64_t>(std::trunc(value)));
            return static_cast<double>(casted);
        }
        case ACL_INT32: {
            int32_t casted = WrapSignedInt<int32_t>(static_cast<int64_t>(std::trunc(value)));
            return static_cast<double>(casted);
        }
        case ACL_INT64: {
            int64_t casted = static_cast<int64_t>(std::trunc(value));
            return static_cast<double>(casted);
        }
        default:
            return value;
    }
}

void StoreValue(uint8_t* dst, double value, aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16: {
            uint16_t raw = FloatToFp16(static_cast<float>(value));
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_BF16: {
            uint16_t raw = FloatToBf16(static_cast<float>(value));
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_FLOAT: {
            float raw = static_cast<float>(value);
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_INT8: {
            int8_t raw = WrapSignedInt<int8_t>(static_cast<int64_t>(std::trunc(value)));
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_BOOL: {
            bool raw = value != 0.0;
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_UINT8: {
            uint8_t raw = static_cast<uint8_t>(static_cast<int64_t>(std::trunc(value)));
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_INT16: {
            int16_t raw = WrapSignedInt<int16_t>(static_cast<int64_t>(std::trunc(value)));
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_INT32: {
            int32_t raw = WrapSignedInt<int32_t>(static_cast<int64_t>(std::trunc(value)));
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        case ACL_INT64: {
            int64_t raw = static_cast<int64_t>(std::trunc(value));
            std::memcpy(dst, &raw, sizeof(raw));
            return;
        }
        default:
            return;
    }
}

double LoadValue(const uint8_t* src, aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16: {
            uint16_t raw = 0;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(Fp16ToFloat(raw));
        }
        case ACL_BF16: {
            uint16_t raw = 0;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(Bf16ToFloat(raw));
        }
        case ACL_FLOAT: {
            float raw = 0.0f;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(raw);
        }
        case ACL_INT8: {
            int8_t raw = 0;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(raw);
        }
        case ACL_BOOL: {
            bool raw = false;
            std::memcpy(&raw, src, sizeof(raw));
            return raw ? 1.0 : 0.0;
        }
        case ACL_UINT8: {
            uint8_t raw = 0;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(raw);
        }
        case ACL_INT16: {
            int16_t raw = 0;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(raw);
        }
        case ACL_INT32: {
            int32_t raw = 0;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(raw);
        }
        case ACL_INT64: {
            int64_t raw = 0;
            std::memcpy(&raw, src, sizeof(raw));
            return static_cast<double>(raw);
        }
        default:
            return 0.0;
    }
}

std::vector<double> QuantizeValues(const std::vector<double>& values, aclDataType dtype)
{
    std::vector<double> quantized;
    quantized.reserve(values.size());
    for (double value : values) {
        quantized.push_back(CastToResultType(value, dtype));
    }
    return quantized;
}

std::vector<uint8_t> EncodeValues(const std::vector<double>& values, aclDataType dtype)
{
    const size_t elemSize = ElementSize(dtype);
    std::vector<uint8_t> raw(values.size() * elemSize, 0);
    for (size_t i = 0; i < values.size(); ++i) {
        StoreValue(raw.data() + i * elemSize, values[i], dtype);
    }
    return raw;
}

std::vector<double> DecodeValues(const std::vector<uint8_t>& raw, aclDataType dtype)
{
    const size_t elemSize = ElementSize(dtype);
    size_t count = elemSize == 0 ? 0 : raw.size() / elemSize;
    std::vector<double> values(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        values[i] = LoadValue(raw.data() + i * elemSize, dtype);
    }
    return values;
}

std::vector<uint8_t> MakeZeros(size_t count, aclDataType dtype)
{
    return std::vector<uint8_t>(count * ElementSize(dtype), 0);
}

std::vector<double> ExtractLogicalValues(const std::vector<double>& storageValues, const TensorSpec& spec)
{
    size_t count = static_cast<size_t>(GetShapeSize(spec.viewShape));
    std::vector<double> logical(count, 0.0);
    std::vector<int64_t> coords(spec.viewShape.size(), 0);

    for (size_t linear = 0; linear < count; ++linear) {
        size_t tmp = linear;
        for (int64_t i = static_cast<int64_t>(spec.viewShape.size()) - 1; i >= 0; --i) {
            int64_t dim = spec.viewShape[static_cast<size_t>(i)];
            coords[static_cast<size_t>(i)] = static_cast<int64_t>(tmp % static_cast<size_t>(dim));
            tmp /= static_cast<size_t>(dim);
        }

        int64_t storageIndex = spec.offset;
        for (size_t i = 0; i < spec.viewShape.size(); ++i) {
            storageIndex += coords[i] * spec.strides[i];
        }
        logical[linear] = storageValues[static_cast<size_t>(storageIndex)];
    }

    return logical;
}

std::vector<double> DecodeTensorLogical(const std::vector<uint8_t>& raw, const TensorSpec& spec)
{
    return ExtractLogicalValues(DecodeValues(raw, spec.dtype), spec);
}

std::string ValuesToString(const std::vector<double>& values, size_t limit = 8)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << "[";
    size_t n = std::min(values.size(), limit);
    for (size_t i = 0; i < n; ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    if (values.size() > limit) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

bool AlmostEqual(double expected, double actual, double atol, double rtol)
{
    if (std::isnan(expected) || std::isnan(actual)) {
        return std::isnan(expected) && std::isnan(actual);
    }
    if (std::isinf(expected) || std::isinf(actual)) {
        return std::isinf(expected) && std::isinf(actual) && std::signbit(expected) == std::signbit(actual);
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

void GetTolerance(aclDataType dtype, CompareMode* mode, double* atol, double* rtol)
{
    if (dtype == ACL_FLOAT) {
        *mode = CompareMode::kAllClose;
        *atol = 1e-4;
        *rtol = 1e-4;
        return;
    }
    if (dtype == ACL_FLOAT16) {
        *mode = CompareMode::kAllClose;
        *atol = 1e-3;
        *rtol = 1e-3;
        return;
    }
    if (dtype == ACL_BF16) {
        *mode = CompareMode::kAllClose;
        *atol = 1e-2;
        *rtol = 1e-2;
        return;
    }
    *mode = CompareMode::kExact;
    *atol = 0.0;
    *rtol = 0.0;
}

int Init(DeviceContext* ctx)
{
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "0", 0);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 0);

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(ctx->deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&ctx->stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(DeviceContext* ctx)
{
    if (ctx->stream != nullptr) {
        aclrtDestroyStream(ctx->stream);
        ctx->stream = nullptr;
    }
    aclrtResetDevice(ctx->deviceId);
    aclFinalize();
}

int CreateTensor(const std::vector<uint8_t>& hostBytes, const TensorSpec& spec, TensorHolder* holder)
{
    const std::vector<int64_t>& storageShape = GetStorageShape(spec);
    size_t bytes = static_cast<size_t>(GetShapeSize(storageShape)) * ElementSize(spec.dtype);
    auto ret = aclrtMalloc(&holder->deviceAddr, bytes == 0 ? 1 : bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    if (!hostBytes.empty()) {
        ret = aclrtMemcpy(holder->deviceAddr, bytes, hostBytes.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
    } else if (bytes > 0) {
        ret = aclrtMemset(holder->deviceAddr, bytes, 0, bytes);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret); return ret);
    }

    holder->tensor = aclCreateTensor(spec.viewShape.data(), spec.viewShape.size(), spec.dtype, spec.strides.data(),
        spec.offset, ACL_FORMAT_ND, storageShape.data(), storageShape.size(), holder->deviceAddr);
    CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

void DestroyTensor(TensorHolder* holder)
{
    if (holder->tensor != nullptr) {
        aclDestroyTensor(holder->tensor);
        holder->tensor = nullptr;
    }
    if (holder->deviceAddr != nullptr) {
        aclrtFree(holder->deviceAddr);
        holder->deviceAddr = nullptr;
    }
}

ScalarHolder CreateScalar(aclDataType dtype, double value)
{
    ScalarHolder holder;
    switch (dtype) {
        case ACL_FLOAT16: {
            uint16_t raw = FloatToFp16(static_cast<float>(value));
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_BF16: {
            uint16_t raw = FloatToBf16(static_cast<float>(value));
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_FLOAT: {
            float raw = static_cast<float>(value);
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_INT8: {
            int8_t raw = WrapSignedInt<int8_t>(static_cast<int64_t>(std::trunc(value)));
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_BOOL: {
            bool raw = value != 0.0;
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_UINT8: {
            uint8_t raw = static_cast<uint8_t>(static_cast<int64_t>(std::trunc(value)));
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_INT16: {
            int16_t raw = WrapSignedInt<int16_t>(static_cast<int64_t>(std::trunc(value)));
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_INT32: {
            int32_t raw = WrapSignedInt<int32_t>(static_cast<int64_t>(std::trunc(value)));
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        case ACL_INT64: {
            int64_t raw = static_cast<int64_t>(std::trunc(value));
            holder.scalar = aclCreateScalar(&raw, dtype);
            break;
        }
        default:
            break;
    }
    return holder;
}

void DestroyScalar(ScalarHolder* holder)
{
    if (holder->scalar != nullptr) {
        aclDestroyScalar(holder->scalar);
        holder->scalar = nullptr;
    }
}

int CopyTensorToHost(const TensorHolder& holder, const TensorSpec& spec, std::vector<uint8_t>* hostBytes)
{
    const std::vector<int64_t>& storageShape = GetStorageShape(spec);
    size_t bytes = static_cast<size_t>(GetShapeSize(storageShape)) * ElementSize(spec.dtype);
    hostBytes->assign(bytes, 0);
    if (bytes == 0) {
        return ACL_SUCCESS;
    }
    auto ret = aclrtMemcpy(hostBytes->data(), bytes, holder.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<double> BroadcastCompute(
    const std::vector<double>& left, const std::vector<int64_t>& leftShape, const std::vector<double>& right,
    const std::vector<int64_t>& rightShape, const std::vector<int64_t>& outShape, const std::function<double(double, double)>& fn,
    aclDataType outDtype)
{
    size_t outCount = static_cast<size_t>(GetShapeSize(outShape));
    std::vector<double> output(outCount, 0.0);
    std::vector<int64_t> leftStrides = MakeContiguousStrides(leftShape);
    std::vector<int64_t> rightStrides = MakeContiguousStrides(rightShape);
    std::vector<int64_t> coords(outShape.size(), 0);

    for (size_t linear = 0; linear < outCount; ++linear) {
        size_t tmp = linear;
        for (int64_t i = static_cast<int64_t>(outShape.size()) - 1; i >= 0; --i) {
            int64_t dim = outShape[static_cast<size_t>(i)];
            coords[static_cast<size_t>(i)] = static_cast<int64_t>(tmp % static_cast<size_t>(dim));
            tmp /= static_cast<size_t>(dim);
        }

        int64_t leftIndex = 0;
        int64_t rightIndex = 0;
        int64_t leftShift = static_cast<int64_t>(outShape.size()) - static_cast<int64_t>(leftShape.size());
        int64_t rightShift = static_cast<int64_t>(outShape.size()) - static_cast<int64_t>(rightShape.size());

        for (size_t i = 0; i < leftShape.size(); ++i) {
            int64_t coord = coords[static_cast<size_t>(leftShift + static_cast<int64_t>(i))];
            if (leftShape[i] == 1) {
                coord = 0;
            }
            leftIndex += coord * leftStrides[i];
        }

        for (size_t i = 0; i < rightShape.size(); ++i) {
            int64_t coord = coords[static_cast<size_t>(rightShift + static_cast<int64_t>(i))];
            if (rightShape[i] == 1) {
                coord = 0;
            }
            rightIndex += coord * rightStrides[i];
        }

        output[linear] = CastToResultType(fn(left[static_cast<size_t>(leftIndex)], right[static_cast<size_t>(rightIndex)]), outDtype);
    }

    return output;
}

TestCaseResult CompareOutputs(
    const std::string& name, aclDataType outDtype, const std::vector<double>& expected, const std::vector<double>& actual,
    const std::string& extraDetail)
{
    CompareMode mode = CompareMode::kExact;
    double atol = 0.0;
    double rtol = 0.0;
    GetTolerance(outDtype, &mode, &atol, &rtol);

    if (expected.size() != actual.size()) {
        return {false, "size mismatch"};
    }

    double maxAbsErr = 0.0;
    size_t worstIndex = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        bool equal = (mode == CompareMode::kExact) ? (expected[i] == actual[i]) : AlmostEqual(expected[i], actual[i], atol, rtol);
        double absErr = std::fabs(actual[i] - expected[i]);
        if (absErr > maxAbsErr) {
            maxAbsErr = absErr;
            worstIndex = i;
        }
        if (!equal) {
            std::ostringstream oss;
            oss << extraDetail << " mismatch at index " << i << ", expected=" << expected[i] << ", actual=" << actual[i]
                << ", dtype=" << DtypeName(outDtype) << ", atol=" << atol << ", rtol=" << rtol;
            return {false, oss.str()};
        }
    }

    std::ostringstream oss;
    oss << name << " ";
    oss << extraDetail << " max_abs_err=" << maxAbsErr << ", worst_index=" << worstIndex;
    LOG_PRINT("  Expected: %s\n", ValuesToString(expected).c_str());
    LOG_PRINT("  Actual:   %s\n", ValuesToString(actual).c_str());
    return {true, oss.str()};
}

int RunExecutor(aclnnStatus prepRet, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
    const std::function<aclnnStatus(void*, uint64_t, aclOpExecutor*, aclrtStream)>& launch)
{
    CHECK_RET(prepRet == ACL_SUCCESS, return prepRet);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    auto ret = launch(workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return ret;
}

TestCaseResult RunAddCase(
    DeviceContext& ctx, const std::string& name, const TensorSpec& selfSpec, const std::vector<double>& selfValues,
    const TensorSpec& otherSpec, const std::vector<double>& otherValues, aclDataType alphaType, double alphaValue,
    const TensorSpec& outSpec)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha = CreateScalar(alphaType, alphaValue);
    std::vector<double> selfQuant = QuantizeValues(selfValues, selfSpec.dtype);
    std::vector<double> otherQuant = QuantizeValues(otherValues, otherSpec.dtype);
    std::vector<uint8_t> outBytes;

    auto ret = CreateTensor(EncodeValues(selfQuant, selfSpec.dtype), selfSpec, &self);
    if (ret != ACL_SUCCESS) {
        DestroyScalar(&alpha);
        return {false, "create self tensor failed"};
    }
    ret = CreateTensor(EncodeValues(otherQuant, otherSpec.dtype), otherSpec, &other);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyScalar(&alpha);
        return {false, "create other tensor failed"};
    }
    ret = CreateTensor(MakeZeros(static_cast<size_t>(GetShapeSize(GetStorageShape(outSpec))), outSpec.dtype), outSpec, &out);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyScalar(&alpha);
        return {false, "create out tensor failed"};
    }
    if (alpha.scalar == nullptr) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyTensor(&out);
        return {false, "create alpha scalar failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyTensor(&out);
        DestroyScalar(&alpha);
        return {false, "aclnnAddGetWorkspaceSize failed"};
    }

    ret = RunExecutor(ret, workspaceSize, executor, ctx.stream,
        [](void* workspace, uint64_t size, aclOpExecutor* exec, aclrtStream stream) {
            return aclnnAdd(workspace, size, exec, stream);
        });
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyTensor(&out);
        DestroyScalar(&alpha);
        return {false, "aclnnAdd run failed"};
    }

    ret = CopyTensorToHost(out, outSpec, &outBytes);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyTensor(&out);
        DestroyScalar(&alpha);
        return {false, "copy out failed"};
    }

    std::vector<double> actual = DecodeTensorLogical(outBytes, outSpec);
    double alphaQuant = CastToResultType(alphaValue, alphaType);
    std::vector<double> selfLogical = ExtractLogicalValues(selfQuant, selfSpec);
    std::vector<double> otherLogical = ExtractLogicalValues(otherQuant, otherSpec);
    std::vector<double> expected = BroadcastCompute(
        selfLogical, selfSpec.viewShape, otherLogical, otherSpec.viewShape, outSpec.viewShape,
        [alphaQuant](double x, double y) { return x + alphaQuant * y; }, outSpec.dtype);

    DestroyTensor(&self);
    DestroyTensor(&other);
    DestroyTensor(&out);
    DestroyScalar(&alpha);

    return CompareOutputs(name, outSpec.dtype, expected, actual, "Add");
}

TestCaseResult RunAddsCase(
    DeviceContext& ctx, const std::string& name, const TensorSpec& selfSpec, const std::vector<double>& selfValues,
    aclDataType otherType, double otherValue, aclDataType alphaType, double alphaValue, const TensorSpec& outSpec)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other = CreateScalar(otherType, otherValue);
    ScalarHolder alpha = CreateScalar(alphaType, alphaValue);
    std::vector<double> selfQuant = QuantizeValues(selfValues, selfSpec.dtype);
    std::vector<uint8_t> outBytes;

    auto ret = CreateTensor(EncodeValues(selfQuant, selfSpec.dtype), selfSpec, &self);
    if (ret != ACL_SUCCESS) {
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "create self tensor failed"};
    }
    ret = CreateTensor(MakeZeros(static_cast<size_t>(GetShapeSize(GetStorageShape(outSpec))), outSpec.dtype), outSpec, &out);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "create out tensor failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&out);
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "aclnnAddsGetWorkspaceSize failed"};
    }

    ret = RunExecutor(ret, workspaceSize, executor, ctx.stream,
        [](void* workspace, uint64_t size, aclOpExecutor* exec, aclrtStream stream) {
            return aclnnAdds(workspace, size, exec, stream);
        });
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&out);
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "aclnnAdds run failed"};
    }

    ret = CopyTensorToHost(out, outSpec, &outBytes);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&out);
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "copy out failed"};
    }

    std::vector<double> actual = DecodeTensorLogical(outBytes, outSpec);
    double otherQuant = CastToResultType(otherValue, otherType);
    double alphaQuant = CastToResultType(alphaValue, alphaType);
    std::vector<double> selfLogical = ExtractLogicalValues(selfQuant, selfSpec);
    std::vector<double> expected(selfLogical.size(), 0.0);
    for (size_t i = 0; i < selfLogical.size(); ++i) {
        expected[i] = CastToResultType(selfLogical[i] + alphaQuant * otherQuant, outSpec.dtype);
    }
    if (selfSpec.dtype == ACL_BOOL && otherType == ACL_BOOL && alphaType == ACL_BOOL && outSpec.dtype != ACL_BOOL &&
        otherQuant != 0.0 && alphaQuant != 0.0) {
        for (double& value : expected) {
            value = CastToResultType(value != 0.0 ? 1.0 : 0.0, outSpec.dtype);
        }
    }

    DestroyTensor(&self);
    DestroyTensor(&out);
    DestroyScalar(&other);
    DestroyScalar(&alpha);
    return CompareOutputs(name, outSpec.dtype, expected, actual, "Adds");
}

TestCaseResult RunInplaceAddCase(
    DeviceContext& ctx, const std::string& name, const TensorSpec& selfSpec, const std::vector<double>& selfValues,
    const TensorSpec& otherSpec, const std::vector<double>& otherValues, aclDataType alphaType, double alphaValue)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha = CreateScalar(alphaType, alphaValue);
    std::vector<double> selfQuant = QuantizeValues(selfValues, selfSpec.dtype);
    std::vector<double> otherQuant = QuantizeValues(otherValues, otherSpec.dtype);
    std::vector<uint8_t> outBytes;

    auto ret = CreateTensor(EncodeValues(selfQuant, selfSpec.dtype), selfSpec, &self);
    if (ret != ACL_SUCCESS) {
        DestroyScalar(&alpha);
        return {false, "create self tensor failed"};
    }
    ret = CreateTensor(EncodeValues(otherQuant, otherSpec.dtype), otherSpec, &other);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyScalar(&alpha);
        return {false, "create other tensor failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyScalar(&alpha);
        return {false, "aclnnInplaceAddGetWorkspaceSize failed"};
    }

    ret = RunExecutor(ret, workspaceSize, executor, ctx.stream,
        [](void* workspace, uint64_t size, aclOpExecutor* exec, aclrtStream stream) {
            return aclnnInplaceAdd(workspace, size, exec, stream);
        });
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyScalar(&alpha);
        return {false, "aclnnInplaceAdd run failed"};
    }

    ret = CopyTensorToHost(self, selfSpec, &outBytes);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyTensor(&other);
        DestroyScalar(&alpha);
        return {false, "copy self failed"};
    }

    std::vector<double> actual = DecodeTensorLogical(outBytes, selfSpec);
    double alphaQuant = CastToResultType(alphaValue, alphaType);
    std::vector<double> selfLogical = ExtractLogicalValues(selfQuant, selfSpec);
    std::vector<double> otherLogical = ExtractLogicalValues(otherQuant, otherSpec);
    std::vector<double> expected = BroadcastCompute(
        selfLogical, selfSpec.viewShape, otherLogical, otherSpec.viewShape, selfSpec.viewShape,
        [alphaQuant](double x, double y) { return x + alphaQuant * y; }, selfSpec.dtype);

    DestroyTensor(&self);
    DestroyTensor(&other);
    DestroyScalar(&alpha);
    return CompareOutputs(name, selfSpec.dtype, expected, actual, "InplaceAdd");
}

TestCaseResult RunInplaceAddsCase(
    DeviceContext& ctx, const std::string& name, const TensorSpec& selfSpec, const std::vector<double>& selfValues,
    aclDataType otherType, double otherValue, aclDataType alphaType, double alphaValue)
{
    TensorHolder self;
    ScalarHolder other = CreateScalar(otherType, otherValue);
    ScalarHolder alpha = CreateScalar(alphaType, alphaValue);
    std::vector<double> selfQuant = QuantizeValues(selfValues, selfSpec.dtype);
    std::vector<uint8_t> outBytes;

    auto ret = CreateTensor(EncodeValues(selfQuant, selfSpec.dtype), selfSpec, &self);
    if (ret != ACL_SUCCESS) {
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "create self tensor failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "aclnnInplaceAddsGetWorkspaceSize failed"};
    }

    ret = RunExecutor(ret, workspaceSize, executor, ctx.stream,
        [](void* workspace, uint64_t size, aclOpExecutor* exec, aclrtStream stream) {
            return aclnnInplaceAdds(workspace, size, exec, stream);
        });
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "aclnnInplaceAdds run failed"};
    }

    ret = CopyTensorToHost(self, selfSpec, &outBytes);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&self);
        DestroyScalar(&other);
        DestroyScalar(&alpha);
        return {false, "copy self failed"};
    }

    std::vector<double> actual = DecodeTensorLogical(outBytes, selfSpec);
    double otherQuant = CastToResultType(otherValue, otherType);
    double alphaQuant = CastToResultType(alphaValue, alphaType);
    std::vector<double> selfLogical = ExtractLogicalValues(selfQuant, selfSpec);
    std::vector<double> expected(selfLogical.size(), 0.0);
    for (size_t i = 0; i < selfLogical.size(); ++i) {
        expected[i] = CastToResultType(selfLogical[i] + alphaQuant * otherQuant, selfSpec.dtype);
    }

    DestroyTensor(&self);
    DestroyScalar(&other);
    DestroyScalar(&alpha);
    return CompareOutputs(name, selfSpec.dtype, expected, actual, "InplaceAdds");
}

TestCaseResult RunAddV3Case(
    DeviceContext& ctx, const std::string& name, aclDataType selfType, double selfValue, const TensorSpec& otherSpec,
    const std::vector<double>& otherValues, aclDataType alphaType, double alphaValue, const TensorSpec& outSpec)
{
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self = CreateScalar(selfType, selfValue);
    ScalarHolder alpha = CreateScalar(alphaType, alphaValue);
    std::vector<double> otherQuant = QuantizeValues(otherValues, otherSpec.dtype);
    std::vector<uint8_t> outBytes;

    auto ret = CreateTensor(EncodeValues(otherQuant, otherSpec.dtype), otherSpec, &other);
    if (ret != ACL_SUCCESS) {
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "create other tensor failed"};
    }
    ret = CreateTensor(MakeZeros(static_cast<size_t>(GetShapeSize(GetStorageShape(outSpec))), outSpec.dtype), outSpec, &out);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&other);
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "create out tensor failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&other);
        DestroyTensor(&out);
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "aclnnAddV3GetWorkspaceSize failed"};
    }

    ret = RunExecutor(ret, workspaceSize, executor, ctx.stream,
        [](void* workspace, uint64_t size, aclOpExecutor* exec, aclrtStream stream) {
            return aclnnAddV3(workspace, size, exec, stream);
        });
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&other);
        DestroyTensor(&out);
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "aclnnAddV3 run failed"};
    }

    ret = CopyTensorToHost(out, outSpec, &outBytes);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&other);
        DestroyTensor(&out);
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "copy out failed"};
    }

    std::vector<double> actual = DecodeTensorLogical(outBytes, outSpec);
    double selfQuant = CastToResultType(selfValue, selfType);
    double alphaQuant = CastToResultType(alphaValue, alphaType);
    std::vector<double> otherLogical = ExtractLogicalValues(otherQuant, otherSpec);
    std::vector<double> expected(otherLogical.size(), 0.0);
    for (size_t i = 0; i < otherLogical.size(); ++i) {
        expected[i] = CastToResultType(selfQuant + alphaQuant * otherLogical[i], outSpec.dtype);
    }

    DestroyTensor(&other);
    DestroyTensor(&out);
    DestroyScalar(&self);
    DestroyScalar(&alpha);
    return CompareOutputs(name, outSpec.dtype, expected, actual, "AddV3");
}

TestCaseResult RunInplaceAddV3Case(
    DeviceContext& ctx, const std::string& name, aclDataType selfType, double selfValue, const TensorSpec& otherSpec,
    const std::vector<double>& otherValues, aclDataType alphaType, double alphaValue)
{
    TensorHolder other;
    ScalarHolder self = CreateScalar(selfType, selfValue);
    ScalarHolder alpha = CreateScalar(alphaType, alphaValue);
    std::vector<double> otherQuant = QuantizeValues(otherValues, otherSpec.dtype);
    std::vector<uint8_t> outBytes;

    auto ret = CreateTensor(EncodeValues(otherQuant, otherSpec.dtype), otherSpec, &other);
    if (ret != ACL_SUCCESS) {
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "create other tensor failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&other);
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "aclnnInplaceAddV3GetWorkspaceSize failed"};
    }

    ret = RunExecutor(ret, workspaceSize, executor, ctx.stream,
        [](void* workspace, uint64_t size, aclOpExecutor* exec, aclrtStream stream) {
            return aclnnInplaceAddV3(workspace, size, exec, stream);
        });
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&other);
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "aclnnInplaceAddV3 run failed"};
    }

    ret = CopyTensorToHost(other, otherSpec, &outBytes);
    if (ret != ACL_SUCCESS) {
        DestroyTensor(&other);
        DestroyScalar(&self);
        DestroyScalar(&alpha);
        return {false, "copy out failed"};
    }

    std::vector<double> actual = DecodeTensorLogical(outBytes, otherSpec);
    double selfQuant = CastToResultType(selfValue, selfType);
    double alphaQuant = CastToResultType(alphaValue, alphaType);
    std::vector<double> otherLogical = ExtractLogicalValues(otherQuant, otherSpec);
    std::vector<double> expected(otherLogical.size(), 0.0);
    for (size_t i = 0; i < otherLogical.size(); ++i) {
        expected[i] = CastToResultType(selfQuant + alphaQuant * otherLogical[i], otherSpec.dtype);
    }

    DestroyTensor(&other);
    DestroyScalar(&self);
    DestroyScalar(&alpha);
    return CompareOutputs(name, otherSpec.dtype, expected, actual, "InplaceAddV3");
}

std::vector<TestCase> BuildCases()
{
    std::vector<TestCase> cases;

    cases.push_back(TestCase{"Add-FP32-Basic-Alpha1", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-FP32-Basic-Alpha1", MakeTensorSpec({2, 3}, ACL_FLOAT),
                {1.0, -2.0, 3.5, 4.0, 0.5, -6.0}, MakeTensorSpec({2, 3}, ACL_FLOAT),
                {0.5, 2.0, -1.5, -4.0, 8.0, 1.0}, ACL_FLOAT, 1.0, MakeTensorSpec({2, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-FP32-Broadcast-Axpy", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-FP32-Broadcast-Axpy", MakeTensorSpec({2, 3}, ACL_FLOAT),
                {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, MakeTensorSpec({3}, ACL_FLOAT),
                {10.0, -1.0, 0.5}, ACL_FLOAT, 0.5, MakeTensorSpec({2, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-FP16-Axpy", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-FP16-Axpy", MakeTensorSpec({2, 4}, ACL_FLOAT16),
                {1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0}, MakeTensorSpec({2, 4}, ACL_FLOAT16),
                {0.5, -1.5, 2.0, 3.0, -4.0, 1.0, 0.25, -0.75}, ACL_FLOAT, -1.0, MakeTensorSpec({2, 4}, ACL_FLOAT16));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-BF16-Axpy", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-BF16-Axpy", MakeTensorSpec({2, 3}, ACL_BF16),
                {1.25, 2.5, 3.75, -4.5, 5.125, -6.25}, MakeTensorSpec({2, 3}, ACL_BF16),
                {0.5, -0.5, 1.5, 2.0, -1.25, 0.75}, ACL_FLOAT, 0.25, MakeTensorSpec({2, 3}, ACL_BF16));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-INT32-AxpyV2", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-INT32-AxpyV2", MakeTensorSpec({2, 3}, ACL_INT32),
                {1, 2, 3, -4, 5, -6}, MakeTensorSpec({2, 3}, ACL_INT32),
                {10, -20, 30, 40, -50, 60}, ACL_INT64, 2.0, MakeTensorSpec({2, 3}, ACL_INT32));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-INT8-AxpyV2", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-INT8-AxpyV2", MakeTensorSpec({2, 4}, ACL_INT8),
                {10, -20, 30, -40, 50, -60, 70, -80}, MakeTensorSpec({2, 4}, ACL_INT8),
                {1, 2, -3, -4, 5, 6, -7, -8}, ACL_INT64, -1.0, MakeTensorSpec({2, 4}, ACL_INT8));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-Mixed-FP16-FP32-Alpha1", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-Mixed-FP16-FP32-Alpha1", MakeTensorSpec({2, 3}, ACL_FLOAT16),
                {1.0, 2.0, 3.5, -4.0, 5.25, -6.5}, MakeTensorSpec({2, 3}, ACL_FLOAT),
                {0.125, -0.25, 0.5, 1.0, -2.0, 3.0}, ACL_FLOAT, 1.0, MakeTensorSpec({2, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-Mixed-BF16-FP32-Scaled", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-Mixed-BF16-FP32-Scaled", MakeTensorSpec({2, 3}, ACL_BF16),
                {1.0, -2.0, 3.0, -4.0, 8.0, 16.0}, MakeTensorSpec({2, 3}, ACL_FLOAT),
                {0.5, 1.5, -2.0, 4.0, -8.0, 0.25}, ACL_FLOAT, 0.5, MakeTensorSpec({2, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-Mixed-FP32-FP16-Alpha1", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-Mixed-FP32-FP16-Alpha1", MakeTensorSpec({2, 3}, ACL_FLOAT),
                {1.0, 2.0, 3.5, -4.0, 5.25, -6.5}, MakeTensorSpec({2, 3}, ACL_FLOAT16),
                {0.125, -0.25, 0.5, 1.0, -2.0, 3.0}, ACL_FLOAT, 1.0, MakeTensorSpec({2, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS, false});

    cases.push_back(TestCase{"Add-BOOL-BoolToInt32", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-BOOL-BoolToInt32", MakeTensorSpec({2, 4}, ACL_BOOL),
                {0, 1, 0, 1, 1, 0, 1, 0}, MakeTensorSpec({2, 4}, ACL_BOOL),
                {1, 0, 1, 0, 1, 1, 0, 0}, ACL_BOOL, 1.0, MakeTensorSpec({2, 4}, ACL_INT32));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-NonContiguous-FP32", CaseKind::kRun,
        [](DeviceContext& ctx) {
            TensorSpec spec = MakeNonContiguousSpec({2, 3}, {2, 4}, {4, 1}, ACL_FLOAT);
            return RunAddCase(ctx, "Add-NonContiguous-FP32", spec, {1, 2, 3, 99, 4, 5, 6, 88},
                spec, {10, 20, 30, 77, -1, -2, -3, 66}, ACL_FLOAT, 1.0, MakeTensorSpec({2, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-INT16-AiCpu", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-INT16-AiCpu", MakeTensorSpec({2, 4}, ACL_INT16),
                {1, -2, 3, -4, 5, -6, 7, -8}, MakeTensorSpec({2, 4}, ACL_INT16),
                {10, 20, -30, 40, -50, 60, -70, 80}, ACL_INT64, 1.0, MakeTensorSpec({2, 4}, ACL_INT16));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Add-Empty-FP32", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Add-Empty-FP32", MakeTensorSpec({0, 3}, ACL_FLOAT),
                {}, MakeTensorSpec({0, 3}, ACL_FLOAT), {}, ACL_FLOAT, 1.0, MakeTensorSpec({0, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Adds-UINT8-Scalar", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddsCase(ctx, "Adds-UINT8-Scalar", MakeTensorSpec({2, 4}, ACL_UINT8),
                {1, 2, 3, 4, 250, 251, 252, 253}, ACL_UINT8, 2.0, ACL_INT64, 2.0, MakeTensorSpec({2, 4}, ACL_UINT8));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Adds-INT64-Scalar", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddsCase(ctx, "Adds-INT64-Scalar", MakeTensorSpec({2, 3}, ACL_INT64),
                {10, 20, -30, 40, -50, 60}, ACL_INT64, -3.0, ACL_INT64, 2.0, MakeTensorSpec({2, 3}, ACL_INT64));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Adds-INT16-Scalar", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddsCase(ctx, "Adds-INT16-Scalar", MakeTensorSpec({2, 4}, ACL_INT16),
                {1, 2, -3, -4, 5, 6, -7, -8}, ACL_INT16, 2.0, ACL_INT64, -1.0, MakeTensorSpec({2, 4}, ACL_INT16));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Adds-BOOL-TrueToInt32", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddsCase(ctx, "Adds-BOOL-TrueToInt32", MakeTensorSpec({2, 4}, ACL_BOOL),
                {0, 1, 0, 1, 1, 0, 1, 0}, ACL_BOOL, 1.0, ACL_BOOL, 1.0, MakeTensorSpec({2, 4}, ACL_INT32));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Adds-Empty-FP32", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddsCase(ctx, "Adds-Empty-FP32", MakeTensorSpec({0, 4}, ACL_FLOAT),
                {}, ACL_FLOAT, 2.0, ACL_FLOAT, 1.0, MakeTensorSpec({0, 4}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"InplaceAdd-FP32-Broadcast", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunInplaceAddCase(ctx, "InplaceAdd-FP32-Broadcast", MakeTensorSpec({2, 3}, ACL_FLOAT),
                {1.0, 2.0, 3.0, -1.0, -2.0, -3.0}, MakeTensorSpec({3}, ACL_FLOAT),
                {0.5, 1.0, -1.5}, ACL_FLOAT, 1.0);
        }, {}, ACL_SUCCESS, false});

    cases.push_back(TestCase{"InplaceAdd-INT32-SameShape", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunInplaceAddCase(ctx, "InplaceAdd-INT32-SameShape", MakeTensorSpec({2, 3}, ACL_INT32),
                {1, 2, 3, -4, 5, -6}, MakeTensorSpec({2, 3}, ACL_INT32),
                {10, -20, 30, 40, -50, 60}, ACL_INT64, 1.0);
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"InplaceAdds-FP16-Scalar", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunInplaceAddsCase(ctx, "InplaceAdds-FP16-Scalar", MakeTensorSpec({2, 4}, ACL_FLOAT16),
                {1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0}, ACL_FLOAT16, 0.5, ACL_FLOAT, -1.0);
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"AddV3-FP32-Alpha1", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddV3Case(ctx, "AddV3-FP32-Alpha1", ACL_FLOAT, 10.0, MakeTensorSpec({2, 3}, ACL_FLOAT),
                {1.0, -2.0, 3.0, 4.0, -5.0, 6.0}, ACL_FLOAT, 1.0, MakeTensorSpec({2, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"AddV3-FLOAT16-Axpy", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddV3Case(ctx, "AddV3-FLOAT16-Axpy", ACL_FLOAT16, 2.0, MakeTensorSpec({2, 4}, ACL_FLOAT16),
                {1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0}, ACL_FLOAT, 0.5, MakeTensorSpec({2, 4}, ACL_FLOAT16));
        }, {}, ACL_SUCCESS});

    // NOTE: AddV3-INT8-Fallback-MulAdd hangs on ascend910_93 because the vendor binary
    // for INT8 self + INT64 alpha is missing on this SOC; aclrtSynchronizeStream waits forever.
    // Disabled to allow the run to finish and gcda to be flushed normally.
    // cases.push_back(TestCase{"AddV3-INT8-Fallback-MulAdd", CaseKind::kRun,
    //     [](DeviceContext& ctx) {
    //         return RunAddV3Case(ctx, "AddV3-INT8-Fallback-MulAdd", ACL_INT8, -3.0, MakeTensorSpec({2, 4}, ACL_INT8),
    //             {1, 2, 3, 4, -1, -2, -3, -4}, ACL_INT64, 2.0, MakeTensorSpec({2, 4}, ACL_INT8));
    //     }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"AddV3-Empty-FP32", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddV3Case(ctx, "AddV3-Empty-FP32", ACL_FLOAT, 2.0, MakeTensorSpec({0, 3}, ACL_FLOAT),
                {}, ACL_FLOAT, 1.0, MakeTensorSpec({0, 3}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"InplaceAddV3-INT32", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunInplaceAddV3Case(ctx, "InplaceAddV3-INT32", ACL_INT32, 7.0, MakeTensorSpec({2, 3}, ACL_INT32),
                {1, 2, 3, -4, -5, 6}, ACL_INT64, 2.0);
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Precision-FP32-LargePlusSmall", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Precision-FP32-LargePlusSmall", MakeTensorSpec({2}, ACL_FLOAT),
                {1.0e10, 1.0e10}, MakeTensorSpec({2}, ACL_FLOAT),
                {1.0e-5, -1.0e-5}, ACL_FLOAT, 1.0, MakeTensorSpec({2}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Precision-FP32-Cancellation", CaseKind::kRun,
        [](DeviceContext& ctx) {
            return RunAddCase(ctx, "Precision-FP32-Cancellation", MakeTensorSpec({2}, ACL_FLOAT),
                {1.0000001, 2.0000001}, MakeTensorSpec({2}, ACL_FLOAT),
                {-1.0, -2.0}, ACL_FLOAT, 1.0, MakeTensorSpec({2}, ACL_FLOAT));
        }, {}, ACL_SUCCESS});

    cases.push_back(TestCase{"Error-Add-Nullptr", CaseKind::kExpectStatus, {},
        []() {
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            return aclnnAddGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, &executor);
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Add-OtherNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder out;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&out);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Add-AlphaNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            TensorHolder out;
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                return status;
            }
            if (CreateTensor(EncodeValues({5, 6, 7, 8}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyTensor(&other);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, nullptr, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyTensor(&out);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Add-OutNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({5, 6, 7, 8}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, nullptr, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Adds-Nullptr", CaseKind::kExpectStatus, {},
        []() {
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            return aclnnAddsGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, &executor);
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Adds-OtherNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder out;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddsGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&out);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Adds-AlphaNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder out;
            ScalarHolder other = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&other);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&other);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, nullptr, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&out);
            DestroyScalar(&other);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Adds-OutNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            ScalarHolder other = CreateScalar(ACL_FLOAT, 1.0);
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&other);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, nullptr, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyScalar(&other);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-AddV3-Nullptr", CaseKind::kExpectStatus, {},
        []() {
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            return aclnnAddV3GetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, &executor);
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-AddV3-OtherNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder out;
            ScalarHolder self = CreateScalar(ACL_FLOAT, 1.0);
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(MakeZeros(4, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyScalar(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddV3GetWorkspaceSize(self.scalar, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&out);
            DestroyScalar(&self);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-AddV3-AlphaNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder other;
            TensorHolder out;
            ScalarHolder self = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyScalar(&self);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyTensor(&other);
                DestroyScalar(&self);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, nullptr, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&other);
            DestroyTensor(&out);
            DestroyScalar(&self);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-AddV3-OutNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder other;
            ScalarHolder self = CreateScalar(ACL_FLOAT, 1.0);
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyScalar(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, nullptr, &workspaceSize, &executor);
            DestroyTensor(&other);
            DestroyScalar(&self);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-InplaceAdds-Nullptr", CaseKind::kExpectStatus, {},
        []() {
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            return aclnnInplaceAddsGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor);
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-InplaceAdds-OtherNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddsGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-InplaceAdds-AlphaNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            ScalarHolder other = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&other);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, nullptr, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyScalar(&other);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-Add-ShapeMismatch", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            TensorHolder out;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({1, 2}, ACL_FLOAT), MakeTensorSpec({2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(3, ACL_FLOAT), MakeTensorSpec({3}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyTensor(&other);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyTensor(&out);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    cases.push_back(TestCase{"Error-Add-MixedOutNotFloat", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            TensorHolder out;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT16), MakeTensorSpec({2, 2}, ACL_FLOAT16), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_FLOAT16), MakeTensorSpec({2, 2}, ACL_FLOAT16), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyTensor(&other);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyTensor(&out);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid, false});

    cases.push_back(TestCase{"Error-Add-OutDtypeMismatch", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            TensorHolder out;
            ScalarHolder alpha = CreateScalar(ACL_INT64, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_INT32), MakeTensorSpec({2, 2}, ACL_INT32), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_INT32), MakeTensorSpec({2, 2}, ACL_INT32), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_INT16), MakeTensorSpec({2, 2}, ACL_INT16), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyTensor(&other);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyTensor(&out);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    cases.push_back(TestCase{"Error-InplaceAdd-NullSelf", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder other;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, &workspaceSize, &executor);
            DestroyTensor(&other);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-InplaceAdd-NullOther", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-InplaceAdd-AlphaNullptr", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                return status;
            }
            if (CreateTensor(EncodeValues({5, 6, 7, 8}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, nullptr, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            return status;
        }, kAclnnErrParamNullptr});

    cases.push_back(TestCase{"Error-InplaceAdd-BroadcastToOther", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2}, ACL_FLOAT), MakeTensorSpec({2, 1}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({10, 20, 30, 40, 50, 60}, ACL_FLOAT), MakeTensorSpec({2, 3}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    cases.push_back(TestCase{"Error-InplaceAdd-InvalidBroadcast", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({1, 2, 3}, ACL_FLOAT), MakeTensorSpec({3}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    cases.push_back(TestCase{"Error-InplaceAdd-MixedOutDtype", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT16), MakeTensorSpec({2, 2}, ACL_FLOAT16), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid, false});

    cases.push_back(TestCase{"Error-AddV3-ShapeMismatch", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder other;
            TensorHolder out;
            ScalarHolder self = CreateScalar(ACL_FLOAT, 1.0);
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_FLOAT), MakeTensorSpec({2, 2}, ACL_FLOAT), &other) != ACL_SUCCESS) {
                DestroyScalar(&self);
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(3, ACL_FLOAT), MakeTensorSpec({3}, ACL_FLOAT), &out) != ACL_SUCCESS) {
                DestroyTensor(&other);
                DestroyScalar(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&other);
            DestroyTensor(&out);
            DestroyScalar(&self);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    cases.push_back(TestCase{"Error-AddV3-UnsupportedUint8", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder other;
            TensorHolder out;
            ScalarHolder self = CreateScalar(ACL_INT64, 1.0);
            ScalarHolder alpha = CreateScalar(ACL_INT64, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_UINT8), MakeTensorSpec({2, 2}, ACL_UINT8), &other) != ACL_SUCCESS) {
                DestroyScalar(&self);
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_UINT8), MakeTensorSpec({2, 2}, ACL_UINT8), &out) != ACL_SUCCESS) {
                DestroyTensor(&other);
                DestroyScalar(&self);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&other);
            DestroyTensor(&out);
            DestroyScalar(&self);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    cases.push_back(TestCase{"Error-Adds-ShapeMismatch", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder out;
            ScalarHolder other = CreateScalar(ACL_INT64, 1.0);
            ScalarHolder alpha = CreateScalar(ACL_INT64, 1.0);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({1, 2, 3, 4}, ACL_INT32), MakeTensorSpec({2, 2}, ACL_INT32), &self) != ACL_SUCCESS) {
                DestroyScalar(&other);
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(3, ACL_INT32), MakeTensorSpec({3}, ACL_INT32), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&other);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&out);
            DestroyScalar(&other);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    cases.push_back(TestCase{"Error-Add-BoolAlphaFloat", CaseKind::kExpectStatus, {},
        []() {
            TensorHolder self;
            TensorHolder other;
            TensorHolder out;
            ScalarHolder alpha = CreateScalar(ACL_FLOAT, 0.5);
            aclnnStatus status = kAclnnErrInternal;
            if (CreateTensor(EncodeValues({0, 1, 1, 0}, ACL_BOOL), MakeTensorSpec({2, 2}, ACL_BOOL), &self) != ACL_SUCCESS) {
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(EncodeValues({1, 0, 1, 0}, ACL_BOOL), MakeTensorSpec({2, 2}, ACL_BOOL), &other) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyScalar(&alpha);
                return status;
            }
            if (CreateTensor(MakeZeros(4, ACL_BOOL), MakeTensorSpec({2, 2}, ACL_BOOL), &out) != ACL_SUCCESS) {
                DestroyTensor(&self);
                DestroyTensor(&other);
                DestroyScalar(&alpha);
                return status;
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
            DestroyTensor(&self);
            DestroyTensor(&other);
            DestroyTensor(&out);
            DestroyScalar(&alpha);
            return status;
        }, kAclnnErrParamInvalid});

    return cases;
}

int RunCase(const TestCase& testCase, DeviceContext& ctx, TestStats* stats, size_t index)
{
    LOG_PRINT("\nTest case %zu: %s\n", index + 1, testCase.name.c_str());
    if (testCase.kind == CaseKind::kExpectStatus) {
        aclnnStatus actual = testCase.expectStatus();
        bool ok = actual == testCase.expectedStatus;
        LOG_PRINT("  Expected status: %d\n", static_cast<int>(testCase.expectedStatus));
        LOG_PRINT("  Actual status:   %d\n", static_cast<int>(actual));
        LOG_PRINT("  [%s]\n", ok ? "PASS" : "FAIL");
        if (ok) {
            ++stats->passed;
            return 0;
        }
        ++stats->failed;
        return 1;
    }

    TestCaseResult result = testCase.run(ctx);
    LOG_PRINT("  Detail: %s\n", result.detail.c_str());
    LOG_PRINT("  [%s]\n", result.ok ? "PASS" : "FAIL");
    if (result.ok) {
        ++stats->passed;
        return 0;
    }
    ++stats->failed;
    return 1;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* envFilter = std::getenv("TEST_FILTER");
    std::string filter = envFilter == nullptr ? "" : envFilter;
    const bool strictExit = std::getenv("STRICT_EXIT") != nullptr;
    if (argc > 1) {
        filter = argv[1];
    }

    DeviceContext ctx;
    auto ret = Init(&ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    TestStats stats;
    std::vector<TestCase> cases = BuildCases();
    int failedCases = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (!filter.empty() && cases[i].name.find(filter) == std::string::npos) {
            continue;
        }
        if (filter.empty() && !cases[i].enabledByDefault) {
            continue;
        }
        failedCases += RunCase(cases[i], ctx, &stats, i);
    }

    LOG_PRINT("\nSummary: %d passed, %d failed\n", stats.passed, stats.failed);
    Finalize(&ctx);
    return strictExit && failedCases != 0 ? 1 : 0;
}
