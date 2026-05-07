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
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_cumsum.h"

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

struct TensorResource {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
};

struct CaseStats {
    int passed = 0;
    int failed = 0;
};

enum class ApiKind {
    Cumsum,
    CumsumV2,
};

struct ValidCase {
    std::string name;
    ApiKind api = ApiKind::Cumsum;
    aclDataType dtype = ACL_FLOAT;
    std::vector<int64_t> shape;
    int64_t dim = 0;
    bool exclusive = false;
    bool reverse = false;
    std::vector<double> input;
    double atol = 1e-5;
    double rtol = 1e-5;
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

size_t GetDataTypeSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_BOOL:
        case ACL_INT8:
        case ACL_UINT8:
            return 1;
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
            return 4;
        case ACL_DOUBLE:
        case ACL_INT64:
            return 8;
        default:
            return 0;
    }
}

const char *DtypeName(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return "float32";
        case ACL_DOUBLE:
            return "double";
        case ACL_FLOAT16:
            return "float16";
        case ACL_BF16:
            return "bfloat16";
        case ACL_INT32:
            return "int32";
        case ACL_INT16:
            return "int16";
        case ACL_INT64:
            return "int64";
        case ACL_INT8:
            return "int8";
        case ACL_UINT8:
            return "uint8";
        case ACL_BOOL:
            return "bool";
        default:
            return "unknown";
    }
}

bool IsFloatingType(aclDataType dtype)
{
    return dtype == ACL_FLOAT || dtype == ACL_DOUBLE || dtype == ACL_FLOAT16 || dtype == ACL_BF16;
}

uint32_t FloatBits(float value)
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

uint16_t FloatToFloat16(float value)
{
    uint32_t bits = FloatBits(value);
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;

    if (std::isnan(value)) {
        return static_cast<uint16_t>(sign | 0x7e00);
    }
    if (std::isinf(value)) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
}

float Float16ToFloat(uint16_t value)
{
    uint32_t sign = (static_cast<uint32_t>(value & 0x8000)) << 16;
    uint32_t exponent = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x03ff;

    if (exponent == 0) {
        if (mantissa == 0) {
            return BitsToFloat(sign);
        }
        float result = static_cast<float>(mantissa) * std::pow(2.0f, -24.0f);
        return (sign != 0) ? -result : result;
    }
    if (exponent == 31) {
        return mantissa == 0 ? ((sign != 0) ? -std::numeric_limits<float>::infinity()
                                            : std::numeric_limits<float>::infinity())
                             : std::numeric_limits<float>::quiet_NaN();
    }

    uint32_t floatExponent = exponent - 15 + 127;
    uint32_t bits = sign | (floatExponent << 23) | (mantissa << 13);
    return BitsToFloat(bits);
}

uint16_t FloatToBFloat16(float value)
{
    return static_cast<uint16_t>(FloatBits(value) >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    return BitsToFloat(static_cast<uint32_t>(value) << 16);
}

void AppendBytes(std::vector<uint8_t> *buffer, const void *data, size_t size)
{
    const auto *ptr = reinterpret_cast<const uint8_t *>(data);
    buffer->insert(buffer->end(), ptr, ptr + size);
}

void AppendEncodedValue(std::vector<uint8_t> *buffer, aclDataType dtype, double value)
{
    switch (dtype) {
        case ACL_FLOAT: {
            float converted = static_cast<float>(value);
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_DOUBLE: {
            double converted = value;
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_FLOAT16: {
            uint16_t converted = FloatToFloat16(static_cast<float>(value));
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_BF16: {
            uint16_t converted = FloatToBFloat16(static_cast<float>(value));
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_INT32: {
            int32_t converted = static_cast<int32_t>(std::llround(value));
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_INT16: {
            int16_t converted = static_cast<int16_t>(std::llround(value));
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_INT64: {
            int64_t converted = static_cast<int64_t>(std::llround(value));
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_INT8: {
            int8_t converted = static_cast<int8_t>(std::llround(value));
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_UINT8: {
            uint8_t converted = static_cast<uint8_t>(std::llround(value));
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        case ACL_BOOL: {
            uint8_t converted = value != 0.0 ? 1 : 0;
            AppendBytes(buffer, &converted, sizeof(converted));
            return;
        }
        default:
            return;
    }
}

std::vector<uint8_t> EncodeValues(const std::vector<double> &values, aclDataType dtype)
{
    std::vector<uint8_t> buffer;
    buffer.reserve(values.size() * GetDataTypeSize(dtype));
    for (double value : values) {
        AppendEncodedValue(&buffer, dtype, value);
    }
    return buffer;
}

double DecodeValue(const uint8_t *ptr, aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT: {
            float value = 0.0f;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(value);
        }
        case ACL_DOUBLE: {
            double value = 0.0;
            std::memcpy(&value, ptr, sizeof(value));
            return value;
        }
        case ACL_FLOAT16: {
            uint16_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(Float16ToFloat(value));
        }
        case ACL_BF16: {
            uint16_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(BFloat16ToFloat(value));
        }
        case ACL_INT32: {
            int32_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(value);
        }
        case ACL_INT16: {
            int16_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(value);
        }
        case ACL_INT64: {
            int64_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(value);
        }
        case ACL_INT8: {
            int8_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(value);
        }
        case ACL_UINT8: {
            uint8_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(value);
        }
        case ACL_BOOL: {
            uint8_t value = 0;
            std::memcpy(&value, ptr, sizeof(value));
            return static_cast<double>(value != 0);
        }
        default:
            return 0.0;
    }
}

std::vector<double> DecodeValues(const std::vector<uint8_t> &bytes, aclDataType dtype)
{
    std::vector<double> values;
    size_t typeSize = GetDataTypeSize(dtype);
    if (typeSize == 0) {
        return values;
    }
    values.reserve(bytes.size() / typeSize);
    for (size_t offset = 0; offset + typeSize <= bytes.size(); offset += typeSize) {
        values.push_back(DecodeValue(bytes.data() + offset, dtype));
    }
    return values;
}

std::vector<double> QuantizeInputForReference(const std::vector<double> &values, aclDataType dtype)
{
    return DecodeValues(EncodeValues(values, dtype), dtype);
}

std::vector<double> CpuCumsum(const std::vector<double> &input, const std::vector<int64_t> &shape, int64_t dim,
                             bool exclusive, bool reverse)
{
    std::vector<double> result(input.size(), 0.0);
    int64_t rank = static_cast<int64_t>(shape.size());
    if (rank == 0) {
        if (!input.empty()) {
            result[0] = exclusive ? 0.0 : input[0];
        }
        return result;
    }
    int64_t axis = dim < 0 ? dim + rank : dim;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[static_cast<size_t>(i)];
    }
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= shape[static_cast<size_t>(i)];
    }
    int64_t axisLen = shape[static_cast<size_t>(axis)];
    int64_t axisBlock = axisLen * inner;

    for (int64_t outIdx = 0; outIdx < outer; ++outIdx) {
        for (int64_t inIdx = 0; inIdx < inner; ++inIdx) {
            double sum = 0.0;
            if (!reverse) {
                for (int64_t a = 0; a < axisLen; ++a) {
                    int64_t offset = outIdx * axisBlock + a * inner + inIdx;
                    if (exclusive) {
                        result[static_cast<size_t>(offset)] = sum;
                        sum += input[static_cast<size_t>(offset)];
                    } else {
                        sum += input[static_cast<size_t>(offset)];
                        result[static_cast<size_t>(offset)] = sum;
                    }
                }
            } else {
                for (int64_t a = axisLen - 1; a >= 0; --a) {
                    int64_t offset = outIdx * axisBlock + a * inner + inIdx;
                    if (exclusive) {
                        result[static_cast<size_t>(offset)] = sum;
                        sum += input[static_cast<size_t>(offset)];
                    } else {
                        sum += input[static_cast<size_t>(offset)];
                        result[static_cast<size_t>(offset)] = sum;
                    }
                }
            }
        }
    }
    return result;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return ACL_SUCCESS;
}

void DestroyTensorResource(TensorResource *resource)
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

int CreateAclTensor(const std::vector<uint8_t> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                    TensorResource *resource)
{
    size_t size = hostData.size();
    auto ret = ACL_SUCCESS;
    if (size > 0) {
        ret = aclrtMalloc(&resource->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret;
        }
        ret = aclrtMemcpy(resource->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    auto strides = MakeStrides(shape);
    resource->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), resource->deviceAddr);
    if (resource->tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        return 1;
    }
    return ACL_SUCCESS;
}

int CopyDeviceToHost(const TensorResource &resource, size_t size, std::vector<uint8_t> *hostData)
{
    hostData->assign(size, 0);
    if (size == 0) {
        return ACL_SUCCESS;
    }
    auto ret = aclrtMemcpy(hostData->data(), size, resource.deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    }
    return ret;
}

bool ExecuteCase(aclrtStream stream, const ValidCase &tc, std::vector<double> *actual)
{
    int64_t numel = GetShapeSize(tc.shape);
    if (numel < 0 || static_cast<size_t>(numel) != tc.input.size()) {
        LOG_PRINT("[FAIL] %s invalid host input size.\n", tc.name.c_str());
        return false;
    }

    TensorResource self;
    TensorResource out;
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool ok = false;

    auto inputBytes = EncodeValues(tc.input, tc.dtype);
    std::vector<uint8_t> outputBytes(static_cast<size_t>(numel) * GetDataTypeSize(tc.dtype), 0);

    do {
        auto ret = CreateAclTensor(inputBytes, tc.shape, tc.dtype, &self);
        if (ret != ACL_SUCCESS) {
            break;
        }
        ret = CreateAclTensor(outputBytes, tc.shape, tc.dtype, &out);
        if (ret != ACL_SUCCESS) {
            break;
        }

        if (tc.api == ApiKind::Cumsum) {
            ret = aclnnCumsumGetWorkspaceSize(self.tensor, tc.dim, tc.dtype, out.tensor, &workspaceSize, &executor);
        } else {
            ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, tc.dim, tc.exclusive, tc.reverse, out.tensor,
                                                &workspaceSize, &executor);
        }
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s GetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
            break;
        }

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_PRINT("[FAIL] %s allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
                break;
            }
        }

        if (tc.api == ApiKind::Cumsum) {
            ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
        } else {
            ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
        }
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s execute failed. ERROR: %d\n", tc.name.c_str(), ret);
            break;
        }

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
            break;
        }

        std::vector<uint8_t> resultBytes;
        ret = CopyDeviceToHost(out, outputBytes.size(), &resultBytes);
        if (ret != ACL_SUCCESS) {
            break;
        }
        *actual = DecodeValues(resultBytes, tc.dtype);
        ok = true;
    } while (false);

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return ok;
}

void PrintShape(const std::vector<int64_t> &shape)
{
    LOG_PRINT("[");
    for (size_t i = 0; i < shape.size(); ++i) {
        LOG_PRINT("%ld%s", shape[i], (i + 1 == shape.size()) ? "" : ",");
    }
    LOG_PRINT("]");
}

void PrintPreview(const char *label, const std::vector<double> &values)
{
    LOG_PRINT("  %s: [", label);
    size_t preview = std::min<size_t>(values.size(), 6);
    for (size_t i = 0; i < preview; ++i) {
        LOG_PRINT("%.8g%s", values[i], (i + 1 == preview && values.size() <= preview) ? "" : ", ");
    }
    if (values.size() > preview) {
        LOG_PRINT("..., %.8g", values.back());
    }
    LOG_PRINT("]\n");
}

bool CheckCaseResult(const ValidCase &tc, const std::vector<double> &actual)
{
    auto quantizedInput = QuantizeInputForReference(tc.input, tc.dtype);
    auto expected = CpuCumsum(quantizedInput, tc.shape, tc.dim, tc.exclusive, tc.reverse);
    if (actual.size() != expected.size()) {
        LOG_PRINT("  Output size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }

    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    size_t maxIdx = 0;
    bool pass = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        double absErr = std::fabs(actual[i] - expected[i]);
        double relErr = (std::fabs(expected[i]) > 0.0) ? absErr / std::fabs(expected[i]) : absErr;
        if (absErr > maxAbsErr) {
            maxAbsErr = absErr;
            maxRelErr = relErr;
            maxIdx = i;
        }

        if (IsFloatingType(tc.dtype)) {
            double tol = tc.atol + tc.rtol * std::fabs(expected[i]);
            if (std::isnan(actual[i]) || absErr > tol) {
                pass = false;
            }
        } else if (actual[i] != expected[i]) {
            pass = false;
        }
    }

    PrintPreview("Expected", expected);
    PrintPreview("Actual  ", actual);
    LOG_PRINT("  Max abs error: %.8e, max rel error: %.8e at index %zu\n", maxAbsErr, maxRelErr, maxIdx);
    return pass;
}

bool RunValidCase(aclrtStream stream, const ValidCase &tc)
{
    LOG_PRINT("\nTest case: %s\n", tc.name.c_str());
    LOG_PRINT("  API: %s, dtype: %s, dim: %ld, exclusive: %s, reverse: %s, shape: ",
              tc.api == ApiKind::Cumsum ? "Cumsum" : "CumsumV2", DtypeName(tc.dtype), tc.dim,
              tc.exclusive ? "true" : "false", tc.reverse ? "true" : "false");
    PrintShape(tc.shape);
    LOG_PRINT("\n");

    std::vector<double> actual;
    bool pass = ExecuteCase(stream, tc, &actual) && CheckCaseResult(tc, actual);
    LOG_PRINT("  [%s]\n", pass ? "PASS" : "FAIL");
    return pass;
}

std::vector<double> Repeated(double value, size_t count)
{
    return std::vector<double>(count, value);
}

std::vector<double> Sequence(size_t count, double scale, double bias)
{
    std::vector<double> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = bias + static_cast<double>((static_cast<int>(i % 17) - 8)) * scale;
    }
    return values;
}

std::vector<double> IntSequence(size_t count)
{
    std::vector<double> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<double>((static_cast<int>(i % 9) - 4));
    }
    return values;
}

std::vector<ValidCase> BuildValidCases()
{
    std::vector<double> mixedMagnitude(64);
    for (size_t i = 0; i < mixedMagnitude.size(); ++i) {
        mixedMagnitude[i] = (i % 2 == 0) ? 1e8 : 1e-3;
    }

    return {
        {"Cumsum-F32-2D-Dim0", ApiKind::Cumsum, ACL_FLOAT, {2, 3}, 0, false, false,
         {1, 2, 3, 4, 5, 6}, 1e-5, 1e-5},
        {"Cumsum-F32-2D-Dim1-MixedSign", ApiKind::Cumsum, ACL_FLOAT, {2, 4}, 1, false, false,
         {1.5, -2.0, 3.25, -4.5, -1.0, 2.0, -3.0, 4.0}, 1e-5, 1e-5},
        {"Cumsum-F32-3D-NegDim", ApiKind::Cumsum, ACL_FLOAT, {2, 3, 4}, -1, false, false,
         Sequence(24, 0.25, 0.5), 1e-5, 1e-5},
        {"Cumsum-F32-Medium-Axis0", ApiKind::Cumsum, ACL_FLOAT, {128, 4}, 0, false, false,
         Sequence(512, 0.125, 0.0), 1e-5, 1e-5},
        {"Cumsum-F32-Long-PointOne", ApiKind::Cumsum, ACL_FLOAT, {2048}, 0, false, false,
         Repeated(0.1, 2048), 1e-4, 5e-5},
        {"Cumsum-F32-MixedMagnitude", ApiKind::Cumsum, ACL_FLOAT, {64}, 0, false, false,
         mixedMagnitude, 1e-5, 1e-5},
        {"Cumsum-F16-Short-Dim1", ApiKind::Cumsum, ACL_FLOAT16, {4, 4}, 1, false, false,
         Sequence(16, 0.25, 0.0), 1e-2, 1e-2},
        {"Cumsum-BF16-Short-Dim0", ApiKind::Cumsum, ACL_BF16, {8}, 0, false, false,
         {0.5, 0.25, 0.125, -0.875, 1.0, -0.5, 0.75, 0.25}, 1e-1, 1e-1},
        {"Cumsum-INT32-Dim1", ApiKind::Cumsum, ACL_INT32, {3, 5}, 1, false, false,
         IntSequence(15), 0.0, 0.0},
        {"Cumsum-INT64-Dim0", ApiKind::Cumsum, ACL_INT64, {4, 3}, 0, false, false,
         IntSequence(12), 0.0, 0.0},
        {"Cumsum-INT8-Dim1", ApiKind::Cumsum, ACL_INT8, {2, 6}, 1, false, false,
         {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6}, 0.0, 0.0},
        {"Cumsum-UINT8-Dim1", ApiKind::Cumsum, ACL_UINT8, {2, 4}, 1, false, false,
         {1, 2, 3, 4, 5, 6, 7, 8}, 0.0, 0.0},
        {"Cumsum-F32-EmptyTensor", ApiKind::Cumsum, ACL_FLOAT, {0}, 0, false, false,
         {}, 1e-5, 1e-5},
        {"CumsumV2-F32-EmptyTensor", ApiKind::CumsumV2, ACL_FLOAT, {0}, 0, false, false,
         {}, 1e-5, 1e-5},
        {"CumsumV2-F32-Scalar", ApiKind::CumsumV2, ACL_FLOAT, {}, 0, false, false,
         {6.25}, 1e-5, 1e-5},
        {"CumsumV2-F32-InclusiveForward", ApiKind::CumsumV2, ACL_FLOAT, {2, 4}, 1, false, false,
         {1, 2, 3, 4, -1, -2, -3, -4}, 1e-5, 1e-5},
        {"CumsumV2-F32-ExclusiveForward", ApiKind::CumsumV2, ACL_FLOAT, {2, 4}, 1, true, false,
         {1, 2, 3, 4, -1, -2, -3, -4}, 1e-5, 1e-5},
        {"CumsumV2-F32-InclusiveReverse", ApiKind::CumsumV2, ACL_FLOAT, {2, 4}, 1, false, true,
         {1, 2, 3, 4, -1, -2, -3, -4}, 1e-5, 1e-5},
        {"CumsumV2-F32-ExclusiveReverse", ApiKind::CumsumV2, ACL_FLOAT, {2, 4}, 1, true, true,
         {1, 2, 3, 4, -1, -2, -3, -4}, 1e-5, 1e-5},
        {"CumsumV2-INT32-ExclusiveReverse-Dim0", ApiKind::CumsumV2, ACL_INT32, {4, 3}, 0, true, true,
         IntSequence(12), 0.0, 0.0},
        {"CumsumV2-F16-ExclusiveForward-Dim1", ApiKind::CumsumV2, ACL_FLOAT16, {2, 5}, 1, true, false,
         {0.25, 0.5, -0.75, 1.0, -1.25, -0.5, 0.75, 1.25, -1.5, 2.0}, 1e-2, 1e-2},
        {"Cumsum-DOUBLE-AiCpu-Dim0", ApiKind::Cumsum, ACL_DOUBLE, {3, 4}, 0, false, false,
         Sequence(12, 0.125, 1.0), 1e-9, 1e-9},
        {"Cumsum-INT16-AiCpu-Dim1", ApiKind::Cumsum, ACL_INT16, {4, 8}, 1, false, false,
         IntSequence(32), 0.0, 0.0},
        {"CumsumV2-DOUBLE-AiCpu-Reverse", ApiKind::CumsumV2, ACL_DOUBLE, {2, 5}, 1, false, true,
         Sequence(10, 0.25, 0.0), 1e-9, 1e-9},
        {"Cumsum-F32-Scalar", ApiKind::Cumsum, ACL_FLOAT, {}, 0, false, false,
         {3.5}, 1e-5, 1e-5},
        {"Cumsum-F16-Cube-MinShape", ApiKind::Cumsum, ACL_FLOAT16, {12800, 512}, 1, false, false,
         Repeated(0.001, 12800 * 512), 8e-2, 8e-2},
        {"Cumsum-F32-NGreaterCl-MSplit-NFull", ApiKind::Cumsum, ACL_FLOAT, {64, 16, 16}, 1, false, false,
         Sequence(64 * 16 * 16, 0.03125, 0.0), 1e-5, 1e-5},
        {"Cumsum-F32-NGreaterCl-BorrowN", ApiKind::Cumsum, ACL_FLOAT, {2, 16, 1024}, 1, false, false,
         Sequence(2 * 16 * 1024, 0.015625, 0.0), 1e-4, 1e-5},
        {"Cumsum-F16-NGreaterCl-RNotFull-MSplit", ApiKind::Cumsum, ACL_FLOAT16, {64, 8192, 16}, 1, false, false,
         Repeated(0.001, 64 * 8192 * 16), 2e-1, 2e-1},
        {"Cumsum-F32-NGreaterCl-BorrowR", ApiKind::Cumsum, ACL_FLOAT, {1, 8192, 16}, 1, false, false,
         Repeated(0.001, 1 * 8192 * 16), 1e-3, 1e-4},
        {"Cumsum-F32-RN-Twoway-RFull", ApiKind::Cumsum, ACL_FLOAT, {64, 1024, 1}, 1, false, false,
         Repeated(0.001, 64 * 1024 * 1), 1e-3, 1e-4},
        {"Cumsum-F16-RN-Twoway-RNotFull", ApiKind::Cumsum, ACL_FLOAT16, {64, 8192, 1}, 1, false, false,
         Repeated(0.001, 64 * 8192 * 1), 2e-1, 2e-1},
        {"Cumsum-BF16-MRN-GreaterCl", ApiKind::Cumsum, ACL_BF16, {16, 1, 1}, 1, false, false,
         Sequence(16, 0.5, 0.0), 2e-1, 2e-1},
        {"Cumsum-F32-MRN-LesserCl", ApiKind::Cumsum, ACL_FLOAT, {1, 1, 1}, 1, false, false,
         {2.0}, 1e-5, 1e-5},
        {"CumsumV2-F32-5D-MiddleAxis", ApiKind::CumsumV2, ACL_FLOAT, {1, 8, 17, 9, 15}, 1, true, false,
         Sequence(1 * 8 * 17 * 9 * 15, 0.03125, 0.0), 1e-4, 1e-5},
        {"CumsumV2-F32-3D-ReverseAxis0", ApiKind::CumsumV2, ACL_FLOAT, {40, 15, 14}, -3, false, true,
         Sequence(40 * 15 * 14, 0.015625, 0.0), 1e-4, 1e-5},
        {"Cumsum-INT32-AxisLast-NoSplit", ApiKind::Cumsum, ACL_INT32, {16, 8, 32}, 2, false, false,
         IntSequence(16 * 8 * 32), 0.0, 0.0},
        {"Cumsum-INT32-Axis0-RightLarge", ApiKind::Cumsum, ACL_INT32, {8, 17, 16}, 0, false, false,
         IntSequence(8 * 17 * 16), 0.0, 0.0},
        {"Cumsum-INT8-Middle-RightLarge", ApiKind::Cumsum, ACL_INT8, {4, 128, 128}, 1, false, false,
         IntSequence(4 * 128 * 128), 0.0, 0.0},
        {"Cumsum-INT64-RBlockAxis", ApiKind::Cumsum, ACL_INT64, {1, 4096, 1}, 1, false, false,
         IntSequence(1 * 4096 * 1), 0.0, 0.0},
        {"Cumsum-UINT8-HighRank-Axis0", ApiKind::Cumsum, ACL_UINT8, {1, 8, 17, 17, 8}, 0, false, false,
         Repeated(1.0, 1 * 8 * 17 * 17 * 8), 0.0, 0.0},
        {"CumsumV2-INT32-Middle-ExclusiveForward", ApiKind::CumsumV2, ACL_INT32, {9, 17, 9, 17}, -2, true, false,
         IntSequence(9 * 17 * 9 * 17), 0.0, 0.0},
        {"Cumsum-F32-NGreaterCl-RFull-NFull", ApiKind::Cumsum, ACL_FLOAT, {64, 8, 512}, 1, false, false,
         Repeated(0.001, 64 * 8 * 512), 1e-3, 1e-4},
        {"Cumsum-F32-NGreaterCl-RFull-NSplit", ApiKind::Cumsum, ACL_FLOAT, {64, 128, 512}, 1, false, false,
         Repeated(0.001, 64 * 128 * 512), 1e-3, 1e-4},
        {"Cumsum-F32-NGreaterCl-RNotFull-MBlock", ApiKind::Cumsum, ACL_FLOAT, {33, 1024, 128}, 1, false, false,
         Repeated(0.001, 33 * 1024 * 128), 1e-3, 1e-4},
        {"Cumsum-F32-NGreaterCl-RNotFull-CoreBorrowR", ApiKind::Cumsum, ACL_FLOAT, {1, 1024, 128}, 1, false, false,
         Repeated(0.001, 1 * 1024 * 128), 1e-3, 1e-4},
        {"Cumsum-F32-NGreaterCl-RHuge-CoreUbSplit", ApiKind::Cumsum, ACL_FLOAT, {1, 32768, 128}, 1, false, false,
         Repeated(0.0001, 1 * 32768 * 128), 1e-2, 1e-4},
        {"Cumsum-F16-NGreaterCl-BorrowN-V2Reverse", ApiKind::CumsumV2, ACL_FLOAT16, {1, 32, 512}, 1, true, true,
         Repeated(0.001, 1 * 32 * 512), 2e-1, 2e-1},
        {"Cumsum-F32-RNGreater-OneWay-RFull", ApiKind::Cumsum, ACL_FLOAT, {64, 1024, 32}, 1, false, false,
         Repeated(0.001, 64 * 1024 * 32), 1e-3, 1e-4},
        {"Cumsum-F32-RNGreater-OneWay-RNotFull", ApiKind::Cumsum, ACL_FLOAT, {64, 4096, 32}, 1, false, false,
         Repeated(0.0005, 64 * 4096 * 32), 2e-3, 1e-4},
        {"Cumsum-F32-RNGreater-Twoway-BorrowM", ApiKind::Cumsum, ACL_FLOAT, {128, 1024, 1}, 1, false, false,
         Repeated(0.001, 128 * 1024 * 1), 1e-3, 1e-4},
        {"Cumsum-F32-RNGreater-Twoway-CoreBorrowR", ApiKind::Cumsum, ACL_FLOAT, {1, 4096, 1}, 1, false, false,
         Repeated(0.001, 1 * 4096 * 1), 1e-3, 1e-4},
        {"Cumsum-F32-RNGreater-Twoway-CoreUbSplit", ApiKind::Cumsum, ACL_FLOAT, {1, 65536, 1}, 1, false, false,
         Repeated(0.0001, 1 * 65536 * 1), 1e-2, 1e-4},
        {"Cumsum-F16-RNGreater-Twoway-V2Reverse", ApiKind::CumsumV2, ACL_FLOAT16, {48, 19, 43}, -2, false, true,
         Repeated(0.001, 48 * 19 * 43), 2e-1, 2e-1},
        {"Cumsum-BF16-MRNGreater-BlockSmall", ApiKind::Cumsum, ACL_BF16, {256, 1, 1}, 1, false, false,
         Repeated(0.5, 256 * 1 * 1), 2e-1, 2e-1},
        {"Cumsum-F32-MRNGreater-BlockLarge", ApiKind::Cumsum, ACL_FLOAT, {16384, 1, 1}, 1, false, false,
         Repeated(0.001, 16384 * 1 * 1), 1e-3, 1e-4},
        {"Cumsum-F32-Cube-BFShape", ApiKind::Cumsum, ACL_FLOAT, {12800, 512}, 1, false, false,
         Repeated(0.0001, 12800 * 512), 1e-2, 1e-4},
        {"CumsumV2-F32-ST-AxisLast-Exclusive", ApiKind::CumsumV2, ACL_FLOAT, {18, 25, 20}, 2, true, false,
         Repeated(0.001, 18 * 25 * 20), 1e-3, 1e-4},
        {"CumsumV2-F32-ST-Axis0-ExclusiveReverse", ApiKind::CumsumV2, ACL_FLOAT, {17, 29, 19}, 0, true, true,
         Repeated(0.001, 17 * 29 * 19), 1e-3, 1e-4},
        {"CumsumV2-F32-ST-MiddleAxis", ApiKind::CumsumV2, ACL_FLOAT, {38, 35, 23}, 1, true, false,
         Repeated(0.001, 38 * 35 * 23), 1e-3, 1e-4},
        {"CumsumV2-F16-ST-NegMiddle", ApiKind::CumsumV2, ACL_FLOAT16, {18, 18, 34}, -2, true, false,
         Repeated(0.001, 18 * 18 * 34), 2e-1, 2e-1},
        {"Cumsum-INT32-LeftAxisDominant", ApiKind::Cumsum, ACL_INT32, {1024, 4, 1}, 1, false, false,
         IntSequence(1024 * 4 * 1), 0.0, 0.0},
        {"Cumsum-INT32-RightAxisDominant", ApiKind::Cumsum, ACL_INT32, {1, 8, 4096}, 1, false, false,
         Repeated(1.0, 1 * 8 * 4096), 0.0, 0.0},
        {"Cumsum-INT32-MidAxisDominant", ApiKind::Cumsum, ACL_INT32, {1, 8192, 1}, 1, false, false,
         IntSequence(1 * 8192 * 1), 0.0, 0.0},
        {"Cumsum-INT8-RightAxisNoSplit", ApiKind::Cumsum, ACL_INT8, {1, 16, 2048}, 1, false, false,
         Repeated(1.0, 1 * 16 * 2048), 0.0, 0.0},
        {"Cumsum-UINT8-AxisLast-ArSplit", ApiKind::Cumsum, ACL_UINT8, {8, 16, 32}, 2, false, false,
         Repeated(1.0, 8 * 16 * 32), 0.0, 0.0},
        {"CumsumV2-INT32-LeftDominant-Reverse", ApiKind::CumsumV2, ACL_INT32, {513, 8, 1}, 1, false, true,
         Repeated(1.0, 513 * 8 * 1), 0.0, 0.0},
        {"CumsumV2-INT8-MidDominant-Exclusive", ApiKind::CumsumV2, ACL_INT8, {1, 4096, 1}, 1, true, false,
         IntSequence(1 * 4096 * 1), 0.0, 0.0},
        {"CumsumV2-UINT8-HighRank-LastAxis", ApiKind::CumsumV2, ACL_UINT8, {1, 16, 1, 1, 8, 16, 15}, 6, false, true,
         Repeated(1.0, 1 * 16 * 1 * 1 * 8 * 16 * 15), 0.0, 0.0},
    };
}

bool ExpectStatus(const char *name, aclnnStatus actual, aclnnStatus expected)
{
    bool pass = actual == expected;
    LOG_PRINT("\nTest case: %s\n  Expected status: %d, actual status: %d\n  [%s]\n", name, expected, actual,
              pass ? "PASS" : "FAIL");
    return pass;
}

bool RunInvalidDimCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4, 5, 6}, ACL_FLOAT);
    auto outData = EncodeValues(std::vector<double>(6, 0), ACL_FLOAT);
    if (CreateAclTensor(data, {2, 3}, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {2, 3}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, 2, ACL_FLOAT, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-DimOutOfRange", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunInvalidNegativeDimCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4, 5, 6}, ACL_FLOAT);
    auto outData = EncodeValues(std::vector<double>(6, 0), ACL_FLOAT);
    if (CreateAclTensor(data, {2, 3}, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {2, 3}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, -3, ACL_FLOAT, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-NegativeDimOutOfRange", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunDtypeMismatchCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4}, ACL_FLOAT);
    auto outData = EncodeValues(std::vector<double>(4, 0), ACL_FLOAT);
    if (CreateAclTensor(data, {4}, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {4}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_INT32, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-DtypeMismatch", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunShapeMismatchV2Case()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4, 5, 6}, ACL_FLOAT);
    auto outData = EncodeValues(std::vector<double>(6, 0), ACL_FLOAT);
    if (CreateAclTensor(data, {2, 3}, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {3, 2}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, 1, false, false, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-CumsumV2-ShapeMismatch", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunShapeMismatchCumsumCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4, 5, 6}, ACL_FLOAT);
    auto outData = EncodeValues(std::vector<double>(6, 0), ACL_FLOAT);
    if (CreateAclTensor(data, {2, 3}, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {3, 2}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, 1, ACL_FLOAT, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-ShapeMismatch", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunNullSelfCase()
{
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto outData = EncodeValues(std::vector<double>(4, 0), ACL_FLOAT);
    if (CreateAclTensor(outData, {4}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-NullSelf", ret, ACLNN_ERR_PARAM_NULLPTR);
    }
    DestroyTensorResource(&out);
    return pass;
}

bool RunNullOutCase()
{
    TensorResource self;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4}, ACL_FLOAT);
    if (CreateAclTensor(data, {4}, ACL_FLOAT, &self) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_FLOAT, nullptr, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-NullOut", ret, ACLNN_ERR_PARAM_NULLPTR);
    }
    DestroyTensorResource(&self);
    return pass;
}

bool RunV2NullSelfCase()
{
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto outData = EncodeValues(std::vector<double>(4, 0), ACL_FLOAT);
    if (CreateAclTensor(outData, {4}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumV2GetWorkspaceSize(nullptr, 0, false, false, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-CumsumV2-NullSelf", ret, ACLNN_ERR_PARAM_NULLPTR);
    }
    DestroyTensorResource(&out);
    return pass;
}

bool RunV2NullOutCase()
{
    TensorResource self;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4}, ACL_FLOAT);
    if (CreateAclTensor(data, {4}, ACL_FLOAT, &self) == ACL_SUCCESS) {
        auto ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, nullptr, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-CumsumV2-NullOut", ret, ACLNN_ERR_PARAM_NULLPTR);
    }
    DestroyTensorResource(&self);
    return pass;
}

bool RunUnsupportedDtypeCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 0, 1, 0}, ACL_BOOL);
    auto outData = EncodeValues({0, 0, 0, 0}, ACL_BOOL);
    if (CreateAclTensor(data, {4}, ACL_BOOL, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {4}, ACL_BOOL, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_BOOL, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-UnsupportedBool", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunV2UnsupportedDtypeCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 0, 1, 0}, ACL_BOOL);
    auto outData = EncodeValues({0, 0, 0, 0}, ACL_BOOL);
    if (CreateAclTensor(data, {4}, ACL_BOOL, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {4}, ACL_BOOL, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-CumsumV2-UnsupportedBool", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunV2DtypeMismatchCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4}, ACL_FLOAT);
    auto outData = EncodeValues(std::vector<double>(4, 0), ACL_INT32);
    if (CreateAclTensor(data, {4}, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {4}, ACL_INT32, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-CumsumV2-DtypeMismatch", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunRankTooHighCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto data = EncodeValues({1}, ACL_FLOAT);
    auto outData = EncodeValues({0}, ACL_FLOAT);
    if (CreateAclTensor(data, shape, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, shape, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_FLOAT, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-Cumsum-RankTooHigh", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunV2RankTooHighCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto data = EncodeValues({1}, ACL_FLOAT);
    auto outData = EncodeValues({0}, ACL_FLOAT);
    if (CreateAclTensor(data, shape, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, shape, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-CumsumV2-RankTooHigh", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

bool RunV2InvalidDimCase()
{
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    bool pass = false;
    auto data = EncodeValues({1, 2, 3, 4}, ACL_FLOAT);
    auto outData = EncodeValues(std::vector<double>(4, 0), ACL_FLOAT);
    if (CreateAclTensor(data, {2, 2}, ACL_FLOAT, &self) == ACL_SUCCESS &&
        CreateAclTensor(outData, {2, 2}, ACL_FLOAT, &out) == ACL_SUCCESS) {
        auto ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, -3, false, true, out.tensor, &workspaceSize, &executor);
        pass = ExpectStatus("Invalid-CumsumV2-DimOutOfRange", ret, ACLNN_ERR_PARAM_INVALID);
    }
    DestroyTensorResource(&self);
    DestroyTensorResource(&out);
    return pass;
}

std::vector<bool (*)()> BuildInvalidCases()
{
    return {RunNullSelfCase, RunV2NullSelfCase, RunNullOutCase, RunV2NullOutCase, RunInvalidDimCase,
            RunInvalidNegativeDimCase, RunDtypeMismatchCase,
            RunShapeMismatchCumsumCase, RunShapeMismatchV2Case, RunUnsupportedDtypeCase, RunV2UnsupportedDtypeCase,
            RunV2DtypeMismatchCase, RunRankTooHighCase, RunV2RankTooHighCase, RunV2InvalidDimCase};
}

}  // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    CaseStats stats;
    for (const auto &tc : BuildValidCases()) {
        if (RunValidCase(stream, tc)) {
            ++stats.passed;
        } else {
            ++stats.failed;
        }
    }
    for (const auto &invalidCase : BuildInvalidCases()) {
        if (invalidCase()) {
            ++stats.passed;
        } else {
            ++stats.failed;
        }
    }

    LOG_PRINT("\nSummary: %d passed, %d failed\n", stats.passed, stats.failed);

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return stats.failed == 0 ? 0 : 1;
}
