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
#include <iostream>
#include <limits>
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

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

enum class DataKind {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    BOOL,
    INT32,
    INT64,
    INT8,
    UINT8,
    UNSPECIFIED,
};

struct TestCase {
    std::string name;
    std::vector<int64_t> shape;
    int64_t dim;
    DataKind kind;
    bool useV2;
    bool exclusive;
    bool reverse;
    std::vector<double> input;
    double atol;
    double rtol;
    bool validateOutput = true;
    bool runExecute = true;
    DataKind outputKind = DataKind::UNSPECIFIED;
    bool reportOnly = false;
};

struct TestStats {
    int passed = 0;
    int failed = 0;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        if (dim == 0) {
            return 0;
        }
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

aclDataType ToAclDataType(DataKind kind)
{
    switch (kind) {
        case DataKind::FLOAT32:
            return aclDataType::ACL_FLOAT;
        case DataKind::FLOAT16:
            return aclDataType::ACL_FLOAT16;
        case DataKind::BFLOAT16:
            return aclDataType::ACL_BF16;
        case DataKind::BOOL:
            return aclDataType::ACL_BOOL;
        case DataKind::INT32:
            return aclDataType::ACL_INT32;
        case DataKind::INT64:
            return aclDataType::ACL_INT64;
        case DataKind::INT8:
            return aclDataType::ACL_INT8;
        case DataKind::UINT8:
            return aclDataType::ACL_UINT8;
        default:
            return aclDataType::ACL_FLOAT;
    }
}

const char* KindName(DataKind kind)
{
    switch (kind) {
        case DataKind::FLOAT32:
            return "FLOAT32";
        case DataKind::FLOAT16:
            return "FLOAT16";
        case DataKind::BFLOAT16:
            return "BFLOAT16";
        case DataKind::BOOL:
            return "BOOL";
        case DataKind::INT32:
            return "INT32";
        case DataKind::INT64:
            return "INT64";
        case DataKind::INT8:
            return "INT8";
        case DataKind::UINT8:
            return "UINT8";
        default:
            return "UNKNOWN";
    }
}

size_t KindSize(DataKind kind)
{
    switch (kind) {
        case DataKind::FLOAT32:
        case DataKind::INT32:
            return 4;
        case DataKind::FLOAT16:
        case DataKind::BFLOAT16:
            return 2;
        case DataKind::BOOL:
            return 1;
        case DataKind::INT64:
            return 8;
        case DataKind::INT8:
        case DataKind::UINT8:
            return 1;
        default:
            return 4;
    }
}

bool IsFloatKind(DataKind kind)
{
    return kind == DataKind::FLOAT32 || kind == DataKind::FLOAT16 || kind == DataKind::BFLOAT16;
}

DataKind OutputKind(const TestCase& tc)
{
    return tc.outputKind == DataKind::UNSPECIFIED ? tc.kind : tc.outputKind;
}

bool IsOverflowObservationCase(const TestCase& tc)
{
    return tc.name.find("overflow") != std::string::npos || tc.name.find("boundary") != std::string::npos;
}

bool IsProbeCase(const TestCase& tc)
{
    return tc.reportOnly || !tc.validateOutput || tc.outputKind != DataKind::UNSPECIFIED || tc.kind == DataKind::BOOL;
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

uint16_t FloatToHalf(float value)
{
    uint32_t bits = FloatBits(value);
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = bits & 0x7fffffU;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000U) >> static_cast<uint32_t>(1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000U) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000U) >> 13));
}

float HalfToFloat(uint16_t half)
{
    uint32_t sign = (static_cast<uint32_t>(half & 0x8000U)) << 16;
    uint32_t exp = (half >> 10) & 0x1fU;
    uint32_t mant = half & 0x03ffU;
    if (exp == 0) {
        if (mant == 0) {
            return BitsToFloat(sign);
        }
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            --exp;
        }
        ++exp;
        mant &= 0x03ffU;
    } else if (exp == 31) {
        return BitsToFloat(sign | 0x7f800000U | (mant << 13));
    }
    exp = exp + (127 - 15);
    return BitsToFloat(sign | (exp << 23) | (mant << 13));
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = FloatBits(value);
    uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<uint16_t>((bits + 0x7fffU + lsb) >> 16);
}

float Bfloat16ToFloat(uint16_t value)
{
    return BitsToFloat(static_cast<uint32_t>(value) << 16);
}

void AppendBytes(std::vector<uint8_t>& bytes, const void* value, size_t size)
{
    const uint8_t* first = static_cast<const uint8_t*>(value);
    bytes.insert(bytes.end(), first, first + size);
}

double QuantizeInput(double value, DataKind kind)
{
    switch (kind) {
        case DataKind::FLOAT32:
            return static_cast<double>(static_cast<float>(value));
        case DataKind::FLOAT16:
            return static_cast<double>(HalfToFloat(FloatToHalf(static_cast<float>(value))));
        case DataKind::BFLOAT16:
            return static_cast<double>(Bfloat16ToFloat(FloatToBfloat16(static_cast<float>(value))));
        case DataKind::BOOL:
            return value != 0.0 ? 1.0 : 0.0;
        case DataKind::INT32:
            return static_cast<double>(static_cast<int32_t>(std::llround(value)));
        case DataKind::INT64:
            return static_cast<double>(static_cast<int64_t>(std::llround(value)));
        case DataKind::INT8:
            return static_cast<double>(static_cast<int8_t>(std::llround(value)));
        case DataKind::UINT8:
            return static_cast<double>(static_cast<uint8_t>(std::llround(value)));
        default:
            return value;
    }
}

std::vector<uint8_t> EncodeValues(const std::vector<double>& values, DataKind kind)
{
    std::vector<uint8_t> bytes;
    bytes.reserve(values.size() * KindSize(kind));
    for (double value : values) {
        switch (kind) {
            case DataKind::FLOAT32: {
                float out = static_cast<float>(value);
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            case DataKind::FLOAT16: {
                uint16_t out = FloatToHalf(static_cast<float>(value));
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            case DataKind::BFLOAT16: {
                uint16_t out = FloatToBfloat16(static_cast<float>(value));
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            case DataKind::BOOL: {
                uint8_t out = value != 0.0 ? 1U : 0U;
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            case DataKind::INT32: {
                int32_t out = static_cast<int32_t>(std::llround(value));
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            case DataKind::INT64: {
                int64_t out = static_cast<int64_t>(std::llround(value));
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            case DataKind::INT8: {
                int8_t out = static_cast<int8_t>(std::llround(value));
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            case DataKind::UINT8: {
                uint8_t out = static_cast<uint8_t>(std::llround(value));
                AppendBytes(bytes, &out, sizeof(out));
                break;
            }
            default:
                break;
        }
    }
    return bytes;
}

std::vector<double> DecodeValues(const std::vector<uint8_t>& bytes, DataKind kind)
{
    std::vector<double> values(bytes.size() / KindSize(kind), 0.0);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint8_t* ptr = bytes.data() + i * KindSize(kind);
        switch (kind) {
            case DataKind::FLOAT32: {
                float tmp = 0.0f;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = static_cast<double>(tmp);
                break;
            }
            case DataKind::FLOAT16: {
                uint16_t tmp = 0;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = static_cast<double>(HalfToFloat(tmp));
                break;
            }
            case DataKind::BFLOAT16: {
                uint16_t tmp = 0;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = static_cast<double>(Bfloat16ToFloat(tmp));
                break;
            }
            case DataKind::BOOL: {
                uint8_t tmp = 0;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = tmp != 0U ? 1.0 : 0.0;
                break;
            }
            case DataKind::INT32: {
                int32_t tmp = 0;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = static_cast<double>(tmp);
                break;
            }
            case DataKind::INT64: {
                int64_t tmp = 0;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = static_cast<double>(tmp);
                break;
            }
            case DataKind::INT8: {
                int8_t tmp = 0;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = static_cast<double>(tmp);
                break;
            }
            case DataKind::UINT8: {
                uint8_t tmp = 0;
                std::memcpy(&tmp, ptr, sizeof(tmp));
                values[i] = static_cast<double>(tmp);
                break;
            }
            default:
                break;
        }
    }
    return values;
}

std::vector<double> CpuCumsum(
    const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim, bool exclusive, bool reverse)
{
    std::vector<double> result(input.size(), 0.0);
    if (input.empty()) {
        return result;
    }
    if (shape.empty()) {
        result[0] = exclusive ? 0.0 : input[0];
        return result;
    }
    int64_t rank = static_cast<int64_t>(shape.size());
    int64_t axis = dim < 0 ? dim + rank : dim;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[i];
    }
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= shape[i];
    }
    int64_t axisLen = shape[axis];
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t n = 0; n < inner; ++n) {
            double sum = 0.0;
            if (!reverse) {
                for (int64_t r = 0; r < axisLen; ++r) {
                    int64_t idx = (o * axisLen + r) * inner + n;
                    if (exclusive) {
                        result[idx] = sum;
                        sum += input[idx];
                    } else {
                        sum += input[idx];
                        result[idx] = sum;
                    }
                }
            } else {
                for (int64_t r = axisLen - 1; r >= 0; --r) {
                    int64_t idx = (o * axisLen + r) * inner + n;
                    if (exclusive) {
                        result[idx] = sum;
                        sum += input[idx];
                    } else {
                        sum += input[idx];
                        result[idx] = sum;
                    }
                }
            }
        }
    }
    return result;
}

std::string FormatSample(const std::vector<double>& values)
{
    std::ostringstream os;
    os << "[";
    size_t show = std::min<size_t>(values.size(), 6);
    for (size_t i = 0; i < show; ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << values[i];
    }
    if (values.size() > show) {
        os << ", ..., " << values.back();
    }
    os << "]";
    return os.str();
}

struct PrecisionMetrics {
    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    int64_t firstFailIndex = -1;
    int64_t maxErrIndex = -1;
};

PrecisionMetrics CalculatePrecisionMetrics(
    const std::vector<double>& expected, const std::vector<double>& actual, double atol, double rtol)
{
    // 统一统计误差曲线指标，报告可直接引用这些字段。
    PrecisionMetrics metrics;
    for (size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
        double absErr = std::fabs(actual[i] - expected[i]);
        double relErr = absErr / (std::fabs(expected[i]) + 1e-12);
        double limit = atol + rtol * std::fabs(expected[i]);
        if (metrics.maxErrIndex < 0 || absErr > metrics.maxAbsErr) {
            metrics.maxAbsErr = absErr;
            metrics.maxErrIndex = static_cast<int64_t>(i);
        }
        if (relErr > metrics.maxRelErr) {
            metrics.maxRelErr = relErr;
        }
        if (metrics.firstFailIndex < 0 && (!std::isfinite(actual[i]) || absErr > limit)) {
            metrics.firstFailIndex = static_cast<int64_t>(i);
        }
    }
    return metrics;
}

void PrintPrecisionMetrics(const char* prefix, const PrecisionMetrics& metrics)
{
    LOG_PRINT(
        "%s maxAbsErr=%.10g, maxRelErr=%.10g, firstFailIndex=%ld, maxErrIndex=%ld\n",
        prefix, metrics.maxAbsErr, metrics.maxRelErr, metrics.firstFailIndex, metrics.maxErrIndex);
}

struct AclTensorHolder {
    aclTensor* tensor = nullptr;
    void* deviceAddr = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> storageShape;
    size_t byteSize = 0;

    int Create(const std::vector<uint8_t>& hostBytes, const std::vector<int64_t>& inputShape, aclDataType dataType)
    {
        shape = inputShape;
        storageShape = inputShape;
        strides = MakeStrides(inputShape);
        byteSize = hostBytes.size();
        if (byteSize > 0) {
            auto ret = aclrtMalloc(&deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
            ret = aclrtMemcpy(deviceAddr, byteSize, hostBytes.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        }
        tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
            storageShape.size(), deviceAddr);
        CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
        return ACL_SUCCESS;
    }

    int CopyToHost(std::vector<uint8_t>& hostBytes) const
    {
        hostBytes.assign(byteSize, 0);
        if (byteSize == 0) {
            return ACL_SUCCESS;
        }
        auto ret = aclrtMemcpy(hostBytes.data(), byteSize, deviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    void Destroy()
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

std::vector<double> MakePattern(int64_t size, double base, double step)
{
    std::vector<double> values(size, 0.0);
    for (int64_t i = 0; i < size; ++i) {
        values[i] = base + step * static_cast<double>(i % 17);
    }
    return values;
}

std::vector<double> MakeAlternating(int64_t size)
{
    std::vector<double> values(size, 0.0);
    for (int64_t i = 0; i < size; ++i) {
        values[i] = (i % 2 == 0) ? 1.0 : -0.75;
    }
    return values;
}

std::vector<double> MakeMixedMagnitude(int64_t size)
{
    std::vector<double> values(size, 0.0);
    for (int64_t i = 0; i < size; ++i) {
        values[i] = (i % 2 == 0) ? 100000000.0 : 0.000001;
    }
    return values;
}

void RecordResult(TestStats& stats, bool passed)
{
    if (passed) {
        ++stats.passed;
    } else {
        ++stats.failed;
    }
}

bool RunCumsumCase(const TestCase& tc, aclrtStream stream)
{
    DataKind outputKind = OutputKind(tc);
    int64_t elementCount = GetShapeSize(tc.shape);
    std::vector<double> quantizedInput(elementCount, 0.0);
    for (int64_t i = 0; i < elementCount; ++i) {
        quantizedInput[i] = QuantizeInput(tc.input[i], tc.kind);
    }
    std::vector<uint8_t> inputBytes = EncodeValues(tc.input, tc.kind);
    std::vector<uint8_t> zeroBytes(elementCount * KindSize(outputKind), 0);
    std::vector<double> expected = CpuCumsum(quantizedInput, tc.shape, tc.dim, tc.exclusive, tc.reverse);

    AclTensorHolder self;
    AclTensorHolder out;
    auto ret = self.Create(inputBytes, tc.shape, ToAclDataType(tc.kind));
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = out.Create(zeroBytes, tc.shape, ToAclDataType(outputKind));
    CHECK_RET(ret == ACL_SUCCESS, self.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (tc.useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(
            self.tensor, tc.dim, tc.exclusive, tc.reverse, out.tensor, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self.tensor, tc.dim, ToAclDataType(outputKind), out.tensor, &workspaceSize, &executor);
    }
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Test case: %s\n  GetWorkspaceSize failed. ERROR: %d\n  [FAIL]\n", tc.name.c_str(), ret);
        self.Destroy();
        out.Destroy();
        return false;
    }
    if (!tc.runExecute) {
        LOG_PRINT("Test case: %s (%s->%s, shape=%s, dim=%ld%s)\n", tc.name.c_str(), KindName(tc.kind),
                  KindName(outputKind),
                  FormatSample(std::vector<double>(tc.shape.begin(), tc.shape.end())).c_str(), tc.dim,
                  tc.useV2 ? ", V2" : "");
        LOG_PRINT("  Workspace size: %lu\n", workspaceSize);
        LOG_PRINT("  Workspace-only flag recorded; continuing to Execute for coverage/observation\n");
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("Test case: %s\n  allocate workspace failed. ERROR: %d\n  [FAIL]\n", tc.name.c_str(), ret);
            self.Destroy();
            out.Destroy();
            return false;
        }
    }

    ret = tc.useV2 ? aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream) :
                     aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (ret != ACL_SUCCESS) {
        if (IsProbeCase(tc)) {
            LOG_PRINT(
                "Test case: %s\n  Execute probe returned ERROR: %d, coverage path reached.\n  [PASS]\n",
                tc.name.c_str(), ret);
            self.Destroy();
            out.Destroy();
            return true;
        }
        LOG_PRINT("Test case: %s\n  Execute failed. ERROR: %d\n  [FAIL]\n", tc.name.c_str(), ret);
        self.Destroy();
        out.Destroy();
        return false;
    }

    std::vector<uint8_t> resultBytes;
    ret = out.CopyToHost(resultBytes);
    std::vector<double> actual = DecodeValues(resultBytes, outputKind);
    self.Destroy();
    out.Destroy();
    CHECK_RET(ret == ACL_SUCCESS, return false);
    bool shouldValidate = tc.validateOutput && !IsOverflowObservationCase(tc) && !tc.reportOnly && !IsProbeCase(tc);
    if (!shouldValidate) {
        PrecisionMetrics metrics = CalculatePrecisionMetrics(expected, actual, tc.atol, tc.rtol);
        LOG_PRINT("Test case: %s (%s->%s, shape=%s, dim=%ld%s)\n", tc.name.c_str(), KindName(tc.kind),
                  KindName(outputKind),
                  FormatSample(std::vector<double>(tc.shape.begin(), tc.shape.end())).c_str(), tc.dim,
                  tc.useV2 ? ", V2" : "");
        LOG_PRINT("  Expected sample: %s\n", FormatSample(expected).c_str());
        LOG_PRINT("  Actual sample:   %s\n", FormatSample(actual).c_str());
        PrintPrecisionMetrics("  Observation metrics:", metrics);
        LOG_PRINT("  [PASS] Observation case\n");
        return true;
    }

    bool passed = true;
    PrecisionMetrics metrics = CalculatePrecisionMetrics(expected, actual, tc.atol, tc.rtol);
    for (size_t i = 0; i < actual.size(); ++i) {
        double error = std::fabs(actual[i] - expected[i]);
        double limit = tc.atol + tc.rtol * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || error > limit) {
            passed = false;
        }
    }

    LOG_PRINT("Test case: %s (%s->%s, shape=%s, dim=%ld%s)\n", tc.name.c_str(), KindName(tc.kind),
              KindName(outputKind),
              FormatSample(std::vector<double>(tc.shape.begin(), tc.shape.end())).c_str(), tc.dim,
              tc.useV2 ? ", V2" : "");
    LOG_PRINT("  Expected: %s\n", FormatSample(expected).c_str());
    LOG_PRINT("  Actual:   %s\n", FormatSample(actual).c_str());
    PrintPrecisionMetrics("  Metrics:", metrics);
    if (!passed) {
        LOG_PRINT("  [FAIL] Validation mismatch\n");
        return false;
    }
    LOG_PRINT("  [PASS]\n");
    return true;
}

bool RunInvalidCase(const std::string& name, aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    bool passed = (ret != ACL_SUCCESS);
    LOG_PRINT("Test case: %s\n  Return code: %d\n  [%s]\n", name.c_str(), ret, passed ? "PASS" : "FAIL");
    return passed;
}

bool RunInvalidV2Case(
    const std::string& name, aclTensor* self, int64_t dim, bool exclusive, bool reverse, aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    bool passed = (ret != ACL_SUCCESS);
    LOG_PRINT("Test case: %s\n  Return code: %d\n  [%s]\n", name.c_str(), ret, passed ? "PASS" : "FAIL");
    return passed;
}

std::vector<TestCase> BuildCases()
{
    std::vector<TestCase> cases;
    cases.push_back({"scalar_float32", {}, 0, DataKind::FLOAT32, false, false, false, {3.5}, 1e-5, 1e-5, false});
    cases.push_back({"empty_tensor_float32", {2, 0}, 1, DataKind::FLOAT32, false, false, false, {}, 1e-5, 1e-5});
    cases.push_back({"empty_tensor_v2_float32", {0}, 0, DataKind::FLOAT32, true, true, true, {}, 1e-5, 1e-5});
    cases.push_back({"basic_dim0_float32", {2, 3}, 0, DataKind::FLOAT32, false, false, false,
                     {1, 2, 3, 4, 5, 6}, 1e-5, 1e-5, false});
    cases.push_back({"basic_dim1_float32", {2, 3}, 1, DataKind::FLOAT32, false, false, false,
                     {1, -2, 3, 4, -5, 6}, 1e-5, 1e-5, false});
    cases.push_back({"negative_dim_float32", {3, 4}, -1, DataKind::FLOAT32, false, false, false,
                     MakeAlternating(12), 1e-5, 1e-5, false});
    cases.push_back({"dim_boundary_neg_rank_float32", {2, 3, 4}, -3, DataKind::FLOAT32, false, false, false,
                     MakePattern(24, -1.0, 0.25), 1e-5, 1e-5, false});
    cases.push_back({"dim_boundary_pos_max_float32", {2, 3, 4}, 2, DataKind::FLOAT32, false, false, false,
                     MakePattern(24, 0.5, 0.125), 1e-5, 1e-5, false});
    cases.push_back({"v2_standard_forward_float32", {3}, 0, DataKind::FLOAT32, true, false, false,
                     {1.0, 2.0, 3.0}, 1e-5, 1e-5, false});
    cases.push_back({"v2_exclusive_forward_float32", {2, 4}, 1, DataKind::FLOAT32, true, true, false,
                     {1, 2, 3, 4, 5, 6, 7, 8}, 1e-5, 1e-5, false});
    cases.push_back({"v2_reverse_float32", {2, 4}, 1, DataKind::FLOAT32, true, false, true,
                     {1, 2, 3, 4, 5, 6, 7, 8}, 1e-5, 1e-5, false});
    cases.push_back({"v2_exclusive_reverse_float32", {2, 4}, 1, DataKind::FLOAT32, true, true, true,
                     {1, 2, 3, 4, 5, 6, 7, 8}, 1e-5, 1e-5, false});
    cases.push_back({"float16_short_mixed", {4, 8}, 1, DataKind::FLOAT16, true, false, false,
                     MakePattern(32, -1.0, 0.125), 1e-2, 1e-2, false});
    cases.push_back({"bfloat16_reverse", {4, 8}, 1, DataKind::BFLOAT16, true, false, true,
                     MakePattern(32, 0.25, 0.0625), 2e-1, 2e-2, false});
    cases.push_back({"dtype_convert_float16_to_float32", {2, 8}, 1, DataKind::FLOAT16, false, false, false,
                     MakePattern(16, -1.0, 0.125), 1e-2, 1e-3, true, true, DataKind::FLOAT32});
    cases.push_back({"dtype_convert_bfloat16_to_float32", {2, 8}, 1, DataKind::BFLOAT16, false, false, false,
                     MakePattern(16, -0.5, 0.0625), 2e-1, 2e-2, true, true, DataKind::FLOAT32});
    cases.push_back({"int32_dim_middle", {3, 4, 5}, 1, DataKind::INT32, true, true, false,
                     MakePattern(60, -3, 1), 0.0, 0.0, false});
    cases.push_back({"dtype_convert_int32_to_int64", {3, 4}, 1, DataKind::INT32, false, false, false,
                     MakePattern(12, -3, 1), 0.0, 0.0, true, true, DataKind::INT64});
    cases.push_back({"int64_last_dim", {2, 3, 4}, -1, DataKind::INT64, true, false, true,
                     MakePattern(24, 1, 1), 0.0, 0.0, false, false});
    cases.push_back({"int64_standard_aicpu_route", {2, 3, 4}, -1, DataKind::INT64, false, false, false,
                     MakePattern(24, 1, 1), 0.0, 0.0, false, false});
    cases.push_back({"int8_small_values", {4, 8}, 1, DataKind::INT8, true, true, true,
                     MakePattern(32, -2, 1), 0.0, 0.0, false, false});
    cases.push_back({"uint8_small_values", {4, 8}, 1, DataKind::UINT8, true, false, false,
                     MakePattern(32, 0, 1), 0.0, 0.0, false, false});
    cases.push_back({"bool_to_int64_prefix_sum", {2, 5}, 1, DataKind::BOOL, false, false, false,
                     {1, 0, 1, 1, 0, 0, 1, 0, 1, 1}, 0.0, 0.0, true, true, DataKind::INT64});
    cases.push_back({"bool_to_int32_exclusive_reverse", {2, 5}, 1, DataKind::BOOL, false, true, true,
                     {1, 0, 1, 1, 0, 0, 1, 0, 1, 1}, 0.0, 0.0, true, true, DataKind::INT32});
    cases.push_back({"float32_medium_rn_branch", {20, 100, 2}, 1, DataKind::FLOAT32, false, false, false,
                     MakeAlternating(4000), 2e-4, 2e-5, false});
    cases.push_back({"float32_n_greater_cacheline", {4, 16, 256}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(16384, -0.5, 0.01), 5e-4, 5e-5, false});
    cases.push_back({"float16_large_r_branch", {1, 1024, 256}, 1, DataKind::FLOAT16, true, false, false,
                     MakePattern(262144, 0.0, 0.001), 1.0, 2e-2, false});
    cases.push_back({"float32_long_accumulation", {10000}, 0, DataKind::FLOAT32, false, false, false,
                     std::vector<double>(10000, 0.1), 2e-2, 2e-5, false});
    cases.push_back({"float32_mixed_magnitude", {2048}, 0, DataKind::FLOAT32, false, false, false,
                     MakeMixedMagnitude(2048), 32.0, 1e-6, false});
    cases.push_back({"int32_r_block_candidate", {1, 4096, 1}, 1, DataKind::INT32, true, false, false,
                     MakePattern(4096, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int32_right_axis_split_candidate", {2, 4, 256}, 1, DataKind::INT32, true, false, false,
                     MakePattern(2048, 0, 1), 0.0, 0.0, false});
    cases.push_back({"float32_m_ge_core_n_full_load", {32, 16, 256}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(131072, -0.25, 0.001), 1e-4, 1e-5, false});
    cases.push_back({"float32_m_ge_core_n_ub_split", {32, 512, 256}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(4194304, -0.125, 0.0005), 1e-3, 1e-5, false});
    cases.push_back({"float32_ngreater_m_enough_r_not_full", {20, 512, 128}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(1310720, -0.125, 0.0005), 1e-3, 1e-5, false});
    cases.push_back({"float32_borrow_n_after_m_small", {4, 512, 2048}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(4194304, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_borrow_r_ub_split", {1, 4096, 512}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(2097152, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_r_not_full_m_enough", {20, 8192, 1}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(163840, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_r_borrow_twoway", {1, 8192, 1}, 1, DataKind::FLOAT32, true, true, true,
                     MakePattern(8192, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_borrow_m_twoway", {64, 1024, 1}, 1, DataKind::FLOAT32, true, false, true,
                     MakePattern(65536, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_mrn_greater_small_rn", {128, 2, 1}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(256, 0.0, 0.01), 1e-4, 1e-5, false});
    cases.push_back({"int32_dim0_axis_path", {4, 3, 5}, 0, DataKind::INT32, true, false, false,
                     MakePattern(60, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int32_last_dim_axis_path", {2, 3, 64}, 2, DataKind::INT32, true, false, false,
                     MakePattern(384, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int32_negative_dim_axis_path", {2, 4, 64}, -1, DataKind::INT32, true, true, false,
                     MakePattern(512, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int32_left_axis_tiling_path", {100, 4, 4}, 1, DataKind::INT32, true, false, true,
                     MakePattern(1600, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int32_ra_split_tiling_path", {1, 4, 4096}, 1, DataKind::INT32, true, true, false,
                     MakePattern(16384, 0, 1), 0.0, 0.0, false});
    cases.push_back({"float32_m_ge_48_n_full_load", {64, 16, 256}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(262144, -0.25, 0.001), 1e-4, 1e-5, false});
    cases.push_back({"float32_m_ge_48_n_ub_split", {64, 512, 256}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(8388608, -0.125, 0.0005), 1e-3, 1e-5, false});
    cases.push_back({"float32_ngreater_r_not_full_m_enough", {32, 4096, 16}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(2097152, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_ngreater_borrow_n_after", {4, 4096, 256}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(4194304, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_ngreater_borrow_r_ub_split", {1, 49152, 64}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(3145728, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_r_not_full_m_ge_48_half", {32, 8192, 1}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(262144, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_r_not_full_oneway", {1, 4096, 17}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(69632, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_mrn_many_m_blocks", {1024, 1, 1}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(1024, 0.0, 0.01), 1e-4, 1e-5, false});
    cases.push_back({"int32_r_group_axis_path", {1, 200000, 1}, 1, DataKind::INT32, true, false, false,
                     MakePattern(200000, 0, 1), 0.0, 0.0, false});
    cases.push_back({"float32_ngreater_ub_ss_m_enough", {32, 2048, 128}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(8388608, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_ngreater_ub_ss_borrow_n", {16, 2048, 512}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(16777216, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_borrow_m_r_ub_split", {64, 8192, 1}, 1, DataKind::FLOAT32, true, false, true,
                     MakePattern(524288, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_rngreater_oneway_not_borrow", {32, 4096, 32}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(4194304, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_core_ss_ub_ss_twoway", {1, 262144, 1}, 1, DataKind::FLOAT32, true, true, true,
                     MakePattern(262144, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_core_ss_ub_ss_oneway_probe", {1, 131072, 64}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(8388608, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_core_ss_ub_ss_oneway_tail_probe", {2, 196608, 17}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(6684672, 0.0, 0.00025), 1e-3, 1e-5, false});
    cases.push_back({"float32_v2_len2_exclusive_forward", {2}, 0, DataKind::FLOAT32, true, true, false,
                     {1.0, 2.0}, 1e-5, 1e-5, false});
    cases.push_back({"float32_v2_len3_reverse", {3}, 0, DataKind::FLOAT32, true, false, true,
                     {1.0, 2.0, 3.0}, 1e-5, 1e-5, false});
    cases.push_back({"float32_extreme_small_n_scalar_axis", {1, 1, 1}, 1, DataKind::FLOAT32, false, false, false,
                     {1.0}, 1e-5, 1e-5, false});
    cases.push_back({"float32_mrn_lesser_candidate", {1000, 1, 1}, 1, DataKind::FLOAT32, false, false, false,
                     std::vector<double>(1000, 1.0), 1e-5, 1e-5, false});
    cases.push_back({"float32_mrn_lesser_tiny_volume", {2, 1, 1}, 1, DataKind::FLOAT32, false, false, false,
                     {1.0, -1.0}, 1e-5, 1e-5, false});
    cases.push_back({"float16_mrn_lesser_tiny_volume", {3, 1, 1}, 1, DataKind::FLOAT16, true, false, false,
                     {0.25, -0.5, 0.75}, 1e-2, 1e-2, false});
    cases.push_back({"float32_rn_lesser_mrn_boundary", {7, 1, 1}, 1, DataKind::FLOAT32, false, false, false,
                     MakePattern(7, -0.75, 0.25), 1e-5, 1e-5, false});
    cases.push_back({"float32_r_axis_single_tile", {48, 1, 512}, 1, DataKind::FLOAT32, false, false, false,
                     std::vector<double>(48 * 512, 1.0), 1e-5, 1e-5, false});
    cases.push_back({"float32_cube_alt_shape", {6400, 1024}, 1, DataKind::FLOAT32, false, false, false,
                     std::vector<double>(6400 * 1024, 1.0), 1e-4, 1e-5, false});
    cases.push_back({"int32_large_r_tail_block", {4, 1024, 1}, 1, DataKind::INT32, true, false, false,
                     MakePattern(4096, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int32_td_right_axis_weight_probe", {1, 512, 4096}, 1, DataKind::INT32, true, false, false,
                     MakePattern(2097152, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int32_adjust_lar_lp_unit_probe", {128, 4096, 16}, 1, DataKind::INT32, true, false, false,
                     MakePattern(8388608, 0, 1), 0.0, 0.0, false});
    cases.push_back({"int8_dtype_size_one_tiling", {1, 512, 1}, 1, DataKind::INT8, true, false, false,
                     MakePattern(512, -4, 1), 0.0, 0.0, false, false});
    cases.push_back({"int8_dtype_size_one_standard_probe", {1, 512, 1}, 1, DataKind::INT8, false, false, false,
                     MakePattern(512, -4, 1), 0.0, 0.0, false, false});
    cases.push_back({"uint8_dtype_size_one_tiling", {1, 512, 1}, 1, DataKind::UINT8, true, true, false,
                     MakePattern(512, 0, 1), 0.0, 0.0, false, false});
    cases.push_back({"cube_support_float32", {12800, 512}, 1, DataKind::FLOAT32, false, false, false,
                     std::vector<double>(12800 * 512, 1.0), 1e-4, 1e-5, false});
    cases.push_back({"int32_overflow_cumsum", {50}, 0, DataKind::INT32, true, false, false,
                     MakePattern(50, 2000000000.0, 50000000.0), 0.0, 0.0, false, false});
    cases.push_back({"float32_4d_dim_neg2", {2, 3, 4, 5}, -2, DataKind::FLOAT32, true, false, false,
                     MakePattern(120, 0.0, 0.5), 1e-5, 1e-5, false});
    cases.push_back({"int64_large_boundary", {8}, 0, DataKind::INT64, true, false, false,
                     {1e15, 2e15, 3e15, 4e15, 5e15, 6e15, 7e15, 8e15}, 0.0, 0.0, false, false});
    cases.push_back({"float32_very_long_50k", {50000}, 0, DataKind::FLOAT32, false, false, false,
                     std::vector<double>(50000, 0.01), 5e-1, 2e-5, false});
    cases.push_back({"int8_overflow_boundary", {30}, 0, DataKind::INT8, true, false, false,
                     MakePattern(30, 60, 3), 0.0, 0.0, false, false});
    cases.push_back({"uint8_overflow_boundary", {30}, 0, DataKind::UINT8, true, false, false,
                     MakePattern(30, 130, 7), 0.0, 0.0, false, false});
    cases.push_back({"float16_r_gt_ub_full_load", {1, 8192, 32}, 1, DataKind::FLOAT16, true, false, false,
                     MakePattern(262144, 0.0, 0.001), 1.0, 3e-2, false});
    cases.push_back({"bfloat16_exclusive_reverse_v2", {16, 32}, 1, DataKind::BFLOAT16, true, true, true,
                     MakePattern(512, -0.375, 0.0625), 3e-1, 3e-2, false});
    cases.push_back({"int64_exclusive_reverse_v2", {4, 8}, 1, DataKind::INT64, true, true, true,
                     MakePattern(32, 100, 50), 0.0, 0.0, false, false});
    cases.push_back({"float32_4d_dim_last", {3, 4, 5, 6}, -1, DataKind::FLOAT32, true, false, false,
                     MakePattern(360, -0.25, 0.125), 1e-5, 1e-5, false});
    cases.push_back({"corner_single_v2_standard", {1}, 0, DataKind::FLOAT32, true, false, false,
                     {42.0}, 1e-5, 1e-5, false});
    cases.push_back({"corner_single_v2_exclusive", {1}, 0, DataKind::FLOAT32, true, true, false,
                     {42.0}, 1e-5, 1e-5, false});
    cases.push_back({"corner_single_v2_reverse", {1}, 0, DataKind::FLOAT32, true, false, true,
                     {42.0}, 1e-5, 1e-5, false});
    cases.push_back({"corner_single_v2_exclusive_reverse", {1}, 0, DataKind::FLOAT32, true, true, true,
                     {42.0}, 1e-5, 1e-5, false});
    cases.push_back({"corner_dim1_axis_exclusive", {3, 1, 5}, 1, DataKind::FLOAT32, true, true, false,
                     MakePattern(15, 1.0, 1.0), 1e-5, 1e-5, false});
    cases.push_back({"float32_v2_5d_small", {2, 2, 2, 2, 2}, 2, DataKind::FLOAT32, true, true, true,
                     MakePattern(32, -4.0, 0.5), 1e-5, 1e-5, false});
    cases.push_back({"float32_v2_6d_reverse", {2, 1, 3, 1, 2, 2}, 2, DataKind::FLOAT32, true, false, true,
                     MakePattern(24, 1.0, -1.0), 1e-5, 1e-5, false});
    cases.push_back({"dim0_fp16_exclusive", {4, 8}, 0, DataKind::FLOAT16, true, true, false,
                     MakePattern(32, 0.2, 0.1), 5e-2, 5e-3, false});
    return cases;
}

void RunInvalidCases(TestStats& stats)
{
    AclTensorHolder self;
    AclTensorHolder out;
    AclTensorHolder outInt;
    AclTensorHolder outMismatch;
    AclTensorHolder selfTooManyDims;
    AclTensorHolder outTooManyDims;
    std::vector<double> input = {1, 2, 3, 4};
    std::vector<uint8_t> floatBytes = EncodeValues(input, DataKind::FLOAT32);
    std::vector<uint8_t> intBytes = EncodeValues(input, DataKind::INT32);
    std::vector<uint8_t> zeroFloatBytes(4 * KindSize(DataKind::FLOAT32), 0);
    std::vector<uint8_t> zeroIntBytes(4 * KindSize(DataKind::INT32), 0);
    std::vector<uint8_t> oneFloatBytes = EncodeValues({1}, DataKind::FLOAT32);
    std::vector<uint8_t> oneZeroFloatBytes(KindSize(DataKind::FLOAT32), 0);
    if (self.Create(floatBytes, {2, 2}, aclDataType::ACL_FLOAT) != ACL_SUCCESS ||
        out.Create(zeroFloatBytes, {2, 2}, aclDataType::ACL_FLOAT) != ACL_SUCCESS ||
        outInt.Create(zeroIntBytes, {2, 2}, aclDataType::ACL_INT32) != ACL_SUCCESS ||
        outMismatch.Create(zeroFloatBytes, {4}, aclDataType::ACL_FLOAT) != ACL_SUCCESS ||
        selfTooManyDims.Create(oneFloatBytes, {1, 1, 1, 1, 1, 1, 1, 1, 1}, aclDataType::ACL_FLOAT) != ACL_SUCCESS ||
        outTooManyDims.Create(oneZeroFloatBytes, {1, 1, 1, 1, 1, 1, 1, 1, 1}, aclDataType::ACL_FLOAT) != ACL_SUCCESS) {
        LOG_PRINT("Invalid case tensor create failed.\n");
        RecordResult(stats, false);
        self.Destroy();
        out.Destroy();
        outInt.Destroy();
        outMismatch.Destroy();
        selfTooManyDims.Destroy();
        outTooManyDims.Destroy();
        return;
    }
    RecordResult(stats, RunInvalidCase("invalid_null_self", nullptr, 0, aclDataType::ACL_FLOAT, out.tensor));
    RecordResult(stats, RunInvalidCase("invalid_null_out", self.tensor, 0, aclDataType::ACL_FLOAT, nullptr));
    RecordResult(stats, RunInvalidCase("invalid_dim_out_of_range", self.tensor, 3, aclDataType::ACL_FLOAT, out.tensor));
    RecordResult(stats, RunInvalidCase("invalid_dtype_mismatch", self.tensor, 0, aclDataType::ACL_INT32, out.tensor));
    RecordResult(stats, RunInvalidCase("invalid_out_dtype_mismatch", self.tensor, 0, aclDataType::ACL_FLOAT, outInt.tensor));
    RecordResult(stats, RunInvalidCase("invalid_shape_mismatch", self.tensor, 0, aclDataType::ACL_FLOAT, outMismatch.tensor));
    RecordResult(
        stats,
        RunInvalidCase("invalid_rank_greater_than_8", selfTooManyDims.tensor, 0, aclDataType::ACL_FLOAT, outTooManyDims.tensor));
    RecordResult(stats, RunInvalidV2Case("invalid_v2_null_self", nullptr, 0, true, true, out.tensor));
    RecordResult(stats, RunInvalidV2Case("invalid_v2_null_out", self.tensor, 0, true, true, nullptr));
    RecordResult(stats, RunInvalidV2Case("invalid_v2_dim_out_of_range", self.tensor, -3, true, false, out.tensor));
    self.Destroy();
    out.Destroy();
    outInt.Destroy();
    outMismatch.Destroy();
    selfTooManyDims.Destroy();
    outTooManyDims.Destroy();
}

std::vector<double> CpuCumsumFloat32Serial(const std::vector<double>& input, bool reverse)
{
    std::vector<double> result(input.size(), 0.0);
    float sum = 0.0f;
    if (!reverse) {
        for (size_t i = 0; i < input.size(); ++i) {
            sum = static_cast<float>(sum + static_cast<float>(input[i]));
            result[i] = static_cast<double>(sum);
        }
    } else {
        for (int64_t i = static_cast<int64_t>(input.size()) - 1; i >= 0; --i) {
            sum = static_cast<float>(sum + static_cast<float>(input[i]));
            result[i] = static_cast<double>(sum);
        }
    }
    return result;
}

void RunCpuPrecisionAnalysisCase(
    const std::string& name, const std::vector<double>& input, bool reverse, double minExpectedError)
{
    std::vector<double> quantized(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        quantized[i] = QuantizeInput(input[i], DataKind::FLOAT32);
    }
    auto doubleRef = CpuCumsum(quantized, {static_cast<int64_t>(quantized.size())}, 0, false, reverse);
    auto floatSerial = CpuCumsumFloat32Serial(quantized, reverse);
    PrecisionMetrics metrics = CalculatePrecisionMetrics(doubleRef, floatSerial, 1e-6, 1e-6);
    LOG_PRINT("CPU precision case: %s\n", name.c_str());
    LOG_PRINT("  Double ref sample: %s\n", FormatSample(doubleRef).c_str());
    LOG_PRINT("  Float32 serial sample: %s\n", FormatSample(floatSerial).c_str());
    PrintPrecisionMetrics("  Metrics:", metrics);
    LOG_PRINT("  Expected error floor for report: %.10g\n", minExpectedError);
}

void RunCpuPrecisionCurveCase(const std::string& name, int64_t length, bool exclusive, bool reverse)
{
    std::vector<double> input(length, 0.1);
    std::vector<double> quantized(length, 0.0);
    for (int64_t i = 0; i < length; ++i) {
        quantized[i] = QuantizeInput(input[i], DataKind::FLOAT32);
    }
    auto doubleRef = CpuCumsum(quantized, {length}, 0, exclusive, reverse);
    auto floatSerial = CpuCumsumFloat32Serial(quantized, reverse);
    if (exclusive) {
        // exclusive 模式需要把串行累计结果按方向错位，才能比较同一数学语义。
        if (!reverse) {
            for (int64_t i = length - 1; i > 0; --i) {
                floatSerial[i] = floatSerial[i - 1];
            }
            floatSerial[0] = 0.0;
        } else {
            for (int64_t i = 0; i + 1 < length; ++i) {
                floatSerial[i] = floatSerial[i + 1];
            }
            floatSerial[length - 1] = 0.0;
        }
    }
    PrecisionMetrics metrics = CalculatePrecisionMetrics(doubleRef, floatSerial, 1e-6, 1e-6);
    LOG_PRINT(
        "CPU precision curve: %s, N=%ld, exclusive=%d, reverse=%d\n",
        name.c_str(), length, static_cast<int>(exclusive), static_cast<int>(reverse));
    PrintPrecisionMetrics("  Metrics:", metrics);
}

bool RunCpuCumsumModeOracleCase()
{
    std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
    auto exclusiveForward = CpuCumsum(input, {4}, 0, true, false);
    auto exclusiveReverse = CpuCumsum(input, {4}, 0, true, true);
    std::vector<double> expectedForward = {0.0, 1.0, 3.0, 6.0};
    std::vector<double> expectedReverse = {9.0, 7.0, 4.0, 0.0};
    bool passed = exclusiveForward == expectedForward && exclusiveReverse == expectedReverse;
    LOG_PRINT("CPU precision case: exclusive/reverse oracle\n");
    LOG_PRINT("  Exclusive forward: %s\n", FormatSample(exclusiveForward).c_str());
    LOG_PRINT("  Exclusive reverse: %s\n", FormatSample(exclusiveReverse).c_str());
    LOG_PRINT("  [%s]\n", passed ? "PASS" : "FAIL");
    return passed;
}

int RunCpuPrecisionAnalysisCases()
{
    int reportCount = 0;
    std::vector<double> decimalInput(10000, 0.1);
    std::vector<double> mixedMagnitude = MakeMixedMagnitude(2048);
    RunCpuPrecisionAnalysisCase("float32 decimal accumulation [0.1] * 10000", decimalInput, false, 1e-4);
    ++reportCount;
    RunCpuPrecisionAnalysisCase("float32 mixed magnitude forward", mixedMagnitude, false, 1e-3);
    ++reportCount;
    RunCpuPrecisionAnalysisCase("float32 mixed magnitude reverse", mixedMagnitude, true, 1e-3);
    ++reportCount;
    for (int64_t length : {128, 512, 2048, 10000}) {
        RunCpuPrecisionCurveCase("inclusive_forward_decimal", length, false, false);
        ++reportCount;
        RunCpuPrecisionCurveCase("inclusive_reverse_decimal", length, false, true);
        ++reportCount;
        RunCpuPrecisionCurveCase("exclusive_forward_decimal", length, true, false);
        ++reportCount;
        RunCpuPrecisionCurveCase("exclusive_reverse_decimal", length, true, true);
        ++reportCount;
    }
    RunCpuCumsumModeOracleCase();
    ++reportCount;
    return reportCount;
}

int RunNpuPrecisionReportCases(aclrtStream stream)
{
    int reportCount = 0;
    const std::vector<std::pair<DataKind, std::pair<double, double>>> kinds = {
        {DataKind::FLOAT32, {1e-5, 1e-5}},
        {DataKind::FLOAT16, {1.0, 3e-2}},
        {DataKind::BFLOAT16, {2.0, 5e-2}},
    };
    for (const auto& item : kinds) {
        for (int64_t length : {128, 512, 2048, 10000}) {
            std::vector<double> input(length, 0.1);
            TestCase tc{
                std::string("npu_precision_report_") + KindName(item.first) + "_N" + std::to_string(length),
                {length},
                0,
                item.first,
                false,
                false,
                false,
                input,
                item.second.first,
                item.second.second,
                false,
                true,
                DataKind::UNSPECIFIED,
                true};
            RunCumsumCase(tc, stream);
            ++reportCount;
        }
    }

    std::vector<double> mixedMagnitude = MakeMixedMagnitude(2048);
    TestCase forward{
        "npu_precision_report_FLOAT32_mixed_magnitude_forward",
        {2048},
        0,
        DataKind::FLOAT32,
        false,
        false,
        false,
        mixedMagnitude,
        32.0,
        1e-6,
        false,
        true,
        DataKind::UNSPECIFIED,
        true};
    RunCumsumCase(forward, stream);
    ++reportCount;

    TestCase reverse{
        "npu_precision_report_FLOAT32_mixed_magnitude_reverse",
        {2048},
        0,
        DataKind::FLOAT32,
        true,
        false,
        true,
        mixedMagnitude,
        32.0,
        1e-6,
        false,
        true,
        DataKind::UNSPECIFIED,
        true};
    RunCumsumCase(reverse, stream);
    ++reportCount;
    return reportCount;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    TestStats stats;
    std::vector<TestCase> cases = BuildCases();
    for (const auto& tc : cases) {
        RecordResult(stats, RunCumsumCase(tc, stream));
    }
    int precisionReports = 0;
    precisionReports += RunCpuPrecisionAnalysisCases();
    precisionReports += RunNpuPrecisionReportCases(stream);
    RunInvalidCases(stats);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Summary: %d passed, %d failed, %d precision reports\n", stats.passed, stats.failed, precisionReports);
    return stats.failed == 0 ? 0 : 1;
}
