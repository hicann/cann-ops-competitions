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
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

namespace {

enum class DataKind {
    FLOAT32,
    FLOAT64,
    FLOAT16,
    BF16,
    INT32,
    INT64,
    INT8,
    UINT8,
    BOOL_AS_U8,
};

struct TestStats {
    int passed = 0;
    int failed = 0;
};

struct TensorResource {
    aclTensor* tensor = nullptr;
    void* deviceAddr = nullptr;
    size_t bytes = 0;
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

std::string ShapeToString(const std::vector<int64_t>& shape)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << shape[i];
    }
    os << "]";
    return os.str();
}

void ReleaseTensor(TensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
        resource.tensor = nullptr;
    }
    if (resource.deviceAddr != nullptr) {
        aclrtFree(resource.deviceAddr);
        resource.deviceAddr = nullptr;
    }
    resource.bytes = 0;
}

int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    TensorResource& resource)
{
    const int64_t elementCount = GetShapeSize(shape);
    if (elementCount < 0 || static_cast<size_t>(elementCount) != hostData.size()) {
        std::printf("host data size mismatch. shape=%s elements=%ld host=%zu\n", ShapeToString(shape).c_str(),
                    elementCount, hostData.size());
        return -1;
    }

    resource.bytes = hostData.size() * sizeof(T);
    if (resource.bytes > 0) {
        auto ret = aclrtMalloc(&resource.deviceAddr, resource.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMalloc failed. ERROR: %d bytes=%zu\n", ret, resource.bytes);
                  return ret);
        ret = aclrtMemcpy(resource.deviceAddr, resource.bytes, hostData.data(), resource.bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMemcpy H2D failed. ERROR: %d bytes=%zu\n", ret,
                                                  resource.bytes);
                  return ret);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }

    const int64_t* shapePtr = shape.empty() ? nullptr : shape.data();
    const int64_t* stridePtr = strides.empty() ? nullptr : strides.data();
    resource.tensor = aclCreateTensor(shapePtr, shape.size(), dataType, stridePtr, 0, aclFormat::ACL_FORMAT_ND,
                                      shapePtr, shape.size(), resource.deviceAddr);
    if (resource.tensor == nullptr) {
        std::printf("aclCreateTensor failed. dtype=%d shape=%s\n", static_cast<int>(dataType),
                    ShapeToString(shape).c_str());
        return -1;
    }
    return ACL_SUCCESS;
}

float BitsToFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t FloatToBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float Float16ToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
    if (exponent == 0) {
        if (mantissa == 0) {
            return BitsToFloat(sign);
        }
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        ++exponent;
        mantissa &= 0x03FFU;
    } else if (exponent == 31) {
        return BitsToFloat(sign | 0x7F800000U | (mantissa << 13));
    }
    exponent = exponent + (127 - 15);
    return BitsToFloat(sign | (exponent << 23) | (mantissa << 13));
}

uint16_t FloatToFloat16(float value)
{
    const uint32_t bits = FloatToBits(value);
    const uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    mantissa += 0x00001000U;
    if (mantissa & 0x00800000U) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = FloatToBits(value);
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16) & 1U);
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    return BitsToFloat(static_cast<uint32_t>(value) << 16);
}

double RawToDouble(float value, DataKind)
{
    return static_cast<double>(value);
}

double RawToDouble(double value, DataKind)
{
    return value;
}

double RawToDouble(int32_t value, DataKind)
{
    return static_cast<double>(value);
}

double RawToDouble(int64_t value, DataKind)
{
    return static_cast<double>(value);
}

double RawToDouble(int8_t value, DataKind)
{
    return static_cast<double>(value);
}

double RawToDouble(uint8_t value, DataKind)
{
    return static_cast<double>(value);
}

double RawToDouble(uint16_t value, DataKind kind)
{
    if (kind == DataKind::FLOAT16) {
        return static_cast<double>(Float16ToFloat(value));
    }
    if (kind == DataKind::BF16) {
        return static_cast<double>(BFloat16ToFloat(value));
    }
    return static_cast<double>(value);
}

template <typename T>
std::vector<double> ToDoubleVector(const std::vector<T>& input, DataKind kind)
{
    std::vector<double> values;
    values.reserve(input.size());
    for (const auto& item : input) {
        values.push_back(RawToDouble(item, kind));
    }
    return values;
}

std::vector<double> CpuCumsum(const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim,
                              bool exclusive, bool reverse)
{
    std::vector<double> expected(input.size(), 0.0);
    if (input.empty()) {
        return expected;
    }
    if (shape.empty()) {
        expected[0] = exclusive ? 0.0 : input[0];
        return expected;
    }

    const int64_t rank = static_cast<int64_t>(shape.size());
    int64_t axis = dim < 0 ? dim + rank : dim;
    const int64_t axisLen = shape[static_cast<size_t>(axis)];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= shape[static_cast<size_t>(i)];
    }
    const int64_t total = GetShapeSize(shape);
    const int64_t outer = axisLen == 0 ? 0 : total / axisLen / inner;

    for (int64_t outerIdx = 0; outerIdx < outer; ++outerIdx) {
        for (int64_t innerIdx = 0; innerIdx < inner; ++innerIdx) {
            double sum = 0.0;
            if (!reverse) {
                for (int64_t axisIdx = 0; axisIdx < axisLen; ++axisIdx) {
                    const int64_t idx = outerIdx * axisLen * inner + axisIdx * inner + innerIdx;
                    if (exclusive) {
                        expected[static_cast<size_t>(idx)] = sum;
                        sum += input[static_cast<size_t>(idx)];
                    } else {
                        sum += input[static_cast<size_t>(idx)];
                        expected[static_cast<size_t>(idx)] = sum;
                    }
                }
            } else {
                for (int64_t axisIdx = axisLen - 1; axisIdx >= 0; --axisIdx) {
                    const int64_t idx = outerIdx * axisLen * inner + axisIdx * inner + innerIdx;
                    if (exclusive) {
                        expected[static_cast<size_t>(idx)] = sum;
                        sum += input[static_cast<size_t>(idx)];
                    } else {
                        sum += input[static_cast<size_t>(idx)];
                        expected[static_cast<size_t>(idx)] = sum;
                    }
                }
            }
        }
    }
    return expected;
}

template <typename T>
std::vector<T> ZerosLike(size_t count)
{
    return std::vector<T>(count, static_cast<T>(0));
}

template <>
std::vector<uint16_t> ZerosLike<uint16_t>(size_t count)
{
    return std::vector<uint16_t>(count, 0);
}

bool CompareResult(const std::string& name, const std::vector<double>& actual, const std::vector<double>& expected,
                   double atol, double rtol, bool exact)
{
    if (actual.size() != expected.size()) {
        std::printf("  size mismatch actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    size_t maxIndex = 0;
    size_t mismatchCount = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const bool actualNan = std::isnan(actual[i]);
        const bool expectedNan = std::isnan(expected[i]);
        const bool actualInf = std::isinf(actual[i]);
        const bool expectedInf = std::isinf(expected[i]);
        double diff = 0.0;
        double rel = 0.0;
        double limit = exact ? 0.0 : (atol + rtol * std::fabs(expected[i]));
        bool matchedSpecialValue = false;
        if (actualNan || expectedNan || actualInf || expectedInf) {
            matchedSpecialValue = (actualNan && expectedNan) ||
                                  (actualInf && expectedInf && std::signbit(actual[i]) == std::signbit(expected[i]));
            diff = matchedSpecialValue ? 0.0 : std::numeric_limits<double>::infinity();
            rel = diff;
            limit = 0.0;
        } else {
            diff = std::fabs(actual[i] - expected[i]);
            rel = std::fabs(expected[i]) > 0.0 ? diff / std::fabs(expected[i]) : diff;
        }
        if (diff > maxAbsError) {
            maxAbsError = diff;
            maxRelError = rel;
            maxIndex = i;
        }
        if (diff > limit) {
            ++mismatchCount;
            if (mismatchCount <= 5) {
                std::printf("  mismatch[%zu]: actual=%.12g expected=%.12g diff=%.12g limit=%.12g\n", i, actual[i],
                            expected[i], diff, limit);
            }
        }
    }

    std::printf("  elements=%zu max_abs_error=%.12g max_rel_error=%.12g at=%zu actual=%.12g expected=%.12g\n",
                actual.size(), maxAbsError, maxRelError, maxIndex, actual.empty() ? 0.0 : actual[maxIndex],
                expected.empty() ? 0.0 : expected[maxIndex]);
    if (mismatchCount != 0) {
        std::printf("  mismatch_count=%zu\n", mismatchCount);
    }
    (void)name;
    return mismatchCount == 0;
}

bool RunExecutor(void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream, bool useV2)
{
    auto ret = useV2 ? aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream)
                     : aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::printf("  phase2 failed. ERROR: %d\n", ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::printf("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return false;
    }
    return true;
}

template <typename InT, typename OutT>
bool RunSuccessCase(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                    const std::vector<InT>& inputData, aclDataType inputType, aclDataType outType, DataKind inputKind,
                    DataKind outKind, bool useV2, bool exclusive, bool reverse, double atol, double rtol, bool exact,
                    aclrtStream stream)
{
    std::printf("Test case: %s shape=%s dim=%ld api=%s exclusive=%d reverse=%d\n", name.c_str(),
                ShapeToString(shape).c_str(), dim, useV2 ? "CumsumV2" : "Cumsum", exclusive ? 1 : 0,
                reverse ? 1 : 0);
    TensorResource self;
    TensorResource out;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool ok = false;

    const size_t elementCount = inputData.size();
    std::vector<OutT> outHostData = ZerosLike<OutT>(elementCount);

    auto ret = CreateAclTensor(inputData, shape, inputType, self);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }
    ret = CreateAclTensor(outHostData, shape, outType, out);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }

    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, dim, exclusive, reverse, out.tensor, &workspaceSize,
                                            &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self.tensor, dim, outType, out.tensor, &workspaceSize, &executor);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("  GetWorkspaceSize failed. ERROR: %d\n", ret);
        goto cleanup;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("  allocate workspace failed. ERROR: %d size=%lu\n", ret, workspaceSize);
            goto cleanup;
        }
    }

    if (!RunExecutor(workspaceAddr, workspaceSize, executor, stream, useV2)) {
        goto cleanup;
    }

    if (!outHostData.empty()) {
        ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(outHostData[0]), out.deviceAddr,
                          outHostData.size() * sizeof(outHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::printf("  copy result from device to host failed. ERROR: %d\n", ret);
            goto cleanup;
        }
    }

    {
        const std::vector<double> inputDouble = ToDoubleVector(inputData, inputKind);
        const std::vector<double> expected = CpuCumsum(inputDouble, shape, dim, exclusive, reverse);
        const std::vector<double> actual = ToDoubleVector(outHostData, outKind);
        ok = CompareResult(name, actual, expected, atol, rtol, exact);
    }

cleanup:
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    ReleaseTensor(self);
    ReleaseTensor(out);
    std::printf("  [%s]\n", ok ? "PASS" : "FAIL");
    return ok;
}

template <typename InT, typename OutT>
bool RunExpectedFailureCase(const std::string& name, const std::vector<int64_t>& selfShape,
                            const std::vector<int64_t>& outShape, int64_t dim, const std::vector<InT>& inputData,
                            aclDataType inputType, aclDataType outType, aclDataType dtypeArg, bool useV2,
                            bool exclusive, bool reverse, bool nullSelf, bool nullOut)
{
    std::printf("Test case: %s expected failure api=%s\n", name.c_str(), useV2 ? "CumsumV2" : "Cumsum");
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool ok = false;

    std::vector<OutT> outHostData = ZerosLike<OutT>(static_cast<size_t>(std::max<int64_t>(GetShapeSize(outShape), 0)));
    auto ret = CreateAclTensor(inputData, selfShape, inputType, self);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }
    ret = CreateAclTensor(outHostData, outShape, outType, out);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }

    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(nullSelf ? nullptr : self.tensor, dim, exclusive, reverse,
                                            nullOut ? nullptr : out.tensor, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(nullSelf ? nullptr : self.tensor, dim, dtypeArg,
                                          nullOut ? nullptr : out.tensor, &workspaceSize, &executor);
    }
    ok = (ret != ACL_SUCCESS);
    std::printf("  returned=%d expected_non_success=1\n", ret);

cleanup:
    ReleaseTensor(self);
    ReleaseTensor(out);
    std::printf("  [%s]\n", ok ? "PASS" : "FAIL");
    return ok;
}

template <typename InT, typename OutT>
bool RunWorkspaceOnlyCase(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                          const std::vector<InT>& inputData, aclDataType inputType, aclDataType outType,
                          aclDataType dtypeArg, bool useV2, bool exclusive, bool reverse)
{
    std::printf("Test case: %s workspace-only api=%s shape=%s\n", name.c_str(), useV2 ? "CumsumV2" : "Cumsum",
                ShapeToString(shape).c_str());
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool ok = false;

    std::vector<OutT> outHostData = ZerosLike<OutT>(inputData.size());
    auto ret = CreateAclTensor(inputData, shape, inputType, self);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }
    ret = CreateAclTensor(outHostData, shape, outType, out);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }

    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, dim, exclusive, reverse, out.tensor, &workspaceSize,
                                            &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self.tensor, dim, dtypeArg, out.tensor, &workspaceSize, &executor);
    }
    ok = (ret == ACL_SUCCESS && executor != nullptr);
    std::printf("  returned=%d workspace=%lu executor=%p\n", ret, workspaceSize, static_cast<void*>(executor));

cleanup:
    ReleaseTensor(self);
    ReleaseTensor(out);
    std::printf("  [%s]\n", ok ? "PASS" : "FAIL");
    return ok;
}

template <typename T>
void Record(TestStats& stats, bool passed)
{
    (void)sizeof(T);
    if (passed) {
        ++stats.passed;
    } else {
        ++stats.failed;
    }
}

void Record(TestStats& stats, bool passed)
{
    if (passed) {
        ++stats.passed;
    } else {
        ++stats.failed;
    }
}

bool RunInt32WorkspaceProbe(const std::string& name, const std::vector<int64_t>& shape, int64_t dim)
{
    return RunWorkspaceOnlyCase<int32_t, int32_t>(name, shape, dim,
                                                 std::vector<int32_t>(static_cast<size_t>(GetShapeSize(shape)), 0),
                                                 ACL_INT32, ACL_INT32, ACL_INT32, false, false, false);
}

bool RunFloatWorkspaceProbe(const std::string& name, const std::vector<int64_t>& shape, int64_t dim)
{
    return RunWorkspaceOnlyCase<float, float>(name, shape, dim,
                                             std::vector<float>(static_cast<size_t>(GetShapeSize(shape)), 0.0f),
                                             ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, false);
}

std::vector<float> MakeFloatData(size_t count, float base, float step)
{
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        const int pattern = static_cast<int>(i % 11);
        data[i] = base + step * static_cast<float>(pattern - 5);
    }
    return data;
}

std::vector<double> MakeDoubleData(size_t count)
{
    std::vector<double> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = (i % 2 == 0 ? 0.25 : -0.125) + static_cast<double>(i % 7) * 0.03125;
    }
    return data;
}

std::vector<uint16_t> MakeFp16Data(size_t count)
{
    std::vector<uint16_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        const float value = static_cast<float>((static_cast<int>(i % 9) - 4)) * 0.125f;
        data[i] = FloatToFloat16(value);
    }
    return data;
}

std::vector<uint16_t> MakeBf16Data(size_t count)
{
    std::vector<uint16_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        const float value = 0.03125f * static_cast<float>((static_cast<int>(i % 13) - 6));
        data[i] = FloatToBFloat16(value);
    }
    return data;
}

std::vector<int32_t> MakeInt32Data(size_t count)
{
    std::vector<int32_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<int32_t>(static_cast<int>(i % 7) - 3);
    }
    return data;
}

std::vector<int64_t> MakeInt64Data(size_t count)
{
    std::vector<int64_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<int64_t>(static_cast<int>(i % 5) - 2);
    }
    return data;
}

std::vector<int16_t> MakeInt16Data(size_t count)
{
    std::vector<int16_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<int16_t>(static_cast<int>(i % 9) - 4);
    }
    return data;
}

std::vector<int8_t> MakeInt8Data(size_t count)
{
    std::vector<int8_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<int8_t>(static_cast<int>(i % 3) - 1);
    }
    return data;
}

std::vector<uint8_t> MakeUint8Data(size_t count)
{
    std::vector<uint8_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<uint8_t>(i % 2);
    }
    return data;
}

std::vector<std::complex<float>> MakeComplex64Data(size_t count)
{
    std::vector<std::complex<float>> data(count);
    for (size_t i = 0; i < count; ++i) {
        const float real = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25f;
        const float imag = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
        data[i] = std::complex<float>(real, imag);
    }
    return data;
}

std::vector<std::complex<double>> MakeComplex128Data(size_t count)
{
    std::vector<std::complex<double>> data(count);
    for (size_t i = 0; i < count; ++i) {
        const double real = static_cast<double>(static_cast<int>(i % 5) - 2) * 0.25;
        const double imag = static_cast<double>(static_cast<int>(i % 7) - 3) * 0.125;
        data[i] = std::complex<double>(real, imag);
    }
    return data;
}

std::vector<float> MakeMixedMagnitudeData(size_t count)
{
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = (i % 2 == 0) ? 1.0e8f : 1.0e-3f;
    }
    return data;
}

std::vector<float> MakeAlternatingUnitData(size_t count)
{
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    return data;
}

std::vector<float> MakeLargeSmallCancelData(size_t count)
{
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        switch (i % 4) {
            case 0:
                data[i] = 1.0e8f;
                break;
            case 1:
                data[i] = 1.0f;
                break;
            case 2:
                data[i] = -1.0e8f;
                break;
            default:
                data[i] = 0.0f;
                break;
        }
    }
    return data;
}

std::vector<float> MakeLeadingLargeThenOnesData(size_t count, size_t axisLen)
{
    std::vector<float> data(count, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        if (axisLen != 0 && i % axisLen == 0) {
            data[i] = 1.0e8f;
        }
    }
    return data;
}

}  // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("Init acl failed. ERROR: %d\n", ret); return ret);

    TestStats stats;

    Record(stats, RunSuccessCase<float, float>(
                      "std_float_dim0_matrix", {2, 3}, 0, std::vector<float>(6, 0.0f), ACL_FLOAT, ACL_FLOAT,
                      DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "std_float_negative_last_dim", {3, 5}, -1, std::vector<float>(15, 0.0f), ACL_FLOAT,
                      ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false,
                      stream));
    Record(stats, RunSuccessCase<float, float>(
                      "v2_float_exclusive", {2, 4}, 1, std::vector<float>(8, 0.0f), ACL_FLOAT, ACL_FLOAT,
                      DataKind::FLOAT32, DataKind::FLOAT32, true, true, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "v2_float_reverse", {2, 4}, 1, std::vector<float>(8, 0.0f), ACL_FLOAT, ACL_FLOAT,
                      DataKind::FLOAT32, DataKind::FLOAT32, true, false, true, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "v2_float_exclusive_reverse_dim0", {4, 2}, 0, std::vector<float>(8, 0.0f), ACL_FLOAT,
                      ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, true, true, true, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>("std_empty_float", {0, 3}, 0, std::vector<float>(), ACL_FLOAT,
                                               ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false,
                                               1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>("v2_empty_float", {0, 3}, 0, std::vector<float>(), ACL_FLOAT,
                                               ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, true, false, false,
                                               1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>("std_scalar_float", {}, 0, std::vector<float>{0.0f}, ACL_FLOAT,
                                                ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false,
                                                1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<uint16_t, float>(
                      "std_fp16_input_to_float_cast", {2, 8}, 1, std::vector<uint16_t>(16, FloatToFloat16(0.0f)),
                      ACL_FLOAT16, ACL_FLOAT,
                      DataKind::FLOAT16, DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<uint16_t, uint16_t>(
                      "v2_fp16_twoway_reverse", {1, 512, 1}, 1,
                      std::vector<uint16_t>(512, FloatToFloat16(0.0f)), ACL_FLOAT16, ACL_FLOAT16,
                      DataKind::FLOAT16, DataKind::FLOAT16, true, false, true, 6e-2, 2e-2, false, stream));
    Record(stats, RunSuccessCase<uint16_t, uint16_t>(
                      "std_bf16_small", {2, 384, 2}, 1, std::vector<uint16_t>(1536, FloatToBFloat16(0.0f)),
                      ACL_BF16, ACL_BF16, DataKind::BF16,
                      DataKind::BF16, false, false, false, 2e-1, 2e-2, false, stream));
    Record(stats, RunWorkspaceOnlyCase<float, double>(
                      "workspace_float_to_double_aicpu_dispatch", {2, 6}, 1, std::vector<float>(12, 0.0f),
                      ACL_FLOAT, ACL_DOUBLE, ACL_DOUBLE, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<double, double>(
                      "workspace_double_aicpu_dispatch_v2", {3, 4}, 1, std::vector<double>(12, 0.0), ACL_DOUBLE,
                      ACL_DOUBLE, ACL_DOUBLE, true, true, false));
    Record(stats, RunWorkspaceOnlyCase<int16_t, int16_t>(
                      "workspace_int16_aicpu_dispatch", {2, 5}, 1, MakeInt16Data(10), ACL_INT16, ACL_INT16,
                      ACL_INT16, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<int64_t, int64_t>(
                      "workspace_int64_aicpu_dispatch", {2, 5}, 1, std::vector<int64_t>(10, 0), ACL_INT64,
                      ACL_INT64, ACL_INT64, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<int8_t, int8_t>(
                      "workspace_int8_aicpu_dispatch", {2, 5}, 1, std::vector<int8_t>(10, 0), ACL_INT8, ACL_INT8,
                      ACL_INT8, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<uint8_t, uint8_t>(
                      "workspace_uint8_aicpu_dispatch", {2, 5}, 1, std::vector<uint8_t>(10, 0), ACL_UINT8,
                      ACL_UINT8, ACL_UINT8, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<std::complex<float>, std::complex<float>>(
                      "workspace_complex64_aicpu_dispatch", {2, 5}, 1, MakeComplex64Data(10), ACL_COMPLEX64,
                      ACL_COMPLEX64, ACL_COMPLEX64, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<std::complex<double>, std::complex<double>>(
                      "workspace_complex128_aicpu_dispatch_v2", {2, 5}, 1, MakeComplex128Data(10), ACL_COMPLEX128,
                      ACL_COMPLEX128, ACL_COMPLEX128, true, false, true));

    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_mr_greater_cl", {64, 16, 1}, 1, std::vector<float>(1024, 0.0f), ACL_FLOAT,
                      ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false,
                      stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_two_way", {1, 512, 1}, 1, std::vector<float>(512, 0.0f), ACL_FLOAT,
                      ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false,
                      stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_ub_split", {64, 10000, 1}, 1, std::vector<float>(640000, 0.0f), ACL_FLOAT,
                      ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 2e-3, 1e-5, false,
                      stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_borrow_r_borrow", {1, 10000, 256}, 1,
                      std::vector<float>(2560000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32,
                      DataKind::FLOAT32, false, false, false, 2e-3, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_full_m_split", {64, 4, 128}, 1,
                      std::vector<float>(32768, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_full_borrow_n", {1, 4, 128}, 1,
                      std::vector<float>(512, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_full_n_ub_split", {64, 4, 50000}, 1,
                      std::vector<float>(12800000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32,
                      DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_notfull_m_split", {128, 512, 128}, 1,
                      std::vector<float>(8388608, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_notfull_m_core", {64, 512, 256}, 1,
                      std::vector<float>(8388608, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_notfull_m_core_wide", {64, 256, 1024}, 1,
                      std::vector<float>(16777216, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32,
                      DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_r_notfull_m_split", {40, 1024, 512}, 1,
                      std::vector<float>(20971520, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32,
                      DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_notfull_borrow_n", {4, 512, 4096}, 1,
                      std::vector<float>(8388608, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_r_notfull_borrow_n", {2, 1024, 4096}, 1,
                      std::vector<float>(8388608, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_notfull_borrow_n_wide", {4, 256, 8192}, 1,
                      std::vector<float>(8388608, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_borrow_r_full_small", {1, 4096, 256}, 1,
                      std::vector<float>(1048576, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_n_greater_borrow_r_full", {1, 512, 128}, 1,
                      std::vector<float>(65536, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_r_oway_full", {40, 64, 32}, 1,
                      std::vector<float>(81920, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_r_oway_notborrow_ub", {40, 4000, 32}, 1,
                      std::vector<float>(5120000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_r_twoway_notborrow_ub", {40, 10000, 1}, 1,
                      std::vector<float>(400000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_r_oway_borrow_r_ub", {1, 200000, 32}, 1,
                      std::vector<float>(6400000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_r_twoway_borrow_r_full", {1, 65536, 1}, 1,
                      std::vector<float>(65536, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_borrow_m_full_r", {128, 512, 1}, 1,
                      std::vector<float>(65536, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_borrow_m_core_ratio_96", {96, 512, 1}, 1,
                      std::vector<float>(49152, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_borrow_m_core_ratio_100", {100, 512, 1}, 1,
                      std::vector<float>(51200, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_r_notborrow_ub", {128, 10000, 1}, 1,
                      std::vector<float>(1280000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_borrow_r_twoway", {1, 200000, 1}, 1,
                      std::vector<float>(200000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_ub_ss_twoway", {128, 20000, 1}, 1,
                      std::vector<float>(2560000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_mrn_greater_m_split", {20000, 1, 1}, 1,
                      std::vector<float>(20000, 0.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "tiling_float_mrn_lesser", {1}, 0, std::vector<float>(1, 0.0f), ACL_FLOAT, ACL_FLOAT,
                      DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 1e-5, 1e-5, false, stream));
    Record(stats, RunSuccessCase<uint16_t, uint16_t>(
                      "fp16_core_ss_twoway", {1, 65536, 1}, 1,
                      std::vector<uint16_t>(65536, FloatToFloat16(0.0f)), ACL_FLOAT16, ACL_FLOAT16,
                      DataKind::FLOAT16, DataKind::FLOAT16, false, false, false, 6e-2, 2e-2, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "std_float_cube_route", {12800, 512}, 1, std::vector<float>(6553600, 1.0f), ACL_FLOAT,
                      ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 1e-4, 1e-5, false,
                      stream));
    Record(stats, RunSuccessCase<float, float>(
                      "std_float_cube_route_negative_dim", {12800, 512}, -1,
                      std::vector<float>(6553600, 1.0f), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 1e-4, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "precision_cube_float_0p1", {12800, 512}, 1, std::vector<float>(6553600, 0.1f), ACL_FLOAT,
                      ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 2e-2, 1e-5, false,
                      stream));
    Record(stats, RunSuccessCase<float, float>(
                      "precision_cube_float_0p1_len1024", {12800, 1024}, 1, std::vector<float>(13107200, 0.1f),
                      ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 5e-2, 1e-5,
                      false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "precision_cube_float_exact_pow2", {12800, 512}, 1, std::vector<float>(6553600, 0.125f),
                      ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 0.0, 0.0,
                      true, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "precision_cube_float_alternating_cancel", {12800, 512}, 1,
                      MakeAlternatingUnitData(6553600), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "precision_cube_float_large_small_cancel", {12800, 512}, 1,
                      MakeLargeSmallCancelData(6553600), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32, DataKind::FLOAT32,
                      false, false, false, 2.0, 1e-7, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "precision_cube_float_leading_large_increment_loss", {12800, 512}, 1,
                      MakeLeadingLargeThenOnesData(6553600, 512), ACL_FLOAT, ACL_FLOAT, DataKind::FLOAT32,
                      DataKind::FLOAT32, false, false, false, 6.0e2, 1e-5, false, stream));
    Record(stats, RunSuccessCase<float, float>(
                      "precision_cube_float_subnormal_accum", {12800, 512}, 1,
                      std::vector<float>(6553600, std::numeric_limits<float>::denorm_min()), ACL_FLOAT, ACL_FLOAT,
                      DataKind::FLOAT32, DataKind::FLOAT32, false, false, false, 1e-37, 0.0, false, stream));
    Record(stats, RunSuccessCase<uint16_t, uint16_t>(
                      "precision_cube_fp16_0p1", {12800, 512}, 1,
                      std::vector<uint16_t>(6553600, FloatToFloat16(0.1f)), ACL_FLOAT16, ACL_FLOAT16,
                      DataKind::FLOAT16, DataKind::FLOAT16, false, false, false, 6e-1, 2e-2, false, stream));
    Record(stats, RunSuccessCase<uint16_t, uint16_t>(
                      "precision_cube_fp16_exact_pow2", {12800, 512}, 1,
                      std::vector<uint16_t>(6553600, FloatToFloat16(0.125f)), ACL_FLOAT16, ACL_FLOAT16,
                      DataKind::FLOAT16, DataKind::FLOAT16, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<uint16_t, uint16_t>(
                      "precision_cube_bf16_0p1", {12800, 512}, 1,
                      std::vector<uint16_t>(6553600, FloatToBFloat16(0.1f)), ACL_BF16, ACL_BF16, DataKind::BF16,
                      DataKind::BF16, false, false, false, 1.0, 3e-2, false, stream));

    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_axis0", {8, 16}, 0, std::vector<int32_t>(128, 0), ACL_INT32, ACL_INT32, DataKind::INT32,
                      DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "v2_int32_middle_exclusive_reverse", {4, 16, 64}, 1, std::vector<int32_t>(4096, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, true, true, true, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_r_block_group", {1, 2048, 1}, 1, std::vector<int32_t>(2048, 0), ACL_INT32, ACL_INT32,
                      DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_negative_last_axis", {32, 128}, -1, std::vector<int32_t>(4096, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_left_axis_split", {256, 16, 1}, 1, std::vector<int32_t>(4096, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_right_axis_split", {1, 16, 4096}, 1, std::vector<int32_t>(65536, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_td_r_core_split", {1, 512, 128}, 1, std::vector<int32_t>(65536, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_axis0_right_split", {16, 4096}, 0, std::vector<int32_t>(65536, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_r_group_large", {1, 100000, 1}, 1, std::vector<int32_t>(100000, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_td_r_axis_split", {1, 2048, 4096}, 1, std::vector<int32_t>(8388608, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunSuccessCase<int32_t, int32_t>(
                      "int32_adjust_la_small_unit", {32, 256, 32}, 1, std::vector<int32_t>(262144, 0), ACL_INT32,
                      ACL_INT32, DataKind::INT32, DataKind::INT32, false, false, false, 0.0, 0.0, true, stream));
    Record(stats, RunWorkspaceOnlyCase<int32_t, int32_t>(
                      "workspace_int32_td_ra_r_split_probe32", {1, 10000, 32}, 1, std::vector<int32_t>(320000, 0),
                      ACL_INT32, ACL_INT32, ACL_INT32, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<int32_t, int32_t>(
                      "workspace_int32_td_ra_r_split_probe64", {1, 10000, 64}, 1, std::vector<int32_t>(640000, 0),
                      ACL_INT32, ACL_INT32, ACL_INT32, false, false, false));
    Record(stats, RunWorkspaceOnlyCase<int32_t, int32_t>(
                      "workspace_int32_adjust_la_unit_probe", {16, 10000, 32}, 1,
                      std::vector<int32_t>(5120000, 0), ACL_INT32, ACL_INT32, ACL_INT32, false, false, false));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left13_r2", {13, 512, 2}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left13_r4", {13, 256, 4}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left13_r8", {13, 192, 8}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left13_r16", {13, 512, 16}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left13_r32", {13, 1024, 32}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left32_r4", {32, 2048, 4}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left48_r8", {48, 2048, 8}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left64_r16", {64, 4096, 16}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left16_r32", {16, 8192, 32}, 1));
    Record(stats, RunInt32WorkspaceProbe("workspace_int32_bgc_left128_r24", {128, 1024, 24}, 1));
    Record(stats, RunFloatWorkspaceProbe("workspace_float_borrow_r_full_oway_probe", {1, 3000, 32}, 1));
    Record(stats, RunFloatWorkspaceProbe("workspace_float_judge_fold_count_one_probe", {1, 4, 4}, 1));
    Record(stats, RunFloatWorkspaceProbe("workspace_float_judge_fold_count_one_r16n8", {1, 16, 8}, 1));
    Record(stats, RunFloatWorkspaceProbe("workspace_float_judge_fold_count_one_r32n4", {1, 32, 4}, 1));
    Record(stats, RunFloatWorkspaceProbe("workspace_float_judge_fold_count_one_r8n16", {1, 8, 16}, 1));

    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_null_self", {2, 2}, {2, 2}, 0, MakeFloatData(4, 1.0f, 0.5f), ACL_FLOAT, ACL_FLOAT,
                      ACL_FLOAT, false, false, false, true, false));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_null_out", {2, 2}, {2, 2}, 0, MakeFloatData(4, 1.0f, 0.5f), ACL_FLOAT, ACL_FLOAT,
                      ACL_FLOAT, false, false, false, false, true));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_null_self_v2", {2, 2}, {2, 2}, 0, MakeFloatData(4, 1.0f, 0.5f), ACL_FLOAT,
                      ACL_FLOAT, ACL_FLOAT, true, false, false, true, false));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_null_out_v2", {2, 2}, {2, 2}, 0, MakeFloatData(4, 1.0f, 0.5f), ACL_FLOAT,
                      ACL_FLOAT, ACL_FLOAT, true, false, false, false, true));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_dim_out_of_range", {2, 2}, {2, 2}, 2, MakeFloatData(4, 1.0f, 0.5f), ACL_FLOAT,
                      ACL_FLOAT, ACL_FLOAT, false, false, false, false, false));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_dim_negative_out_of_range", {2, 2}, {2, 2}, -3, MakeFloatData(4, 1.0f, 0.5f),
                      ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, false, false, false));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_dim_negative_out_of_range_v2", {2, 2}, {2, 2}, -3,
                      MakeFloatData(4, 1.0f, 0.5f), ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, false, false,
                      false));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_dtype_arg_mismatch", {2, 2}, {2, 2}, 0, MakeFloatData(4, 1.0f, 0.5f), ACL_FLOAT,
                      ACL_FLOAT, ACL_DOUBLE, false, false, false, false, false));
    Record(stats, RunExpectedFailureCase<float, double>(
                      "invalid_v2_self_out_dtype_mismatch", {2, 2}, {2, 2}, 0, MakeFloatData(4, 1.0f, 0.5f),
                      ACL_FLOAT, ACL_DOUBLE, ACL_DOUBLE, true, false, false, false, false));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_shape_mismatch_v2", {2, 3}, {2, 2}, 1, MakeFloatData(6, 1.0f, 0.5f), ACL_FLOAT,
                      ACL_FLOAT, ACL_FLOAT, true, false, false, false, false));
    Record(stats, RunExpectedFailureCase<float, float>(
                      "invalid_rank_gt_8", {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 0,
                      MakeFloatData(1, 1.0f, 0.5f), ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, false, false,
                      false));
    Record(stats, RunExpectedFailureCase<uint16_t, uint16_t>(
                      "invalid_uint16_dtype", {2, 2}, {2, 2}, 0, std::vector<uint16_t>(4, 1), ACL_UINT16,
                      ACL_UINT16, ACL_UINT16, false, false, false, false, false));
    Record(stats, RunExpectedFailureCase<uint32_t, uint32_t>(
                      "invalid_uint32_dtype", {2, 2}, {2, 2}, 0, std::vector<uint32_t>(4, 1), ACL_UINT32,
                      ACL_UINT32, ACL_UINT32, false, false, false, false, false));
    Record(stats, RunExpectedFailureCase<uint64_t, uint64_t>(
                      "invalid_uint64_dtype", {2, 2}, {2, 2}, 0, std::vector<uint64_t>(4, 1), ACL_UINT64,
                      ACL_UINT64, ACL_UINT64, false, false, false, false, false));
    Record(stats, RunExpectedFailureCase<uint8_t, uint8_t>(
                      "invalid_bool_dtype", {2, 2}, {2, 2}, 0, MakeUint8Data(4), ACL_BOOL, ACL_BOOL, ACL_BOOL,
                      false, false, false, false, false));
    Record(stats, RunExpectedFailureCase<uint16_t, uint16_t>(
                      "invalid_v2_uint16_dtype", {2, 2}, {2, 2}, 0, std::vector<uint16_t>(4, 1), ACL_UINT16,
                      ACL_UINT16, ACL_UINT16, true, false, false, false, false));
    Record(stats, RunExpectedFailureCase<uint32_t, uint32_t>(
                      "invalid_v2_uint32_dtype", {2, 2}, {2, 2}, 0, std::vector<uint32_t>(4, 1), ACL_UINT32,
                      ACL_UINT32, ACL_UINT32, true, false, false, false, false));
    Record(stats, RunExpectedFailureCase<uint8_t, float>(
                      "invalid_v2_self_bool_unsupported", {2, 2}, {2, 2}, 0, MakeUint8Data(4), ACL_BOOL,
                      ACL_FLOAT, ACL_FLOAT, true, false, false, false, false));
    Record(stats, RunExpectedFailureCase<float, uint8_t>(
                      "invalid_v2_out_bool_unsupported", {2, 2}, {2, 2}, 0, MakeFloatData(4, 1.0f, 0.5f),
                      ACL_FLOAT, ACL_BOOL, ACL_BOOL, true, false, false, false, false));

    std::printf("Summary: %d passed, %d failed\n", stats.passed, stats.failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return stats.failed == 0 ? 0 : 1;
}
