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
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "../op_api/aclnn_cumsum.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

namespace {

struct TestStats {
    int passed = 0;
    int failed = 0;
};

struct RunConfig {
    std::string name;
    std::vector<int64_t> shape;
    int64_t dim = 0;
    bool useV2 = false;
    bool exclusive = false;
    bool reverse = false;
    double atol = 0.0;
    double rtol = 0.0;
};

struct TensorHandle {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
    size_t bytes = 0;

    void Reset()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
            deviceAddr = nullptr;
        }
        bytes = 0;
    }

    ~TensorHandle()
    {
        Reset();
    }
};

struct WorkspaceHandle {
    void *addr = nullptr;

    void Reset()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }

    ~WorkspaceHandle()
    {
        Reset();
    }
};

constexpr int32_t kDeviceId = 0;

void Log(const std::string &msg)
{
    std::cout << msg << std::endl;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
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

std::string ShapeToString(const std::vector<int64_t> &shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string StatusToString(aclnnStatus status)
{
    std::ostringstream oss;
    oss << static_cast<int>(status);
    return oss.str();
}

uint16_t FloatToBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits & 0xFFFF);
}

uint32_t FloatToUInt32(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float UInt32ToFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = FloatToUInt32(value);
    uint32_t roundingBias = ((bits >> 16) & 1U) + 0x7FFFU;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    return UInt32ToFloat(bits);
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = FloatToUInt32(value);
    uint32_t sign = (bits >> 16) & 0x8000U;
    uint32_t mantissa = bits & 0x007FFFFFU;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000U) >> (1 - exp);
        if (mantissa & 0x00001000U) {
            mantissa += 0x00002000U;
        }
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (mantissa & 0x00001000U) {
        mantissa += 0x00002000U;
        if (mantissa & 0x00800000U) {
            mantissa = 0;
            ++exp;
            if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mantissa >> 13));
}

float HalfToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    int32_t exp = static_cast<int32_t>((value >> 10) & 0x1FU);
    uint32_t mantissa = value & 0x03FFU;

    if (exp == 0) {
        if (mantissa == 0) {
            return UInt32ToFloat(sign);
        }
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1;
            --exp;
        }
        ++exp;
        mantissa &= ~0x0400U;
    } else if (exp == 31) {
        return UInt32ToFloat(sign | 0x7F800000U | (mantissa << 13));
    }

    exp = exp + (127 - 15);
    uint32_t bits = sign | (static_cast<uint32_t>(exp) << 23) | (mantissa << 13);
    return UInt32ToFloat(bits);
}

template <typename T>
bool CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                     TensorHandle &handle)
{
    auto elemCount = GetShapeSize(shape);
    auto size = static_cast<size_t>(elemCount) * sizeof(T);
    handle.bytes = size;

    if (size > 0) {
        auto ret = aclrtMalloc(&handle.deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            Log("aclrtMalloc failed. ERROR: " + std::to_string(ret));
            return false;
        }

        ret = aclrtMemcpy(handle.deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            Log("aclrtMemcpy host to device failed. ERROR: " + std::to_string(ret));
            return false;
        }
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
    }

    handle.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), handle.deviceAddr);
    if (handle.tensor == nullptr) {
        Log("aclCreateTensor failed.");
        return false;
    }
    return true;
}

template <typename T>
bool CopyDeviceToHost(const TensorHandle &handle, std::vector<T> &hostData)
{
    if (hostData.empty()) {
        return true;
    }
    auto copyBytes = hostData.size() * sizeof(T);
    auto ret = aclrtMemcpy(hostData.data(), copyBytes, handle.deviceAddr, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        Log("aclrtMemcpy device to host failed. ERROR: " + std::to_string(ret));
        return false;
    }
    return true;
}

int NormalizeDim(int64_t dim, size_t rank)
{
    int64_t normalized = dim;
    if (normalized < 0) {
        normalized += static_cast<int64_t>(rank);
    }
    return static_cast<int>(normalized);
}

template <typename T>
std::vector<long double> CpuCumsum(const std::vector<T> &input, const std::vector<int64_t> &shape, int64_t dim,
                                   bool exclusive, bool reverse)
{
    auto elemCount = static_cast<size_t>(GetShapeSize(shape));
    std::vector<long double> output(elemCount, 0.0L);
    auto rank = shape.size();
    int axis = NormalizeDim(dim, rank);

    int64_t outer = 1;
    int64_t inner = 1;
    int64_t axisLen = shape[static_cast<size_t>(axis)];
    for (int i = 0; i < axis; ++i) {
        outer *= shape[static_cast<size_t>(i)];
    }
    for (size_t i = static_cast<size_t>(axis) + 1; i < rank; ++i) {
        inner *= shape[i];
    }

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            long double sum = 0.0L;
            if (!reverse) {
                for (int64_t a = 0; a < axisLen; ++a) {
                    int64_t idx = (o * axisLen + a) * inner + in;
                    long double val = static_cast<long double>(input[static_cast<size_t>(idx)]);
                    if (exclusive) {
                        output[static_cast<size_t>(idx)] = sum;
                        sum += val;
                    } else {
                        sum += val;
                        output[static_cast<size_t>(idx)] = sum;
                    }
                }
            } else {
                for (int64_t a = axisLen - 1; a >= 0; --a) {
                    int64_t idx = (o * axisLen + a) * inner + in;
                    long double val = static_cast<long double>(input[static_cast<size_t>(idx)]);
                    if (exclusive) {
                        output[static_cast<size_t>(idx)] = sum;
                        sum += val;
                    } else {
                        sum += val;
                        output[static_cast<size_t>(idx)] = sum;
                    }
                }
            }
        }
    }
    return output;
}

template <typename T>
std::string Preview(const std::vector<T> &values, size_t limit = 8)
{
    std::ostringstream oss;
    oss << "[";
    auto actualLimit = std::min(limit, values.size());
    for (size_t i = 0; i < actualLimit; ++i) {
        if (i != 0) {
            oss << ", ";
        }
        if constexpr (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
            oss << static_cast<int>(values[i]);
        } else {
            oss << values[i];
        }
    }
    if (values.size() > actualLimit) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

std::string PreviewFloatLike(const std::vector<float> &values, size_t limit = 8)
{
    std::ostringstream oss;
    oss << "[";
    auto actualLimit = std::min(limit, values.size());
    for (size_t i = 0; i < actualLimit; ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    if (values.size() > actualLimit) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

template <typename T>
bool CompareOutputs(const std::vector<T> &actual, const std::vector<long double> &expected, double atol, double rtol,
                    bool exact, double &maxError, size_t &maxIndex, std::string &detail)
{
    maxError = 0.0;
    maxIndex = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        auto actualValue = static_cast<long double>(actual[i]);
        auto expectedValue = expected[i];
        auto error = std::fabs(static_cast<double>(actualValue - expectedValue));
        if (error > maxError) {
            maxError = error;
            maxIndex = i;
        }
        if (exact) {
            if (actualValue != expectedValue) {
                std::ostringstream oss;
                oss << "Exact mismatch at index " << i << ", expected=" << static_cast<long long>(expectedValue)
                    << ", actual=" << static_cast<long long>(actualValue);
                detail = oss.str();
                return false;
            }
        } else {
            auto tol = atol + rtol * std::fabs(static_cast<double>(expectedValue));
            if (error > tol) {
                std::ostringstream oss;
                oss << std::setprecision(10) << "Tolerance mismatch at index " << i << ", expected="
                    << static_cast<double>(expectedValue) << ", actual=" << static_cast<double>(actualValue)
                    << ", error=" << error << ", tol=" << tol;
                detail = oss.str();
                return false;
            }
        }
    }
    detail = "Max error=" + std::to_string(maxError) + " at index " + std::to_string(maxIndex);
    return true;
}

bool InitAcl(aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = aclrtSetDevice(kDeviceId);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    return true;
}

void FinalizeAcl(aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(kDeviceId);
    aclFinalize();
}

template <typename T>
bool RunStandardCase(const RunConfig &config, const std::vector<T> &input, aclDataType selfType, aclDataType outType,
                     std::vector<T> &actual, std::string &detail, aclrtStream stream)
{
    TensorHandle self;
    TensorHandle out;
    WorkspaceHandle workspace;
    auto outElemCount = static_cast<size_t>(GetShapeSize(config.shape));
    std::vector<T> zero(outElemCount, static_cast<T>(0));
    if (!CreateAclTensor(input, config.shape, selfType, self)) {
        detail = "Create self tensor failed.";
        return false;
    }
    if (!CreateAclTensor(zero, config.shape, outType, out)) {
        detail = "Create out tensor failed.";
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self.tensor, config.dim, outType, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        detail = "aclnnCumsumGetWorkspaceSize failed, ret=" + StatusToString(ret);
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            detail = "Workspace alloc failed, ret=" + std::to_string(ret);
            return false;
        }
    }

    ret = aclnnCumsum(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        detail = "aclnnCumsum failed, ret=" + StatusToString(ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        detail = "aclrtSynchronizeStream failed, ret=" + std::to_string(ret);
        return false;
    }

    actual.assign(outElemCount, static_cast<T>(0));
    if (!CopyDeviceToHost(out, actual)) {
        detail = "Copy output failed.";
        return false;
    }
    return true;
}

template <typename T>
bool RunV2Case(const RunConfig &config, const std::vector<T> &input, aclDataType tensorType, std::vector<T> &actual,
               std::string &detail, aclrtStream stream)
{
    TensorHandle self;
    TensorHandle out;
    WorkspaceHandle workspace;
    auto outElemCount = static_cast<size_t>(GetShapeSize(config.shape));
    std::vector<T> zero(outElemCount, static_cast<T>(0));
    if (!CreateAclTensor(input, config.shape, tensorType, self)) {
        detail = "Create self tensor failed.";
        return false;
    }
    if (!CreateAclTensor(zero, config.shape, tensorType, out)) {
        detail = "Create out tensor failed.";
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self.tensor, config.dim, config.exclusive, config.reverse, out.tensor,
                                             &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        detail = "aclnnCumsumV2GetWorkspaceSize failed, ret=" + StatusToString(ret);
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            detail = "Workspace alloc failed, ret=" + std::to_string(ret);
            return false;
        }
    }

    ret = aclnnCumsumV2(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        detail = "aclnnCumsumV2 failed, ret=" + StatusToString(ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        detail = "aclrtSynchronizeStream failed, ret=" + std::to_string(ret);
        return false;
    }

    actual.assign(outElemCount, static_cast<T>(0));
    if (!CopyDeviceToHost(out, actual)) {
        detail = "Copy output failed.";
        return false;
    }
    return true;
}

template <typename T>
bool RunExactOrToleranceCase(const RunConfig &config, const std::vector<T> &input, aclDataType tensorType, bool exact,
                             TestStats &stats, aclrtStream stream)
{
    std::cout << "Case: " << config.name << "\n"
              << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim
              << ", api=" << (config.useV2 ? "CumsumV2" : "Cumsum") << std::endl;

    std::vector<T> actual;
    std::string detail;
    bool ok = false;
    if (config.useV2) {
        ok = RunV2Case(config, input, tensorType, actual, detail, stream);
    } else {
        ok = RunStandardCase(config, input, tensorType, tensorType, actual, detail, stream);
    }

    if (!ok) {
        std::cout << "  [FAIL] " << detail << std::endl;
        ++stats.failed;
        return false;
    }

    auto expected = CpuCumsum(input, config.shape, config.dim, config.exclusive, config.reverse);
    double maxError = 0.0;
    size_t maxIndex = 0;
    std::string cmpDetail;
    ok = CompareOutputs(actual, expected, config.atol, config.rtol, exact, maxError, maxIndex, cmpDetail);

    std::cout << "  input preview   : " << Preview(input) << "\n"
              << "  actual preview  : " << Preview(actual) << "\n"
              << "  result          : " << (ok ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;

    if (ok) {
        ++stats.passed;
    } else {
        ++stats.failed;
    }
    return ok;
}

template <typename T>
std::vector<T> MakePatternData(size_t size, const std::function<T(size_t)> &fn)
{
    std::vector<T> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = fn(i);
    }
    return data;
}

bool ExpectStandardApiFailure(const std::string &name, const aclTensor *self, int64_t dim, aclDataType dtype,
                              const aclTensor *out, TestStats &stats)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, const_cast<aclTensor *>(out), &workspaceSize, &executor);
    bool pass = ret != ACL_SUCCESS;
    std::cout << "Case: " << name << "\n"
              << "  expected failure, ret=" << static_cast<int>(ret) << "\n"
              << "  result          : " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    if (pass) {
        ++stats.passed;
    } else {
        ++stats.failed;
    }
    return pass;
}

bool ExpectV2ApiFailure(const std::string &name, const aclTensor *self, int64_t dim, bool exclusive, bool reverse,
                        aclTensor *out, TestStats &stats)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    bool pass = ret != ACL_SUCCESS;
    std::cout << "Case: " << name << "\n"
              << "  expected failure, ret=" << static_cast<int>(ret) << "\n"
              << "  result          : " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    if (pass) {
        ++stats.passed;
    } else {
        ++stats.failed;
    }
    return pass;
}

bool RunInterfaceValidationCases(TestStats &stats)
{
    std::cout << "\n=== Interface Validation Layer ===" << std::endl;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> floatData = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> floatOutData(6, 0.f);
    std::vector<int32_t> intOutData(6, 0);

    TensorHandle selfFloat;
    TensorHandle outFloat;
    TensorHandle outFloatBadShape;
    TensorHandle outInt;
    std::vector<int64_t> badShape = {3, 2};

    if (!CreateAclTensor(floatData, shape, ACL_FLOAT, selfFloat)) {
        return false;
    }
    if (!CreateAclTensor(floatOutData, shape, ACL_FLOAT, outFloat)) {
        return false;
    }
    if (!CreateAclTensor(floatOutData, badShape, ACL_FLOAT, outFloatBadShape)) {
        return false;
    }
    if (!CreateAclTensor(intOutData, shape, ACL_INT32, outInt)) {
        return false;
    }

    bool ok = true;
    ok &= ExpectStandardApiFailure("Null self pointer", nullptr, 0, ACL_FLOAT, outFloat.tensor, stats);
    ok &= ExpectStandardApiFailure("Null out pointer", selfFloat.tensor, 1, ACL_FLOAT, nullptr, stats);
    ok &= ExpectStandardApiFailure("Invalid dim out of range", selfFloat.tensor, 3, ACL_FLOAT, outFloat.tensor, stats);
    ok &= ExpectStandardApiFailure("Output dtype mismatch with dtype argument", selfFloat.tensor, 1, ACL_INT32,
                                   outFloat.tensor, stats);
    ok &= ExpectStandardApiFailure("Shape mismatch between self and out", selfFloat.tensor, 1, ACL_FLOAT,
                                   outFloatBadShape.tensor, stats);
    ok &= ExpectV2ApiFailure("V2 requires self/out same dtype", selfFloat.tensor, 1, false, false, outInt.tensor,
                             stats);
    ok &= ExpectV2ApiFailure("V2 invalid dim out of range", selfFloat.tensor, 3, false, false, outFloat.tensor, stats);
    return ok;
}

bool RunExecutionCases(TestStats &stats, aclrtStream stream)
{
    std::cout << "\n=== Execution Validation Layer ===" << std::endl;

    bool ok = true;

    {
        RunConfig config = {"float32_basic_dim0", {2, 3}, 0, false, false, false, 1e-5, 1e-5};
        auto input = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_empty_tensor", {0, 4}, 1, false, false, false, 1e-5, 1e-5};
        auto input = std::vector<float>{};
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_negative_dim_mixed_sign", {2, 3, 4}, -2, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(24, [](size_t i) {
            return (i % 2 == 0) ? static_cast<float>(i + 1) * 0.5f : -static_cast<float>(i + 1) * 0.25f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_mrnlessercl_case", {1, 2, 2}, 1, false, false, false, 1e-5, 1e-5};
        auto input = std::vector<float>{1.0f, -1.0f, 0.5f, 2.0f};
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_mrngreatercl_case", {1024, 2, 2}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 11) - 5)) * 0.25f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_twoway_candidate", {1, 1024, 1}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_borrowr_candidate", {2, 2048, 1}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 15) - 7)) * 0.0625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_borrowm_candidate", {128, 1024, 1}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 27) - 13)) * 0.03125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_oneway_fullub_smallr", {128, 128, 7}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 33) - 16)) * 0.03125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_twoway_fullub", {8, 1024, 7}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 35) - 17)) * 0.03125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_rfullload_oneway", {128, 512, 15}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.0625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_twoway_notfull_notborrowr", {64, 4096, 7}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 37) - 18)) * 0.015625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_borrowr_twoway_fullub", {2, 4096, 7}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 39) - 19)) * 0.015625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_borrowr_twoway_notfull", {2, 65536, 7}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 41) - 20)) * 0.015625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_rnotfull_notborrowr", {64, 8192, 15}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 21) - 10)) * 0.03125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_rnotfull_notborrowr_large_r", {32, 32768, 15}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 53) - 26)) * 0.0078125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_rngreatercl_borrowm_divisible_candidate", {512, 1024, 1}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 55) - 27)) * 0.015625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rfullload_n_notfull", {128, 1024, 256}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 25) - 12)) * 0.0625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rfullload_n_notfull_alt", {64, 256, 1024}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 43) - 21)) * 0.03125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rnotfull_borrown", {2, 8192, 2048}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 31) - 15)) * 0.015625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rnotfull_borrown_alt", {3, 4096, 512}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 45) - 22)) * 0.015625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rnotfull_borrowr", {1, 8192, 16}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 19) - 9)) * 0.0625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rnotfull_borrowr_alt", {1, 4096, 128}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 47) - 23)) * 0.03125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rfullload_mbig", {128, 8, 1024}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.03125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rfullload_msmall", {2, 8, 2048}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.0625f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_ngreatercl_rnotfull", {64, 4096, 64}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>((static_cast<int>(i % 19) - 9)) * 0.125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_large_cube_candidate", {12800, 512}, 1, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"float32_long_sequence_point_one", {10000}, 0, false, false, false, 1e-5, 1e-5};
        auto input = std::vector<float>(10000, 0.1f);
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"int32_exact_negative_dim", {4, 5}, -1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(20, [](size_t i) { return static_cast<int32_t>(static_cast<int>(i % 7) - 3); });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_axis0_large_rightaxis", {7, 4096}, 0, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 13) - 6);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_middle_axis_large_rightaxis", {9, 257, 1024}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 9) - 4);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_group_rblock_candidate", {2, 8192, 1}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 5) - 2);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_tdla_candidate", {512, 257, 4}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 3) - 1);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_tdra_candidate", {4, 512, 4096}, 1, false, false, false, 0.0, 0.0};
        auto input = std::vector<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), 0);
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_group_rblock_strong_candidate", {1, 65536, 1}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<int32_t>((i % 3) + 1);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_tdra_split_r_candidate", {1, 8192, 4096}, 1, false, false, false, 0.0, 0.0};
        auto input = std::vector<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), 1);
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int32_tdla_large_left_candidate", {2048, 257, 4}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 5) - 2);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"int64_exact_cpu_path_candidate", {3, 4, 5}, 2, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int64_t>(60, [](size_t i) { return static_cast<int64_t>(static_cast<int>(i % 11) - 5); });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT64, true, stats, stream);
    }

    {
        RunConfig config = {"uint8_exact_unsigned", {2, 8}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<uint8_t>(16, [](size_t i) { return static_cast<uint8_t>((i % 5) + 1); });
        ok &= RunExactOrToleranceCase(config, input, ACL_UINT8, true, stats, stream);
    }

    {
        RunConfig config = {"uint8_axis0_large_rightaxis", {9, 4096}, 0, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<uint8_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<uint8_t>((i % 13) + 1);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_UINT8, true, stats, stream);
    }

    {
        RunConfig config = {"uint8_middle_axis_large_rightaxis", {5, 257, 1024}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<uint8_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<uint8_t>((i % 7) + 1);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_UINT8, true, stats, stream);
    }

    {
        RunConfig config = {"v2_float32_exclusive", {2, 4}, 1, true, true, false, 1e-5, 1e-5};
        auto input = std::vector<float>{1.f, 2.f, 3.f, 4.f, -1.f, -2.f, 0.5f, 1.5f};
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"v2_float32_empty_tensor", {0, 8}, 1, true, false, false, 1e-5, 1e-5};
        auto input = std::vector<float>{};
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"v2_float32_reverse", {2, 4}, 1, true, false, true, 1e-5, 1e-5};
        auto input = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f};
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"v2_float32_exclusive_reverse_mixed_magnitude", {2, 4}, 1, true, true, true, 1e-5, 1e-5};
        auto input = std::vector<float>{1e8f, 1e-3f, 1e8f, 1e-3f, -1e4f, 2.f, -1e4f, 2.f};
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    {
        RunConfig config = {"v2_int64_reverse_aicpu_candidate", {3, 5, 7}, 1, true, false, true, 0.0, 0.0};
        auto input = MakePatternData<int64_t>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<int64_t>(static_cast<int>(i % 15) - 7);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT64, true, stats, stream);
    }

    {
        RunConfig config = {"float16_execution_case", {2, 8}, 1, false, false, false, 1e-3, 1e-3};
        std::vector<float> inputFloat = {0.5f, -0.25f, 1.5f, 2.0f, -3.0f, 0.125f, 0.5f, 1.0f,
                                         -1.0f, 2.0f, -2.0f, 4.0f, 8.0f, -0.5f, 0.25f, 0.75f};
        std::vector<uint16_t> inputRaw(inputFloat.size());
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputRaw[i] = FloatToHalf(inputFloat[i]);
        }

        std::cout << "Case: " << config.name << "\n"
                  << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim << ", api=Cumsum" << std::endl;
        std::vector<uint16_t> actualRaw;
        std::string detail;
        if (!RunStandardCase(config, inputRaw, ACL_FLOAT16, ACL_FLOAT16, actualRaw, detail, stream)) {
            std::cout << "  [FAIL] " << detail << std::endl;
            ++stats.failed;
            ok = false;
        } else {
            std::vector<float> actualFloat(actualRaw.size());
            for (size_t i = 0; i < actualRaw.size(); ++i) {
                actualFloat[i] = HalfToFloat(actualRaw[i]);
            }
            auto expected = CpuCumsum(inputFloat, config.shape, config.dim, false, false);
            double maxError = 0.0;
            size_t maxIndex = 0;
            std::string cmpDetail;
            bool caseOk = CompareOutputs(actualFloat, expected, config.atol, config.rtol, false, maxError, maxIndex, cmpDetail);
            std::cout << "  input preview   : " << PreviewFloatLike(inputFloat) << "\n"
                      << "  actual preview  : " << PreviewFloatLike(actualFloat) << "\n"
                      << "  result          : " << (caseOk ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;
            if (caseOk) {
                ++stats.passed;
            } else {
                ++stats.failed;
                ok = false;
            }
        }
    }

    {
        RunConfig config = {"float16_ngreatercl_case", {16, 64, 256}, 1, false, false, false, 1e-3, 1e-3};
        std::vector<float> inputFloat(static_cast<size_t>(GetShapeSize(config.shape)));
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputFloat[i] = static_cast<float>((static_cast<int>(i % 21) - 10)) * 0.125f;
        }
        std::vector<uint16_t> inputRaw(inputFloat.size());
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputRaw[i] = FloatToHalf(inputFloat[i]);
        }

        std::cout << "Case: " << config.name << "\n"
                  << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim << ", api=Cumsum" << std::endl;
        std::vector<uint16_t> actualRaw;
        std::string detail;
        if (!RunStandardCase(config, inputRaw, ACL_FLOAT16, ACL_FLOAT16, actualRaw, detail, stream)) {
            std::cout << "  [FAIL] " << detail << std::endl;
            ++stats.failed;
            ok = false;
        } else {
            std::vector<float> actualFloat(actualRaw.size());
            for (size_t i = 0; i < actualRaw.size(); ++i) {
                actualFloat[i] = HalfToFloat(actualRaw[i]);
            }
            auto expected = CpuCumsum(inputFloat, config.shape, config.dim, false, false);
            double maxError = 0.0;
            size_t maxIndex = 0;
            std::string cmpDetail;
            bool caseOk = CompareOutputs(actualFloat, expected, config.atol, config.rtol, false, maxError, maxIndex, cmpDetail);
            std::cout << "  input preview   : " << PreviewFloatLike(inputFloat) << "\n"
                      << "  actual preview  : " << PreviewFloatLike(actualFloat) << "\n"
                      << "  result          : " << (caseOk ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;
            if (caseOk) {
                ++stats.passed;
            } else {
                ++stats.failed;
                ok = false;
            }
        }
    }

    {
        RunConfig config = {"float16_rngreatercl_twoway_candidate", {1, 1024, 1}, 1, false, false, false, 1e-3, 1e-3};
        std::vector<float> inputFloat(static_cast<size_t>(GetShapeSize(config.shape)));
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputFloat[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.125f;
        }
        std::vector<uint16_t> inputRaw(inputFloat.size());
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputRaw[i] = FloatToHalf(inputFloat[i]);
        }

        std::cout << "Case: " << config.name << "\n"
                  << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim << ", api=Cumsum" << std::endl;
        std::vector<uint16_t> actualRaw;
        std::string detail;
        if (!RunStandardCase(config, inputRaw, ACL_FLOAT16, ACL_FLOAT16, actualRaw, detail, stream)) {
            std::cout << "  [FAIL] " << detail << std::endl;
            ++stats.failed;
            ok = false;
        } else {
            std::vector<float> actualFloat(actualRaw.size());
            for (size_t i = 0; i < actualRaw.size(); ++i) {
                actualFloat[i] = HalfToFloat(actualRaw[i]);
            }
            auto expected = CpuCumsum(inputFloat, config.shape, config.dim, false, false);
            double maxError = 0.0;
            size_t maxIndex = 0;
            std::string cmpDetail;
            bool caseOk = CompareOutputs(actualFloat, expected, config.atol, config.rtol, false, maxError, maxIndex, cmpDetail);
            std::cout << "  input preview   : " << PreviewFloatLike(inputFloat) << "\n"
                      << "  actual preview  : " << PreviewFloatLike(actualFloat) << "\n"
                      << "  result          : " << (caseOk ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;
            if (caseOk) {
                ++stats.passed;
            } else {
                ++stats.failed;
                ok = false;
            }
        }
    }

    {
        RunConfig config = {"float16_rngreatercl_borrowr_twoway_candidate", {2, 4096, 1}, 1, false, false, false, 1e-3, 1e-3};
        std::vector<float> inputFloat(static_cast<size_t>(GetShapeSize(config.shape)));
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputFloat[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.0625f;
        }
        std::vector<uint16_t> inputRaw(inputFloat.size());
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputRaw[i] = FloatToHalf(inputFloat[i]);
        }

        std::cout << "Case: " << config.name << "\n"
                  << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim << ", api=Cumsum" << std::endl;
        std::vector<uint16_t> actualRaw;
        std::string detail;
        if (!RunStandardCase(config, inputRaw, ACL_FLOAT16, ACL_FLOAT16, actualRaw, detail, stream)) {
            std::cout << "  [FAIL] " << detail << std::endl;
            ++stats.failed;
            ok = false;
        } else {
            std::vector<float> actualFloat(actualRaw.size());
            for (size_t i = 0; i < actualRaw.size(); ++i) {
                actualFloat[i] = HalfToFloat(actualRaw[i]);
            }
            auto expected = CpuCumsum(inputFloat, config.shape, config.dim, false, false);
            double maxError = 0.0;
            size_t maxIndex = 0;
            std::string cmpDetail;
            bool caseOk = CompareOutputs(actualFloat, expected, config.atol, config.rtol, false, maxError, maxIndex, cmpDetail);
            std::cout << "  input preview   : " << PreviewFloatLike(inputFloat) << "\n"
                      << "  actual preview  : " << PreviewFloatLike(actualFloat) << "\n"
                      << "  result          : " << (caseOk ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;
            if (caseOk) {
                ++stats.passed;
            } else {
                ++stats.failed;
                ok = false;
            }
        }
    }

    {
        RunConfig config = {"bf16_execution_case", {2, 8}, -1, false, false, false, 1e-3, 1e-3};
        std::vector<float> inputFloat = {1.0f, 0.1f, -0.1f, 2.0f, -4.0f, 8.0f, -16.0f, 0.5f,
                                         3.0f, -3.0f, 6.0f, -6.0f, 12.0f, -12.0f, 24.0f, -24.0f};
        std::vector<uint16_t> inputRaw(inputFloat.size());
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputRaw[i] = FloatToBFloat16(inputFloat[i]);
        }

        std::cout << "Case: " << config.name << "\n"
                  << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim << ", api=Cumsum" << std::endl;
        std::vector<uint16_t> actualRaw;
        std::string detail;
        if (!RunStandardCase(config, inputRaw, ACL_BF16, ACL_BF16, actualRaw, detail, stream)) {
            std::cout << "  [FAIL] " << detail << std::endl;
            ++stats.failed;
            ok = false;
        } else {
            std::vector<float> actualFloat(actualRaw.size());
            for (size_t i = 0; i < actualRaw.size(); ++i) {
                actualFloat[i] = BFloat16ToFloat(actualRaw[i]);
            }
            auto expected = CpuCumsum(inputFloat, config.shape, config.dim, false, false);
            double maxError = 0.0;
            size_t maxIndex = 0;
            std::string cmpDetail;
            bool caseOk = CompareOutputs(actualFloat, expected, config.atol, config.rtol, false, maxError, maxIndex, cmpDetail);
            std::cout << "  input preview   : " << PreviewFloatLike(inputFloat) << "\n"
                      << "  actual preview  : " << PreviewFloatLike(actualFloat) << "\n"
                      << "  result          : " << (caseOk ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;
            if (caseOk) {
                ++stats.passed;
            } else {
                ++stats.failed;
                ok = false;
            }
        }
    }

    {
        RunConfig config = {"bf16_ngreatercl_case", {8, 128, 256}, 1, false, false, false, 1e-3, 1e-3};
        std::vector<float> inputFloat(static_cast<size_t>(GetShapeSize(config.shape)));
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputFloat[i] = static_cast<float>((static_cast<int>(i % 31) - 15)) * 0.0625f;
        }
        std::vector<uint16_t> inputRaw(inputFloat.size());
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputRaw[i] = FloatToBFloat16(inputFloat[i]);
        }

        std::cout << "Case: " << config.name << "\n"
                  << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim << ", api=Cumsum" << std::endl;
        std::vector<uint16_t> actualRaw;
        std::string detail;
        if (!RunStandardCase(config, inputRaw, ACL_BF16, ACL_BF16, actualRaw, detail, stream)) {
            std::cout << "  [FAIL] " << detail << std::endl;
            ++stats.failed;
            ok = false;
        } else {
            std::vector<float> actualFloat(actualRaw.size());
            for (size_t i = 0; i < actualRaw.size(); ++i) {
                actualFloat[i] = BFloat16ToFloat(actualRaw[i]);
            }
            auto expected = CpuCumsum(inputFloat, config.shape, config.dim, false, false);
            double maxError = 0.0;
            size_t maxIndex = 0;
            std::string cmpDetail;
            bool caseOk = CompareOutputs(actualFloat, expected, config.atol, config.rtol, false, maxError, maxIndex, cmpDetail);
            std::cout << "  input preview   : " << PreviewFloatLike(inputFloat) << "\n"
                      << "  actual preview  : " << PreviewFloatLike(actualFloat) << "\n"
                      << "  result          : " << (caseOk ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;
            if (caseOk) {
                ++stats.passed;
            } else {
                ++stats.failed;
                ok = false;
            }
        }
    }

    {
        RunConfig config = {"bf16_rngreatercl_borrowr_candidate", {2, 2048, 1}, 1, false, false, false, 1e-3, 1e-3};
        std::vector<float> inputFloat(static_cast<size_t>(GetShapeSize(config.shape)));
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputFloat[i] = static_cast<float>((static_cast<int>(i % 25) - 12)) * 0.0625f;
        }
        std::vector<uint16_t> inputRaw(inputFloat.size());
        for (size_t i = 0; i < inputFloat.size(); ++i) {
            inputRaw[i] = FloatToBFloat16(inputFloat[i]);
        }

        std::cout << "Case: " << config.name << "\n"
                  << "  shape=" << ShapeToString(config.shape) << ", dim=" << config.dim << ", api=Cumsum" << std::endl;
        std::vector<uint16_t> actualRaw;
        std::string detail;
        if (!RunStandardCase(config, inputRaw, ACL_BF16, ACL_BF16, actualRaw, detail, stream)) {
            std::cout << "  [FAIL] " << detail << std::endl;
            ++stats.failed;
            ok = false;
        } else {
            std::vector<float> actualFloat(actualRaw.size());
            for (size_t i = 0; i < actualRaw.size(); ++i) {
                actualFloat[i] = BFloat16ToFloat(actualRaw[i]);
            }
            auto expected = CpuCumsum(inputFloat, config.shape, config.dim, false, false);
            double maxError = 0.0;
            size_t maxIndex = 0;
            std::string cmpDetail;
            bool caseOk = CompareOutputs(actualFloat, expected, config.atol, config.rtol, false, maxError, maxIndex, cmpDetail);
            std::cout << "  input preview   : " << PreviewFloatLike(inputFloat) << "\n"
                      << "  actual preview  : " << PreviewFloatLike(actualFloat) << "\n"
                      << "  result          : " << (caseOk ? "[PASS]" : "[FAIL]") << " " << cmpDetail << std::endl;
            if (caseOk) {
                ++stats.passed;
            } else {
                ++stats.failed;
                ok = false;
            }
        }
    }


    {
        RunConfig config = {"int8_exact_basic", {2, 8}, 1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int8_t>(16, [](size_t i) {
            return static_cast<int8_t>(static_cast<int>(i % 7) - 3);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT8, true, stats, stream);
    }

    {
        RunConfig config = {"int8_negative_dim", {3, 4, 5}, -1, false, false, false, 0.0, 0.0};
        auto input = MakePatternData<int8_t>(60, [](size_t i) {
            return static_cast<int8_t>(static_cast<int>(i % 5) - 2);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT8, true, stats, stream);
    }

    {
        RunConfig config = {"v2_int32_exclusive", {3, 6}, 1, true, true, false, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(18, [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 9) - 4);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"v2_int32_reverse", {3, 6}, 1, true, false, true, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(18, [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 11) - 5);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"v2_int32_exclusive_reverse", {3, 6}, 1, true, true, true, 0.0, 0.0};
        auto input = MakePatternData<int32_t>(18, [](size_t i) {
            return static_cast<int32_t>(static_cast<int>(i % 13) - 6);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT32, true, stats, stream);
    }

    {
        RunConfig config = {"v2_uint8_reverse", {2, 16}, -1, true, false, true, 0.0, 0.0};
        auto input = MakePatternData<uint8_t>(32, [](size_t i) {
            return static_cast<uint8_t>((i % 5) + 1);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_UINT8, true, stats, stream);
    }

    {
        RunConfig config = {"v2_int8_exclusive_reverse", {2, 16}, 1, true, true, true, 0.0, 0.0};
        auto input = MakePatternData<int8_t>(32, [](size_t i) {
            return static_cast<int8_t>(static_cast<int>(i % 7) - 3);
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_INT8, true, stats, stream);
    }

    {
        RunConfig config = {"float32_dim0_wide", {17, 33}, 0, false, false, false, 1e-5, 1e-5};
        auto input = MakePatternData<float>(static_cast<size_t>(GetShapeSize(config.shape)), [](size_t i) {
            return static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        });
        ok &= RunExactOrToleranceCase(config, input, ACL_FLOAT, false, stats, stream);
    }

    return ok;
}

} // namespace

int main()
{
    aclrtStream stream = nullptr;
    if (!InitAcl(&stream)) {
        std::cout << "ACL init failed." << std::endl;
        return 1;
    }

    TestStats stats;
    bool ok = true;
    ok &= RunInterfaceValidationCases(stats);
    ok &= RunExecutionCases(stats, stream);

    std::cout << "\nSummary: " << stats.passed << " passed, " << stats.failed << " failed" << std::endl;

    FinalizeAcl(stream);
    return (ok && stats.failed == 0) ? 0 : 1;
}
