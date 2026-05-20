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
#include <functional>
#include <limits>
#include <string>
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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

struct TensorHolder {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;

    TensorHolder() = default;
    TensorHolder(const TensorHolder&) = delete;
    TensorHolder& operator=(const TensorHolder&) = delete;

    ~TensorHolder()
    {
        Reset();
    }

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
    }
};

struct CaseResult {
    std::string name;
    bool pass;
    std::string detail;
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape,
                    aclDataType dataType,
                    TensorHolder* holder,
                    aclFormat format = ACL_FORMAT_ND)
{
    CHECK_RET(holder != nullptr, return -1);
    holder->Reset();

    const int64_t elemCount = GetShapeSize(shape);
    const int64_t bytes = elemCount * static_cast<int64_t>(sizeof(T));
    auto strides = GetContiguousStrides(shape);

    if (bytes > 0) {
        auto ret = aclrtMalloc(&holder->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(holder->deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
    }

    holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                                     shape.size(), holder->deviceAddr);
    CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
    return ACL_SUCCESS;
}

template <typename T>
int CopyTensorToHost(const TensorHolder& holder, std::vector<T>* hostData)
{
    CHECK_RET(hostData != nullptr, return -1);
    const int64_t bytes = static_cast<int64_t>(hostData->size() * sizeof(T));
    if (bytes == 0) {
        return ACL_SUCCESS;
    }
    CHECK_RET(holder.deviceAddr != nullptr, return -1);
    auto ret = aclrtMemcpy(hostData->data(), bytes, holder.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

int RunWithWorkspace(const std::function<aclnnStatus(uint64_t*, aclOpExecutor**)>& getWorkspace,
                     const std::function<aclnnStatus(void*, uint64_t, aclOpExecutor*, aclrtStream)>& run,
                     aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = getWorkspace(&workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = run(workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

int RunCumsum(const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out, aclrtStream stream)
{
    return RunWithWorkspace(
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, workspaceSize, executor);
        },
        [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnCumsum(workspace, workspaceSize, executor, runStream);
        },
        stream);
}

int RunCumsumV2(const aclTensor* self, int64_t dim, bool exclusive, bool reverse, aclTensor* out, aclrtStream stream)
{
    return RunWithWorkspace(
        [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, workspaceSize, executor);
        },
        [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
            return aclnnCumsumV2(workspace, workspaceSize, executor, runStream);
        },
        stream);
}

static inline uint32_t FloatToBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline float BitsToFloat(uint32_t bits)
{
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline uint16_t FloatToFp16Bits(float value)
{
    uint32_t bits = FloatToBits(value);
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | static_cast<uint16_t>(mantissa >> 13));
}

static inline float Fp16BitsToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>((value >> 15) & 0x1U);
    uint32_t exponent = static_cast<uint32_t>((value >> 10) & 0x1FU);
    uint32_t mantissa = static_cast<uint32_t>(value & 0x3FFU);

    if (exponent == 0) {
        float denorm = static_cast<float>(mantissa) * 0.000000059604644775390625F;
        return sign ? -denorm : denorm;
    }
    if (exponent == 31) {
        if (mantissa == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    float normal =
        (1.0F + static_cast<float>(mantissa) * 0.0009765625F) * std::pow(2.0F, static_cast<float>(exponent) - 15.0F);
    return sign ? -normal : normal;
}

static inline uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = FloatToBits(value);
    uint32_t roundingBias = ((bits >> 16) & 1U) + 0x7FFFU;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

static inline float Bf16BitsToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    return BitsToFloat(bits);
}

bool IsClose(double actual, double expected, double atol, double rtol)
{
    return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

std::string FormatDoubleValue(double value)
{
    char buffer[64] = {};
    std::snprintf(buffer, sizeof(buffer), "%.9g", value);
    return std::string(buffer);
}

std::string FormatDoubleSamples(const std::vector<double>& values)
{
    const size_t sampleCount = 3;
    const size_t size = values.size();
    std::string text = "[";
    if (size == 0) {
        text += "]";
        return text;
    }

    size_t firstCount = size < sampleCount ? size : sampleCount;
    for (size_t i = 0; i < firstCount; ++i) {
        if (i > 0) {
            text += ", ";
        }
        text += FormatDoubleValue(values[i]);
    }

    size_t secondStart = firstCount;
    if (size > sampleCount * 2) {
        text += ", ...";
        secondStart = size - sampleCount;
    }
    for (size_t i = secondStart; i < size; ++i) {
        text += ", ";
        text += FormatDoubleValue(values[i]);
    }
    text += "]";
    return text;
}

std::string FormatActualExpectedSamples(const std::vector<double>& actual, const std::vector<double>& expected)
{
    return " actual_samples=" + FormatDoubleSamples(actual) + " expected_samples=" + FormatDoubleSamples(expected);
}

bool CheckDoubleVector(const std::vector<double>& actual,
                       const std::vector<double>& expected,
                       double atol,
                       double rtol,
                       std::string* err)
{
    if (actual.size() != expected.size()) {
        if (err != nullptr) {
            *err = "size mismatch";
        }
        return false;
    }
    double maxErr = 0.0;
    size_t maxIdx = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(actual[i] - expected[i]);
        if (diff > maxErr) {
            maxErr = diff;
            maxIdx = i;
        }
        if (!IsClose(actual[i], expected[i], atol, rtol)) {
            if (err != nullptr) {
                *err = "index=" + std::to_string(i) + " actual=" + std::to_string(actual[i]) +
                       " expected=" + std::to_string(expected[i]) + " max_error=" + std::to_string(maxErr) +
                       " max_index=" + std::to_string(maxIdx) + " max_actual=" + FormatDoubleValue(actual[maxIdx]) +
                       " max_expected=" + FormatDoubleValue(expected[maxIdx]) +
                       FormatActualExpectedSamples(actual, expected);
            }
            return false;
        }
    }
    if (err != nullptr) {
        *err = "max_error=" + std::to_string(maxErr) + " max_index=" + std::to_string(maxIdx) +
               " max_actual=" + FormatDoubleValue(actual.empty() ? 0.0 : actual[maxIdx]) +
               " max_expected=" + FormatDoubleValue(expected.empty() ? 0.0 : expected[maxIdx]) +
               FormatActualExpectedSamples(actual, expected);
    }
    return true;
}

bool CheckFloatResult(const std::vector<float>& actual,
                      const std::vector<double>& expected,
                      double atol,
                      double rtol,
                      std::string* err)
{
    std::vector<double> actualDouble(actual.size(), 0.0);
    for (size_t i = 0; i < actual.size(); ++i) {
        actualDouble[i] = static_cast<double>(actual[i]);
    }
    return CheckDoubleVector(actualDouble, expected, atol, rtol, err);
}

bool CheckDoubleResult(const std::vector<double>& actual,
                       const std::vector<double>& expected,
                       double atol,
                       double rtol,
                       std::string* err)
{
    return CheckDoubleVector(actual, expected, atol, rtol, err);
}

bool CheckFp16Result(const std::vector<uint16_t>& actual,
                     const std::vector<double>& expected,
                     double atol,
                     double rtol,
                     std::string* err)
{
    std::vector<double> actualDouble(actual.size(), 0.0);
    for (size_t i = 0; i < actual.size(); ++i) {
        actualDouble[i] = static_cast<double>(Fp16BitsToFloat(actual[i]));
    }
    return CheckDoubleVector(actualDouble, expected, atol, rtol, err);
}

bool CheckBf16Result(const std::vector<uint16_t>& actual,
                     const std::vector<double>& expected,
                     double atol,
                     double rtol,
                     std::string* err)
{
    std::vector<double> actualDouble(actual.size(), 0.0);
    for (size_t i = 0; i < actual.size(); ++i) {
        actualDouble[i] = static_cast<double>(Bf16BitsToFloat(actual[i]));
    }
    return CheckDoubleVector(actualDouble, expected, atol, rtol, err);
}

template <typename T>
bool CheckExactResult(const std::vector<T>& actual, const std::vector<double>& expected, std::string* err)
{
    if (actual.size() != expected.size()) {
        if (err != nullptr) {
            *err = "size mismatch";
        }
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        int64_t a = static_cast<int64_t>(actual[i]);
        int64_t e = static_cast<int64_t>(std::llround(expected[i]));
        if (a != e) {
            if (err != nullptr) {
                std::vector<double> actualDouble(actual.size(), 0.0);
                for (size_t j = 0; j < actual.size(); ++j) {
                    actualDouble[j] = static_cast<double>(actual[j]);
                }
                *err = "index=" + std::to_string(i) + " actual=" + std::to_string(a) +
                       " expected=" + std::to_string(e) + FormatActualExpectedSamples(actualDouble, expected);
            }
            return false;
        }
    }
    return true;
}

int64_t NormalizeDim(const std::vector<int64_t>& shape, int64_t dim)
{
    int64_t rank = static_cast<int64_t>(shape.size());
    if (rank == 0) {
        return 0;
    }
    return dim < 0 ? dim + rank : dim;
}

std::vector<double> ComputeCumsum(const std::vector<double>& input,
                                  const std::vector<int64_t>& shape,
                                  int64_t dim,
                                  bool exclusive = false,
                                  bool reverse = false)
{
    std::vector<double> expected(input.size(), 0.0);
    if (input.empty()) {
        return expected;
    }
    if (shape.empty()) {
        expected[0] = exclusive ? 0.0 : input[0];
        return expected;
    }

    int64_t axis = NormalizeDim(shape, dim);
    int64_t left = 1;
    int64_t right = 1;
    for (int64_t i = 0; i < axis; ++i) {
        left *= shape[static_cast<size_t>(i)];
    }
    for (size_t i = static_cast<size_t>(axis + 1); i < shape.size(); ++i) {
        right *= shape[i];
    }
    const int64_t axisLen = shape[static_cast<size_t>(axis)];

    for (int64_t l = 0; l < left; ++l) {
        for (int64_t r = 0; r < right; ++r) {
            double sum = 0.0;
            if (!reverse) {
                for (int64_t k = 0; k < axisLen; ++k) {
                    const int64_t offset = (l * axisLen + k) * right + r;
                    if (exclusive) {
                        expected[static_cast<size_t>(offset)] = sum;
                        sum += input[static_cast<size_t>(offset)];
                    } else {
                        sum += input[static_cast<size_t>(offset)];
                        expected[static_cast<size_t>(offset)] = sum;
                    }
                }
            } else {
                for (int64_t k = axisLen - 1; k >= 0; --k) {
                    const int64_t offset = (l * axisLen + k) * right + r;
                    if (exclusive) {
                        expected[static_cast<size_t>(offset)] = sum;
                        sum += input[static_cast<size_t>(offset)];
                    } else {
                        sum += input[static_cast<size_t>(offset)];
                        expected[static_cast<size_t>(offset)] = sum;
                    }
                }
            }
        }
    }
    return expected;
}

template <typename T>
std::vector<double> ToDoubleVector(const std::vector<T>& input)
{
    std::vector<double> out(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = static_cast<double>(input[i]);
    }
    return out;
}

std::vector<double> Fp16ToDoubleVector(const std::vector<uint16_t>& input)
{
    std::vector<double> out(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = static_cast<double>(Fp16BitsToFloat(input[i]));
    }
    return out;
}

std::vector<double> Bf16ToDoubleVector(const std::vector<uint16_t>& input)
{
    std::vector<double> out(input.size(), 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = static_cast<double>(Bf16BitsToFloat(input[i]));
    }
    return out;
}

template <typename InT, typename OutT>
CaseResult RunCumsumCase(const std::string& name,
                         aclrtStream stream,
                         const std::vector<int64_t>& shape,
                         int64_t dim,
                         const std::vector<InT>& input,
                         aclDataType inputType,
                         aclDataType outType,
                         aclDataType dtype,
                         const std::function<std::vector<double>(const std::vector<InT>&)>& toDouble,
                         const std::function<bool(const std::vector<OutT>&, const std::vector<double>&, std::string*)>& check)
{
    CaseResult result{name, false, ""};
    TensorHolder self;
    TensorHolder out;
    std::vector<OutT> outHost(static_cast<size_t>(GetShapeSize(shape)), static_cast<OutT>(0));

    int ret = CreateAclTensor(input, shape, inputType, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, outType, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    ret = RunCumsum(self.tensor, dim, dtype, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "RunCumsum failed, ret=" + std::to_string(ret);
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected = ComputeCumsum(toDouble(input), shape, dim);
    std::string err;
    result.pass = check(outHost, expected, &err);
    if (!result.pass) {
        result.detail = "mismatch: " + err;
    } else {
        result.detail = err;
    }
    return result;
}

template <typename T>
CaseResult RunCumsumV2Case(
    const std::string& name,
    aclrtStream stream,
    const std::vector<int64_t>& shape,
    int64_t dim,
    bool exclusive,
    bool reverse,
    const std::vector<T>& input,
    aclDataType dataType,
    const std::function<std::vector<double>(const std::vector<T>&)>& toDouble,
    const std::function<bool(const std::vector<T>&, const std::vector<double>&, std::string*)>& check)
{
    CaseResult result{name, false, ""};
    TensorHolder self;
    TensorHolder out;
    std::vector<T> outHost(static_cast<size_t>(GetShapeSize(shape)), static_cast<T>(0));

    int ret = CreateAclTensor(input, shape, dataType, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, dataType, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    ret = RunCumsumV2(self.tensor, dim, exclusive, reverse, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "RunCumsumV2 failed, ret=" + std::to_string(ret);
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected = ComputeCumsum(toDouble(input), shape, dim, exclusive, reverse);
    std::string err;
    result.pass = check(outHost, expected, &err);
    result.detail = result.pass ? err : ("mismatch: " + err);
    return result;
}

CaseResult RunNegativeStatusCase(const std::string& name,
                                 const std::function<aclnnStatus(uint64_t*, aclOpExecutor**)>& getWorkspace)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = getWorkspace(&workspaceSize, &executor);
    CaseResult result{name, status != ACL_SUCCESS, "status=" + std::to_string(static_cast<int64_t>(status))};
    if (status == ACL_SUCCESS) {
        result.detail += " (expected failure)";
    }
    return result;
}

void PrintCaseResult(const CaseResult& result)
{
    std::string line = result.name;
    if (line.size() < 48) {
        line.append(48 - line.size(), ' ');
    } else {
        line.push_back(' ');
    }
    line += result.pass ? "[PASS]" : "[FAIL]";
    if (!result.detail.empty()) {
        line += "  " + result.detail;
    }
    LOG_PRINT("%s\n", line.c_str());
}

std::vector<float> MakePatternFloat(size_t n, float start, float step)
{
    std::vector<float> data(n, 0.0F);
    for (size_t i = 0; i < n; ++i) {
        int selector = static_cast<int>(i % 7);
        data[i] = start + step * static_cast<float>(selector - 3);
    }
    return data;
}

std::vector<float> MakeLargeSmallPattern(size_t n)
{
    std::vector<float> data(n, 0.0F);
    for (size_t i = 0; i < n; ++i) {
        data[i] = (i % 2 == 0) ? 1000000.0F : 0.25F;
    }
    return data;
}

std::vector<uint16_t> MakeFp16Vector(const std::vector<float>& input)
{
    std::vector<uint16_t> data(input.size(), 0);
    for (size_t i = 0; i < input.size(); ++i) {
        data[i] = FloatToFp16Bits(input[i]);
    }
    return data;
}

std::vector<uint16_t> MakeBf16Vector(const std::vector<float>& input)
{
    std::vector<uint16_t> data(input.size(), 0);
    for (size_t i = 0; i < input.size(); ++i) {
        data[i] = FloatToBf16Bits(input[i]);
    }
    return data;
}

CaseResult CaseFloatBasicDim0(aclrtStream stream)
{
    std::vector<float> input = {1.0F, -2.0F, 3.5F, 4.0F, 0.5F, -1.0F};
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_Dim0_Mixed", stream, {2, 3}, 0, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseFloatNegativeDim(aclrtStream stream)
{
    std::vector<float> input = {1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F};
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_NegativeDim", stream, {2, 3}, -1, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseFloatLongPointOne(aclrtStream stream)
{
    std::vector<float> input(10000, 0.1F);
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_Long_0p1_ErrorAccum", stream, {10000}, 0, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 2e-3, 2e-6, err);
        });
}

CaseResult CaseFloatMixedMagnitude(aclrtStream stream)
{
    std::vector<float> input = MakeLargeSmallPattern(2048);
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_MixedMagnitude", stream, {2048}, 0, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 5e-1, 1e-6, err);
        });
}

CaseResult CaseFloat3DMiddleAxis(aclrtStream stream)
{
    std::vector<float> input = MakePatternFloat(65 * 17 * 2, 0.5F, 0.25F);
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_3D_MiddleAxis", stream, {65, 17, 2}, 1, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseFloatLargeNPath(aclrtStream stream)
{
    std::vector<float> input = MakePatternFloat(4 * 1024 * 16, 1.0F, 0.125F);
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_LargeN_Tiling", stream, {4, 1024, 16}, 1, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 2e-4, 1e-5, err);
        });
}

CaseResult CaseFloatCubeCandidate(aclrtStream stream)
{
    std::vector<float> input(12800 * 512, 1.0F);
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_CubeCandidate_LastDim", stream, {12800, 512}, 1, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-4, 1e-5, err);
        });
}

CaseResult CaseDoubleAiCpu(aclrtStream stream)
{
    std::vector<double> input = {1.25, -0.25, 2.0, 4.0, -8.0, 16.0};
    return RunCumsumCase<double, double>(
        "Cumsum_Double_AiCpuProbe", stream, {2, 3}, 1, input, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE,
        ToDoubleVector<double>,
        [](const std::vector<double>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckDoubleResult(actual, expected, 1e-10, 1e-10, err);
        });
}

CaseResult CaseFp16Basic(aclrtStream stream)
{
    std::vector<uint16_t> input = MakeFp16Vector({0.5F, 1.0F, -0.25F, 2.0F, -1.0F, 0.25F, 1.5F, -0.5F});
    return RunCumsumCase<uint16_t, uint16_t>(
        "Cumsum_Float16_Basic", stream, {2, 4}, 1, input, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        Fp16ToDoubleVector,
        [](const std::vector<uint16_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFp16Result(actual, expected, 2e-2, 2e-3, err);
        });
}

CaseResult CaseFp16CastFromFloat(aclrtStream stream)
{
    std::vector<float> input = {0.5F, 1.0F, 1.5F, 2.0F, -0.5F, -1.0F};
    return RunCumsumCase<float, uint16_t>(
        "Cumsum_Float32_To_Float16_Cast", stream, {2, 3}, 1, input, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT16,
        ToDoubleVector<float>,
        [](const std::vector<uint16_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFp16Result(actual, expected, 2e-2, 2e-3, err);
        });
}

CaseResult CaseBf16Optional(aclrtStream stream)
{
    std::vector<uint16_t> input = MakeBf16Vector({1.0F, 0.5F, -0.25F, 2.0F});
    return RunCumsumCase<uint16_t, uint16_t>(
        "Cumsum_BFloat16_Basic", stream, {4}, 0, input, ACL_BF16, ACL_BF16, ACL_BF16,
        Bf16ToDoubleVector,
        [](const std::vector<uint16_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckBf16Result(actual, expected, 5e-2, 5e-3, err);
        });
}

CaseResult CaseInt32MiddleAxis(aclrtStream stream)
{
    std::vector<int32_t> input = {1, -2, 3, 4, 5, -6, 7, 8, 9, -1, -2, -3};
    return RunCumsumCase<int32_t, int32_t>(
        "Cumsum_Int32_3D_MiddleAxis", stream, {2, 3, 2}, 1, input, ACL_INT32, ACL_INT32, ACL_INT32,
        ToDoubleVector<int32_t>,
        [](const std::vector<int32_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckExactResult(actual, expected, err);
        });
}

CaseResult CaseInt64AiCpu(aclrtStream stream)
{
    std::vector<int64_t> input = {10000000000LL, -1LL, 2LL, 3LL, -4LL, 5LL};
    return RunCumsumCase<int64_t, int64_t>(
        "Cumsum_Int64_AiCpuProbe", stream, {2, 3}, 1, input, ACL_INT64, ACL_INT64, ACL_INT64,
        ToDoubleVector<int64_t>,
        [](const std::vector<int64_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckExactResult(actual, expected, err);
        });
}

CaseResult CaseInt8SmallNoOverflow(aclrtStream stream)
{
    std::vector<int8_t> input = {1, 2, -1, 3, -2, 1};
    return RunCumsumCase<int8_t, int8_t>(
        "Cumsum_Int8_SmallNoOverflow", stream, {2, 3}, 1, input, ACL_INT8, ACL_INT8, ACL_INT8,
        ToDoubleVector<int8_t>,
        [](const std::vector<int8_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckExactResult(actual, expected, err);
        });
}

CaseResult CaseUint8SmallNoOverflow(aclrtStream stream)
{
    std::vector<uint8_t> input = {1, 2, 3, 4, 5, 6};
    return RunCumsumCase<uint8_t, uint8_t>(
        "Cumsum_UInt8_SmallNoOverflow", stream, {2, 3}, 1, input, ACL_UINT8, ACL_UINT8, ACL_UINT8,
        ToDoubleVector<uint8_t>,
        [](const std::vector<uint8_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckExactResult(actual, expected, err);
        });
}

CaseResult CaseInt8ToInt32Cast(aclrtStream stream)
{
    std::vector<int8_t> input = {1, -2, 3, 4, -5, 6};
    return RunCumsumCase<int8_t, int32_t>(
        "Cumsum_Int8_To_Int32_Cast", stream, {2, 3}, 1, input, ACL_INT8, ACL_INT32, ACL_INT32,
        ToDoubleVector<int8_t>,
        [](const std::vector<int32_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckExactResult(actual, expected, err);
        });
}

CaseResult CaseScalar0D(aclrtStream stream)
{
    std::vector<float> input = {7.0F};
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_0DScalar", stream, {}, 0, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseEmptyTensor(aclrtStream stream)
{
    std::vector<float> input;
    return RunCumsumCase<float, float>(
        "Cumsum_Float32_EmptyTensor", stream, {2, 0, 3}, 1, input, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseV2FloatInclusiveForward(aclrtStream stream)
{
    std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -2.0F, -3.0F, -4.0F};
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_Inclusive_Forward", stream, {2, 4}, 1, false, false, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseV2FloatExclusiveForward(aclrtStream stream)
{
    std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -2.0F, -3.0F, -4.0F};
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_Exclusive_Forward", stream, {2, 4}, 1, true, false, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseV2FloatInclusiveReverse(aclrtStream stream)
{
    std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -2.0F, -3.0F, -4.0F};
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_Inclusive_Reverse", stream, {2, 4}, 1, false, true, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseV2FloatExclusiveReverse(aclrtStream stream)
{
    std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -2.0F, -3.0F, -4.0F};
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_Exclusive_Reverse", stream, {2, 4}, 1, true, true, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseV2FloatNegativeDim(aclrtStream stream)
{
    std::vector<float> input = MakePatternFloat(3 * 4 * 2, 0.25F, 0.5F);
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_NegativeDim", stream, {3, 4, 2}, -2, true, true, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseV2FloatLargeTilingReverse(aclrtStream stream)
{
    std::vector<float> input = MakePatternFloat(4 * 1024 * 16, 1.0F, 0.125F);
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_LargeTiling_Reverse", stream, {4, 1024, 16}, 1, false, true, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 2e-4, 1e-5, err);
        });
}

CaseResult CaseV2Fp16ExclusiveReverse(aclrtStream stream)
{
    std::vector<uint16_t> input = MakeFp16Vector({0.5F, 1.0F, -0.25F, 2.0F, -1.0F, 0.25F, 1.5F, -0.5F});
    return RunCumsumV2Case<uint16_t>(
        "CumsumV2_Float16_Exclusive_Reverse", stream, {2, 4}, 1, true, true, input, ACL_FLOAT16,
        Fp16ToDoubleVector,
        [](const std::vector<uint16_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFp16Result(actual, expected, 2e-2, 2e-3, err);
        });
}

CaseResult CaseV2Int32Reverse(aclrtStream stream)
{
    std::vector<int32_t> input = {1, -2, 3, 4, 5, -6, 7, 8, 9, -1, -2, -3};
    return RunCumsumV2Case<int32_t>(
        "CumsumV2_Int32_Reverse_MiddleAxis", stream, {2, 3, 2}, 1, false, true, input, ACL_INT32,
        ToDoubleVector<int32_t>,
        [](const std::vector<int32_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckExactResult(actual, expected, err);
        });
}

CaseResult CaseV2Int64Exclusive(aclrtStream stream)
{
    std::vector<int64_t> input = {10, -1, 2, 3, -4, 5};
    return RunCumsumV2Case<int64_t>(
        "CumsumV2_Int64_Exclusive_AiCpuProbe", stream, {2, 3}, 1, true, false, input, ACL_INT64,
        ToDoubleVector<int64_t>,
        [](const std::vector<int64_t>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckExactResult(actual, expected, err);
        });
}

CaseResult CaseV2ScalarExclusive(aclrtStream stream)
{
    std::vector<float> input = {7.0F};
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_0DScalar_Exclusive", stream, {}, 0, true, false, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseV2EmptyTensor(aclrtStream stream)
{
    std::vector<float> input;
    return RunCumsumV2Case<float>(
        "CumsumV2_Float32_EmptyTensor", stream, {2, 0, 3}, 1, true, true, input, ACL_FLOAT,
        ToDoubleVector<float>,
        [](const std::vector<float>& actual, const std::vector<double>& expected, std::string* err) {
            return CheckFloatResult(actual, expected, 1e-5, 1e-5, err);
        });
}

CaseResult CaseNegSelfNull(aclrtStream)
{
    TensorHolder out;
    std::vector<float> outHost(4, 0.0F);
    int ret = CreateAclTensor(outHost, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_SelfNull", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_Cumsum_SelfNull", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, out.tensor, ws, ex);
    });
}

CaseResult CaseNegOutNull(aclrtStream)
{
    TensorHolder self;
    std::vector<float> input(4, 1.0F);
    int ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_OutNull", false, "create self failed"};
    }
    return RunNegativeStatusCase("NEG_Cumsum_OutNull", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_FLOAT, nullptr, ws, ex);
    });
}

CaseResult CaseNegDtypeMismatch(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<float> input(4, 1.0F);
    int ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_DtypeMismatch", false, "create self failed"};
    }
    ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_DtypeMismatch", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_Cumsum_DtypeMismatch", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_INT32, out.tensor, ws, ex);
    });
}

CaseResult CaseNegUnsupportedBool(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<uint8_t> input = {1, 0, 1, 0};
    int ret = CreateAclTensor(input, {2, 2}, ACL_BOOL, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_UnsupportedBool", false, "create self failed"};
    }
    ret = CreateAclTensor(input, {2, 2}, ACL_BOOL, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_UnsupportedBool", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_Cumsum_UnsupportedBool", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_BOOL, out.tensor, ws, ex);
    });
}

CaseResult CaseNegShapeMismatch(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<float> input(6, 1.0F);
    std::vector<float> outHost(4, 0.0F);
    int ret = CreateAclTensor(input, {2, 3}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_ShapeMismatch", false, "create self failed"};
    }
    ret = CreateAclTensor(outHost, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_ShapeMismatch", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_Cumsum_ShapeMismatch", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_FLOAT, out.tensor, ws, ex);
    });
}

CaseResult CaseNegDimOutOfRange(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<float> input(4, 1.0F);
    int ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_DimOutOfRange", false, "create self failed"};
    }
    ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_DimOutOfRange", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_Cumsum_DimOutOfRange", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumGetWorkspaceSize(self.tensor, 2, ACL_FLOAT, out.tensor, ws, ex);
    });
}

CaseResult CaseNegRankTooLarge(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> input(1, 1.0F);
    int ret = CreateAclTensor(input, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_RankTooLarge", false, "create self failed"};
    }
    ret = CreateAclTensor(input, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Cumsum_RankTooLarge", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_Cumsum_RankTooLarge", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumGetWorkspaceSize(self.tensor, 0, ACL_FLOAT, out.tensor, ws, ex);
    });
}

CaseResult CaseV2NegSelfNull(aclrtStream)
{
    TensorHolder out;
    std::vector<float> outHost(4, 0.0F);
    int ret = CreateAclTensor(outHost, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_SelfNull", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_CumsumV2_SelfNull", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumV2GetWorkspaceSize(nullptr, 0, false, false, out.tensor, ws, ex);
    });
}

CaseResult CaseV2NegOutNull(aclrtStream)
{
    TensorHolder self;
    std::vector<float> input(4, 1.0F);
    int ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_OutNull", false, "create self failed"};
    }
    return RunNegativeStatusCase("NEG_CumsumV2_OutNull", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, nullptr, ws, ex);
    });
}

CaseResult CaseV2NegUnsupportedBool(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<uint8_t> input = {1, 0, 1, 0};
    int ret = CreateAclTensor(input, {2, 2}, ACL_BOOL, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_UnsupportedBool", false, "create self failed"};
    }
    ret = CreateAclTensor(input, {2, 2}, ACL_BOOL, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_UnsupportedBool", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_CumsumV2_UnsupportedBool", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, out.tensor, ws, ex);
    });
}

CaseResult CaseV2NegShapeMismatch(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<float> input(6, 1.0F);
    std::vector<float> outHost(4, 0.0F);
    int ret = CreateAclTensor(input, {2, 3}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_ShapeMismatch", false, "create self failed"};
    }
    ret = CreateAclTensor(outHost, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_ShapeMismatch", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_CumsumV2_ShapeMismatch", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, out.tensor, ws, ex);
    });
}

CaseResult CaseV2NegDimOutOfRange(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<float> input(4, 1.0F);
    int ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_DimOutOfRange", false, "create self failed"};
    }
    ret = CreateAclTensor(input, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_DimOutOfRange", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_CumsumV2_DimOutOfRange", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumV2GetWorkspaceSize(self.tensor, -3, false, false, out.tensor, ws, ex);
    });
}

CaseResult CaseV2NegRankTooLarge(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> input(1, 1.0F);
    int ret = CreateAclTensor(input, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_RankTooLarge", false, "create self failed"};
    }
    ret = CreateAclTensor(input, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_CumsumV2_RankTooLarge", false, "create out failed"};
    }
    return RunNegativeStatusCase("NEG_CumsumV2_RankTooLarge", [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnCumsumV2GetWorkspaceSize(self.tensor, 0, false, false, out.tensor, ws, ex);
    });
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<CaseResult (*)(aclrtStream)> caseFuncs = {
        CaseFloatBasicDim0,
        CaseFloatNegativeDim,
        CaseFloatLongPointOne,
        CaseFloatMixedMagnitude,
        CaseFloat3DMiddleAxis,
        CaseFloatLargeNPath,
        CaseFloatCubeCandidate,
        CaseDoubleAiCpu,
        CaseFp16Basic,
        CaseFp16CastFromFloat,
        CaseBf16Optional,
        CaseInt32MiddleAxis,
        CaseInt64AiCpu,
        CaseInt8SmallNoOverflow,
        CaseUint8SmallNoOverflow,
        CaseInt8ToInt32Cast,
        CaseScalar0D,
        CaseEmptyTensor,
        CaseV2FloatInclusiveForward,
        CaseV2FloatExclusiveForward,
        CaseV2FloatInclusiveReverse,
        CaseV2FloatExclusiveReverse,
        CaseV2FloatNegativeDim,
        CaseV2FloatLargeTilingReverse,
        CaseV2Fp16ExclusiveReverse,
        CaseV2Int32Reverse,
        CaseV2Int64Exclusive,
        CaseV2ScalarExclusive,
        CaseV2EmptyTensor,
        CaseNegSelfNull,
        CaseNegOutNull,
        CaseNegDtypeMismatch,
        CaseNegUnsupportedBool,
        CaseNegShapeMismatch,
        CaseNegDimOutOfRange,
        CaseNegRankTooLarge,
        CaseV2NegSelfNull,
        CaseV2NegOutNull,
        CaseV2NegUnsupportedBool,
        CaseV2NegShapeMismatch,
        CaseV2NegDimOutOfRange,
        CaseV2NegRankTooLarge,
    };

    std::vector<CaseResult> results;
    for (auto fn : caseFuncs) {
        CaseResult result = fn(stream);
        PrintCaseResult(result);
        results.push_back(result);
    }

    int passCount = 0;
    for (const auto& result : results) {
        if (result.pass) {
            ++passCount;
        }
    }
    int failCount = static_cast<int>(results.size()) - passCount;
    LOG_PRINT("\n==== Cumsum Example Summary ====\n");
    LOG_PRINT("Total: %d, Pass: %d, Fail: %d\n", static_cast<int>(results.size()), passCount, failCount);

    Finalize(deviceId, stream);
    return failCount == 0 ? 0 : 1;
}
