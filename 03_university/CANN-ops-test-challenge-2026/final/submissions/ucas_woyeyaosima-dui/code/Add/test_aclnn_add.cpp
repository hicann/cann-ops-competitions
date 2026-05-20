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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
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
    } while (0)


#ifndef ACL_SYNC_TIMEOUT_MS
#define ACL_SYNC_TIMEOUT_MS 10000U
#endif

// 自定义返回码：用于区分 aclnn* 入队失败 和 stream 同步超时/失败，避免失败后再次调用 GetWorkspace。
static constexpr int kSyncTimeoutError = -1000001;
static int g_lastSyncRuntimeError = ACL_SUCCESS;

uint32_t GetSyncTimeoutMs()
{
    constexpr uint32_t defaultTimeout = ACL_SYNC_TIMEOUT_MS;
    const char* envValue = std::getenv("ACL_SYNC_TIMEOUT_MS");
    if (envValue == nullptr || envValue[0] == '\0') {
        return defaultTimeout;
    }

    char* endPtr = nullptr;
    long parsed = std::strtol(envValue, &endPtr, 10);
    if (endPtr == envValue || *endPtr != '\0' || parsed <= 0) {
        LOG_PRINT("[WARN] invalid ACL_SYNC_TIMEOUT_MS=%s, use default=%u\n", envValue, defaultTimeout);
        return defaultTimeout;
    }
    return static_cast<uint32_t>(parsed);
}

bool IsEnvEnabled(const char* envName)
{
    const char* value = std::getenv(envName);
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON" || text == "yes" || text == "YES";
}

bool GetBoolEnv(const char* envName, bool defaultValue)
{
    const char* value = std::getenv(envName);
    if (value == nullptr) {
        return defaultValue;
    }

    const std::string text(value);
    if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON" || text == "yes" || text == "YES") {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF" || text == "no" || text == "NO") {
        return false;
    }
    LOG_PRINT("[WARN] invalid %s=%s, use default=%s\n", envName, value, defaultValue ? "1" : "0");
    return defaultValue;
}

uint32_t g_syncTimeoutMs = GetSyncTimeoutMs();

int SyncStreamWithTimeout(aclrtStream stream)
{
    g_lastSyncRuntimeError = ACL_SUCCESS;
    auto ret = aclrtSynchronizeStreamWithTimeout(stream, g_syncTimeoutMs);
    if (ret != ACL_SUCCESS) {
        g_lastSyncRuntimeError = static_cast<int>(ret);
        LOG_PRINT("aclrtSynchronizeStreamWithTimeout failed/timeout. timeout=%u ms, ERROR: %d\n",
                  g_syncTimeoutMs,
                  static_cast<int>(ret));
        return kSyncTimeoutError;
    }
    return ACL_SUCCESS;
}

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

struct ScalarHolder {
    aclScalar* scalar = nullptr;

    ScalarHolder() = default;
    ScalarHolder(const ScalarHolder&) = delete;
    ScalarHolder& operator=(const ScalarHolder&) = delete;

    ~ScalarHolder()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
            scalar = nullptr;
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

bool RecoverRuntimeContext(int32_t deviceId, aclrtStream* stream)
{
    CHECK_RET(stream != nullptr, return false);

    if (*stream != nullptr) {
        auto ret = aclrtDestroyStream(*stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[WARN] aclrtDestroyStream during recovery failed. ERROR: %d\n", ret);
        }
        *stream = nullptr;
    }

    auto ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[WARN] aclrtResetDevice during recovery failed. ERROR: %d\n", ret);
    }

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice during recovery failed. ERROR: %d\n", ret); return false);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream during recovery failed. ERROR: %d\n", ret); return false);
    return true;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                                        const std::vector<int64_t>& shape,
                                        aclDataType dataType,
                                        TensorHolder* holder,
                                        aclFormat format = ACL_FORMAT_ND)
{
    CHECK_RET(holder != nullptr, return ACL_ERROR_INVALID_PARAM);
    holder->Reset();

    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto strides = GetContiguousStrides(shape);

    if (size > 0) {
        auto ret = aclrtMalloc(&holder->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        ret = aclrtMemcpy(holder->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    holder->tensor = aclCreateTensor(shape.data(),
                                                                     shape.size(),
                                                                     dataType,
                                                                     strides.data(),
                                                                     0,
                                                                     format,
                                                                     shape.data(),
                                                                     shape.size(),
                                                                     holder->deviceAddr);
    CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

template <typename T>
int CopyTensorToHost(const TensorHolder& holder, std::vector<T>* hostData)
{
    CHECK_RET(hostData != nullptr, return ACL_ERROR_INVALID_PARAM);
    auto size = static_cast<int64_t>(hostData->size() * sizeof(T));
    if (size == 0) {
        return ACL_SUCCESS;
    }
    CHECK_RET(holder.deviceAddr != nullptr, return ACL_ERROR_INVALID_PARAM);
    auto ret = aclrtMemcpy(hostData->data(), size, holder.deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclScalar(const T& value, aclDataType dataType, ScalarHolder* holder)
{
    CHECK_RET(holder != nullptr, return ACL_ERROR_INVALID_PARAM);
    if (holder->scalar != nullptr) {
        aclDestroyScalar(holder->scalar);
        holder->scalar = nullptr;
    }
    T mutableValue = value;
    holder->scalar = aclCreateScalar(&mutableValue, dataType);
    CHECK_RET(holder->scalar != nullptr, return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
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

bool CheckFloatResult(const std::vector<float>& actual,
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

    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = static_cast<double>(actual[i]);
        const double e = expected[i];
        const bool bothNaN = std::isnan(a) && std::isnan(e);
        const bool bothInf = std::isinf(a) && std::isinf(e) && ((a > 0) == (e > 0));
        if (bothNaN || bothInf) {
            continue;
        }
        if (!IsClose(a, e, atol, rtol)) {
            if (err != nullptr) {
                *err = "index=" + std::to_string(i) + " actual=" + std::to_string(a) + " expected=" + std::to_string(e);
            }
            return false;
        }
    }
    return true;
}

bool CheckFp16Result(const std::vector<uint16_t>& actual,
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

    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = static_cast<double>(Fp16BitsToFloat(actual[i]));
        const double e = expected[i];
        if (!IsClose(a, e, atol, rtol)) {
            if (err != nullptr) {
                *err = "index=" + std::to_string(i) + " actual=" + std::to_string(a) + " expected=" + std::to_string(e);
            }
            return false;
        }
    }
    return true;
}

bool CheckBf16Result(const std::vector<uint16_t>& actual,
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

    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = static_cast<double>(Bf16BitsToFloat(actual[i]));
        const double e = expected[i];
        if (!IsClose(a, e, atol, rtol)) {
            if (err != nullptr) {
                *err = "index=" + std::to_string(i) + " actual=" + std::to_string(a) + " expected=" + std::to_string(e);
            }
            return false;
        }
    }
    return true;
}

template <typename T>
bool CheckExactResult(const std::vector<T>& actual, const std::vector<T>& expected, std::string* err)
{
    if (actual.size() != expected.size()) {
        if (err != nullptr) {
            *err = "size mismatch";
        }
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (err != nullptr) {
                *err = "index=" + std::to_string(i) + " actual=" + std::to_string(static_cast<int64_t>(actual[i])) +
                             " expected=" + std::to_string(static_cast<int64_t>(expected[i]));
            }
            return false;
        }
    }
    return true;
}

std::string FormatDoubleScientific(double value)
{
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    char buffer[64] = {0};
    std::snprintf(buffer, sizeof(buffer), "%.9e", value);
    return std::string(buffer);
}

struct FloatErrorMetrics {
    bool valid = true;
    bool specialValueMismatch = false;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    size_t maxAbsIndex = 0;
    size_t maxRelIndex = 0;
    double maxAbsActual = 0.0;
    double maxAbsExpected = 0.0;
};

FloatErrorMetrics ComputeFloatErrorMetrics(const std::vector<float>& actual, const std::vector<double>& expected)
{
    FloatErrorMetrics metrics;
    if (actual.size() != expected.size()) {
        metrics.valid = false;
        return metrics;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = static_cast<double>(actual[i]);
        const double e = expected[i];
        const bool bothNaN = std::isnan(a) && std::isnan(e);
        const bool bothInf = std::isinf(a) && std::isinf(e) && ((a > 0) == (e > 0));
        if (bothNaN || bothInf) {
            continue;
        }

        if (!std::isfinite(a) || !std::isfinite(e)) {
            metrics.specialValueMismatch = true;
            metrics.maxAbsError = std::numeric_limits<double>::infinity();
            metrics.maxRelError = std::numeric_limits<double>::infinity();
            metrics.maxAbsIndex = i;
            metrics.maxRelIndex = i;
            metrics.maxAbsActual = a;
            metrics.maxAbsExpected = e;
            break;
        }

        const double absErr = std::fabs(a - e);
        const double relDenominator = std::max(std::fabs(e), 1e-30);
        const double relErr = absErr / relDenominator;
        if (absErr > metrics.maxAbsError) {
            metrics.maxAbsError = absErr;
            metrics.maxAbsIndex = i;
            metrics.maxAbsActual = a;
            metrics.maxAbsExpected = e;
        }
        if (relErr > metrics.maxRelError) {
            metrics.maxRelError = relErr;
            metrics.maxRelIndex = i;
        }
    }
    return metrics;
}

std::string BuildFloatPrecisionDetail(const std::vector<float>& actual,
                                      const std::vector<double>& expected,
                                      double warnAbsError,
                                      double warnRelError)
{
    auto metrics = ComputeFloatErrorMetrics(actual, expected);
    if (!metrics.valid) {
        return "precision_metrics=size_mismatch";
    }

    const bool warning = metrics.specialValueMismatch ||
                         (metrics.maxAbsError > warnAbsError) ||
                         (metrics.maxRelError > warnRelError);
    return "precision_metrics: maxAbsErr=" + FormatDoubleScientific(metrics.maxAbsError) +
           " (idx=" + std::to_string(metrics.maxAbsIndex) +
           ", actual=" + FormatDoubleScientific(metrics.maxAbsActual) +
           ", expected=" + FormatDoubleScientific(metrics.maxAbsExpected) +
           "), maxRelErr=" + FormatDoubleScientific(metrics.maxRelError) +
           " (idx=" + std::to_string(metrics.maxRelIndex) + ")" +
           ", warning=" + (warning ? "YES" : "NO");
}

void GetCoordsFromLinear(int64_t linearIndex, const std::vector<int64_t>& shape, std::vector<int64_t>* coords)
{
    coords->assign(shape.size(), 0);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        int64_t dim = shape[static_cast<size_t>(i)];
        if (dim > 0) {
            (*coords)[static_cast<size_t>(i)] = linearIndex % dim;
            linearIndex /= dim;
        }
    }
}

int64_t GetBroadcastOffset(const std::vector<int64_t>& inShape,
                                                    const std::vector<int64_t>& outShape,
                                                    const std::vector<int64_t>& outCoords)
{
    auto inStrides = GetContiguousStrides(inShape);
    int64_t inRank = static_cast<int64_t>(inShape.size());
    int64_t outRank = static_cast<int64_t>(outShape.size());
    int64_t rankDiff = outRank - inRank;
    int64_t offset = 0;

    for (int64_t i = 0; i < inRank; ++i) {
        int64_t outAxis = i + rankDiff;
        int64_t coord = (inShape[static_cast<size_t>(i)] == 1) ? 0 : outCoords[static_cast<size_t>(outAxis)];
        offset += coord * inStrides[static_cast<size_t>(i)];
    }
    return offset;
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

std::vector<double> ComputeExpectedBroadcastAdd(const std::vector<double>& self,
                                                                                                const std::vector<int64_t>& selfShape,
                                                                                                const std::vector<double>& other,
                                                                                                const std::vector<int64_t>& otherShape,
                                                                                                const std::vector<int64_t>& outShape,
                                                                                                double alpha)
{
    std::vector<double> expected(static_cast<size_t>(GetShapeSize(outShape)), 0.0);
    std::vector<int64_t> coords;

    for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
        GetCoordsFromLinear(i, outShape, &coords);
        int64_t selfOffset = GetBroadcastOffset(selfShape, outShape, coords);
        int64_t otherOffset = GetBroadcastOffset(otherShape, outShape, coords);
        expected[static_cast<size_t>(i)] =
                self[static_cast<size_t>(selfOffset)] + alpha * other[static_cast<size_t>(otherOffset)];
    }
    return expected;
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
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = SyncStreamWithTimeout(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

int RunAdd(const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, aclrtStream stream)
{
    return RunWithWorkspace(
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self, other, alpha, out, workspaceSize, executor);
            },
            [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
                return aclnnAdd(workspace, workspaceSize, executor, runStream);
            },
            stream);
}

int RunAdds(const aclTensor* self,
                        const aclScalar* other,
                        const aclScalar* alpha,
                        aclTensor* out,
                        aclrtStream stream)
{
    return RunWithWorkspace(
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self, other, alpha, out, workspaceSize, executor);
            },
            [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
                return aclnnAdds(workspace, workspaceSize, executor, runStream);
            },
            stream);
}

int RunInplaceAdd(aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha, aclrtStream stream)
{
    return RunWithWorkspace(
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddGetWorkspaceSize(selfRef, other, alpha, workspaceSize, executor);
            },
            [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
                return aclnnInplaceAdd(workspace, workspaceSize, executor, runStream);
            },
            stream);
}

int RunInplaceAdds(aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha, aclrtStream stream)
{
    return RunWithWorkspace(
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddsGetWorkspaceSize(selfRef, other, alpha, workspaceSize, executor);
            },
            [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
                return aclnnInplaceAdds(workspace, workspaceSize, executor, runStream);
            },
            stream);
}

int RunAddV3(const aclScalar* self,
                         const aclTensor* other,
                         const aclScalar* alpha,
                         aclTensor* out,
                         aclrtStream stream)
{
    return RunWithWorkspace(
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self, other, alpha, out, workspaceSize, executor);
            },
            [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
                return aclnnAddV3(workspace, workspaceSize, executor, runStream);
            },
            stream);
}

int RunInplaceAddV3(const aclScalar* selfRef, const aclTensor* other, const aclScalar* alpha, aclrtStream stream)
{
    return RunWithWorkspace(
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddV3GetWorkspaceSize(selfRef, other, alpha, workspaceSize, executor);
            },
            [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
                return aclnnInplaceAddV3(workspace, workspaceSize, executor, runStream);
            },
            stream);
}

struct CaseResult {
    std::string name;
    bool pass = false;
    std::string detail;
};

CaseResult MakeStatusResult(const std::string& name, aclnnStatus actual, aclnnStatus expected)
{
    CaseResult result{name, actual == expected, ""};
    if (!result.pass) {
        result.detail = "status=" + std::to_string(static_cast<int>(actual)) + " expected=" +
                                        std::to_string(static_cast<int>(expected));
    }
    return result;
}

CaseResult MakeStatusNotSuccessResult(const std::string& name, aclnnStatus actual)
{
    CaseResult result{name, actual != ACL_SUCCESS, ""};
    if (!result.pass) {
        result.detail = "status=ACL_SUCCESS (unexpected)";
    }
    return result;
}

CaseResult MakeFloatCompareResult(const std::string& name,
                                  const std::vector<float>& actual,
                                  const std::vector<double>& expected,
                                  double atol,
                                  double rtol,
                                  double warnAbsError,
                                  double warnRelError)
{
    CaseResult result{name, false, ""};
    std::string err;
    result.pass = CheckFloatResult(actual, expected, atol, rtol, &err);
    const std::string precisionDetail = BuildFloatPrecisionDetail(actual, expected, warnAbsError, warnRelError);
    if (result.pass) {
        result.detail = precisionDetail;
    } else {
        result.detail = err.empty() ? precisionDetail : (err + "; " + precisionDetail);
    }
    return result;
}

bool TryWorkspaceOnlyOnRunFail(
    CaseResult* result,
    const char* apiName,
    aclnnStatus runStatus,
    const std::function<aclnnStatus(uint64_t*, aclOpExecutor**)>& getWorkspace)
{
    if (runStatus == ACL_SUCCESS) {
        return false;
    }

    // 如果是 stream 同步超时/失败，不要再次调用 GetWorkspaceSize。
    // 设备队列里可能已有未完成任务，再次查询可能造成二次阻塞或掩盖真实错误。
    if (static_cast<int>(runStatus) == kSyncTimeoutError) {
        result->pass = false;
        result->detail = std::string(apiName) + " sync stream failed/timeout, runtime=" +
                         std::to_string(g_lastSyncRuntimeError) + ", timeout_ms=" +
                         std::to_string(ACL_SYNC_TIMEOUT_MS);
        return true;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto wsStatus = getWorkspace(&workspaceSize, &executor);
    result->pass = false;
    result->detail = std::string(apiName) + " run failed, run=" +
                     std::to_string(static_cast<int>(runStatus)) + ", workspace=" +
                     std::to_string(static_cast<int>(wsStatus));
    return true;
}

void PrintCaseResult(const CaseResult& result)
{
    if (result.pass) {
        if (result.detail.empty()) {
            LOG_PRINT("[PASS] %s\n", result.name.c_str());
        } else {
            LOG_PRINT("[PASS] %s -> %s\n", result.name.c_str(), result.detail.c_str());
        }
    } else {
        LOG_PRINT("[FAIL] %s -> %s\n", result.name.c_str(), result.detail.c_str());
    }
}

CaseResult RunAddFloatCase(const std::string& name,
                                                     aclrtStream stream,
                                                     const std::vector<int64_t>& selfShape,
                                                     const std::vector<float>& selfHost,
                                                     const std::vector<int64_t>& otherShape,
                                                     const std::vector<float>& otherHost,
                                                     const std::vector<int64_t>& outShape,
                                                     float alphaValue,
                                                     double atol,
                                                     double rtol,
                                                     aclFormat format = ACL_FORMAT_ND)
{
    CaseResult result{name, false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<float> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0.0F);

    int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self, format);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other, format);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, outShape, ACL_FLOAT, &out, format);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            ToDoubleVector(selfHost), selfShape, ToDoubleVector(otherHost), otherShape, outShape, static_cast<double>(alphaValue));

    std::string err;
    result.pass = CheckFloatResult(outHost, expected, atol, rtol, &err);
    result.detail = err;
    return result;
}

template <typename T, typename AlphaT>
CaseResult RunAddExactCase(const std::string& name,
                                                     aclrtStream stream,
                                                     aclDataType dtype,
                                                     const std::vector<int64_t>& selfShape,
                                                     const std::vector<T>& selfHost,
                                                     const std::vector<int64_t>& otherShape,
                                                     const std::vector<T>& otherHost,
                                                     const std::vector<int64_t>& outShape,
                                                     AlphaT alphaValue,
                                                     aclDataType alphaType)
{
    CaseResult result{name, false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<T> outHost(static_cast<size_t>(GetShapeSize(outShape)), static_cast<T>(0));

    int ret = CreateAclTensor(selfHost, selfShape, dtype, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, otherShape, dtype, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, outShape, dtype, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, alphaType, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expectedD = ComputeExpectedBroadcastAdd(ToDoubleVector(selfHost),
                                                                                             selfShape,
                                                                                             ToDoubleVector(otherHost),
                                                                                             otherShape,
                                                                                             outShape,
                                                                                             static_cast<double>(alphaValue));
    std::vector<T> expected(outHost.size(), static_cast<T>(0));
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<T>(expectedD[i]);
    }

    std::string err;
    result.pass = CheckExactResult(outHost, expected, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddFloatBasicAlpha1(aclrtStream stream)
{
    return RunAddFloatCase("Add_Float32_Basic_Alpha1",
                                                 stream,
                                                 {2, 4},
                                                 {0.0F, 1.0F, 2.0F, 3.0F, -1.0F, -2.0F, 4.0F, 5.0F},
                                                 {2, 4},
                                                 {1.0F, -1.0F, 3.0F, 2.0F, 0.5F, 0.5F, -2.0F, 1.0F},
                                                 {2, 4},
                                                 1.0F,
                                                 1e-5,
                                                 1e-5);
}

CaseResult CaseAddFloatBroadcastAlphaNeg(aclrtStream stream)
{
    return RunAddFloatCase("Add_Float32_Broadcast_AlphaNeg",
                                                 stream,
                                                 {2, 3},
                                                 {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                                 {3},
                                                 {2.0F, -1.0F, 0.5F},
                                                 {2, 3},
                                                 -0.5F,
                                                 1e-5,
                                                 1e-5);
}

CaseResult CaseAddFloatAlphaZeroLarge(aclrtStream stream)
{
    const int64_t elemCount = 4096;
    std::vector<float> selfHost(static_cast<size_t>(elemCount), 0.0F);
    std::vector<float> otherHost(static_cast<size_t>(elemCount), 0.0F);
    for (int64_t i = 0; i < elemCount; ++i) {
        selfHost[static_cast<size_t>(i)] = static_cast<float>((i % 97) - 48) * 0.125F;
        otherHost[static_cast<size_t>(i)] = static_cast<float>((i % 53) - 26) * 3.0F;
    }

    return RunAddFloatCase("Add_Float32_Large_AlphaZero",
                                                 stream,
                                                 {elemCount},
                                                 selfHost,
                                                 {elemCount},
                                                 otherHost,
                                                 {elemCount},
                                                 0.0F,
                                                 1e-6,
                                                 1e-6);
}

CaseResult CaseAddFloatBroadcast4DAlphaQuarter(aclrtStream stream)
{
    return RunAddFloatCase("Add_Float32_Broadcast4D_AlphaQuarter",
                                                 stream,
                                                 {2, 1, 3, 1},
                                                 {1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 6.0F},
                                                 {1, 4, 1, 5},
                                                 {0.0F, 1.0F, -1.0F, 2.0F, -2.0F,
                                                  3.0F, -3.0F, 4.0F, -4.0F, 5.0F,
                                                  0.5F, -0.5F, 1.5F, -1.5F, 2.5F,
                                                  -2.5F, 6.0F, -6.0F, 7.0F, -7.0F},
                                                 {2, 4, 3, 5},
                                                 0.25F,
                                                 1e-5,
                                                 1e-5);
}

CaseResult CaseAddFloatPrecisionStress(aclrtStream stream)
{
    return RunAddFloatCase("Add_Float32_PrecisionStress",
                                                 stream,
                                                 {8},
                                                 {1.0e10F, -1.0e10F, 1.0000001F, -1.0000001F,
                                                  0.1F, -0.3F, 1.0e-37F, -1.0e-37F},
                                                 {8},
                                                 {1.0e-5F, -1.0e-5F, -1.0F, 1.0F,
                                                  0.2F, 0.1F, 1.0e-37F, -1.0e-37F},
                                                 {8},
                                                 1.0F,
                                                 1e-6,
                                                 1e-6);
}

CaseResult CaseAddFloatPrecisionLargePlusTiny(aclrtStream stream)
{
    CaseResult result{"Add_Float32_Precision_LargePlusTiny", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {6};
    std::vector<float> selfHost = {1.0e10F, -1.0e10F, 16777216.0F, -16777216.0F, 123456.75F, -123456.75F};
    std::vector<float> otherHost = {1.0e-3F, -1.0e-3F, 1.0F, -1.0F, 1.0e-5F, -1.0e-5F};
    std::vector<float> outHost(selfHost.size(), 0.0F);

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, static_cast<double>(alphaValue));
    result = MakeFloatCompareResult("Add_Float32_Precision_LargePlusTiny", outHost, expected, 2.0, 1e-6, 1e-4, 1e-6);
    result.detail = "scenario=large_plus_tiny; " + result.detail;
    return result;
}

CaseResult CaseAddFloatPrecisionCancellationAlpha(aclrtStream stream)
{
    CaseResult result{"Add_Float32_Precision_CancellationAlpha", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {6};
    std::vector<float> selfHost = {100000.0F, -100000.0F, 123456.789F, -123456.789F, 1.0000001F, -1.0000001F};
    std::vector<float> otherHost = {-1000000.0F, 1000000.0F, -123456.0F, 123456.0F, -1.0F, 1.0F};
    std::vector<float> outHost(selfHost.size(), 0.0F);

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 0.1F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            ToDoubleVector(selfHost), shape, ToDoubleVector(otherHost), shape, shape, static_cast<double>(alphaValue));
    result = MakeFloatCompareResult("Add_Float32_Precision_CancellationAlpha", outHost, expected, 5e-3, 5e-6, 1e-4, 1e-5);
    result.detail = "scenario=cancellation_with_alpha; " + result.detail;
    return result;
}

CaseResult CaseAddFloatNaNInf(aclrtStream stream)
{
    return RunAddFloatCase("Add_Float32_NaN_Inf",
                                                 stream,
                                                 {4},
                                                 {std::numeric_limits<float>::quiet_NaN(),
                                                    std::numeric_limits<float>::infinity(),
                                                    -std::numeric_limits<float>::infinity(),
                                                    3.0F},
                                                 {4},
                                                 {2.0F, -1.0F, 0.0F, std::numeric_limits<float>::infinity()},
                                                 {4},
                                                 1.0F,
                                                 1e-5,
                                                 1e-5);
}

CaseResult CaseAddInt32Alpha2(aclrtStream stream)
{
    return RunAddExactCase<int32_t, int32_t>("Add_Int32_Alpha2",
                                                                                        stream,
                                                                                        ACL_INT32,
                                                                                        {2, 3},
                                                                                        {0, -1, 7, 9, -3, 2},
                                                                                        {2, 3},
                                                                                        {8, 2, -2, 0, -4, 10},
                                                                                        {2, 3},
                                                                                        2,
                                                                                        ACL_INT32);
}

CaseResult CaseAddInt64Alpha2(aclrtStream stream)
{
    return RunAddExactCase<int64_t, int64_t>("Add_Int64_Alpha2",
                                                                                        stream,
                                                                                        ACL_INT64,
                                                                                        {2, 3},
                                                                                        {1, -2, 3, -4, 5, -6},
                                                                                        {2, 3},
                                                                                        {2, 3, -1, -2, 1, 2},
                                                                                        {2, 3},
                                                                                        2,
                                                                                        ACL_INT64);
}

CaseResult CaseAddUInt8Alpha3(aclrtStream stream)
{
    return RunAddExactCase<uint8_t, uint8_t>("Add_UInt8_Alpha3",
                                                                                        stream,
                                                                                        ACL_UINT8,
                                                                                        {2, 4},
                                                                                        {1, 2, 3, 4, 5, 6, 7, 8},
                                                                                        {2, 4},
                                                                                        {2, 3, 1, 2, 1, 1, 2, 3},
                                                                                        {2, 4},
                                                                                        static_cast<uint8_t>(3),
                                                                                        ACL_UINT8);
}

CaseResult CaseAddInt32Alpha1(aclrtStream stream)
{
    return RunAddExactCase<int32_t, int32_t>("Add_Int32_Alpha1",
                                                                                        stream,
                                                                                        ACL_INT32,
                                                                                        {2, 2},
                                                                                        {1, -2, 3, 4},
                                                                                        {2, 2},
                                                                                        {5, 6, -7, 8},
                                                                                        {2, 2},
                                                                                        1,
                                                                                        ACL_INT32);
}

CaseResult CaseAddInt64Alpha1(aclrtStream stream)
{
    return RunAddExactCase<int64_t, int64_t>("Add_Int64_Alpha1",
                                                                                        stream,
                                                                                        ACL_INT64,
                                                                                        {2, 2},
                                                                                        {10, -20, 30, -40},
                                                                                        {2, 2},
                                                                                        {1, 2, -3, -4},
                                                                                        {2, 2},
                                                                                        1,
                                                                                        ACL_INT64);
}

CaseResult CaseAddInt8Alpha1(aclrtStream stream)
{
    return RunAddExactCase<int8_t, int8_t>("Add_Int8_Alpha1",
                                                                                    stream,
                                                                                    ACL_INT8,
                                                                                    {2, 4},
                                                                                    {1, -2, 3, -4, 5, -6, 7, -8},
                                                                                    {2, 4},
                                                                                    {-1, 1, -1, 1, -2, 2, -2, 2},
                                                                                    {2, 4},
                                                                                    static_cast<int8_t>(1),
                                                                                    ACL_INT8);
}

CaseResult CaseAddUInt8Alpha1(aclrtStream stream)
{
    return RunAddExactCase<uint8_t, uint8_t>("Add_UInt8_Alpha1",
                                                                                        stream,
                                                                                        ACL_UINT8,
                                                                                        {2, 3},
                                                                                        {1, 2, 3, 4, 5, 6},
                                                                                        {2, 3},
                                                                                        {6, 5, 4, 3, 2, 1},
                                                                                        {2, 3},
                                                                                        static_cast<uint8_t>(1),
                                                                                        ACL_UINT8);
}

CaseResult CaseAddDoubleAiCpuProbe(aclrtStream stream)
{
    CaseResult result{"Add_Double_AiCpu_Probe", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 2};
    std::vector<double> selfHost = {1.0, -2.0, 3.5, 4.0};
    std::vector<double> otherHost = {2.0, 1.0, -1.5, 0.5};
    std::vector<double> outHost(shape[0] * shape[1], 0.0);

    int ret = CreateAclTensor(selfHost, shape, ACL_DOUBLE, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_DOUBLE, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_DOUBLE, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    double alphaValue = 1.0;
    ret = CreateAclScalar(alphaValue, ACL_DOUBLE, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto wsRet = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (wsRet != ACL_SUCCESS) {
        result.detail = "get workspace failed, status=" + std::to_string(static_cast<int>(wsRet));
        return result;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            result.detail = "malloc workspace failed";
            return result;
        }
    }

    auto runRet = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
    }
    if (runRet != ACL_SUCCESS) {
        result.detail = "run add failed, status=" + std::to_string(static_cast<int>(runRet));
        return result;
    }

    ret = SyncStreamWithTimeout(stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "sync stream failed/timeout, runtime=" + std::to_string(g_lastSyncRuntimeError) +
                        ", timeout_ms=" + std::to_string(ACL_SYNC_TIMEOUT_MS);
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    bool ok = true;
    std::string err;
    for (size_t i = 0; i < outHost.size(); ++i) {
        double expected = selfHost[i] + alphaValue * otherHost[i];
        if (!IsClose(outHost[i], expected, 1e-12, 1e-12)) {
            ok = false;
            err = "index=" + std::to_string(i) + " actual=" + std::to_string(outHost[i]) +
                  " expected=" + std::to_string(expected);
            break;
        }
    }
    result.pass = ok;
    result.detail = err;
    return result;
}

CaseResult CaseAddDoubleMulPathAlphaHalf(aclrtStream stream)
{
    CaseResult result{"Add_Double_MulPath_AlphaHalf", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 2};
    std::vector<double> selfHost = {1.0, -2.0, 3.5, 4.0};
    std::vector<double> otherHost = {2.0, 1.0, -1.5, 0.5};
    std::vector<double> outHost(shape[0] * shape[1], 0.0);

    int ret = CreateAclTensor(selfHost, shape, ACL_DOUBLE, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_DOUBLE, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_DOUBLE, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    double alphaValue = 0.5;
    ret = CreateAclScalar(alphaValue, ACL_DOUBLE, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    bool ok = true;
    std::string err;
    for (size_t i = 0; i < outHost.size(); ++i) {
        double expected = selfHost[i] + alphaValue * otherHost[i];
        if (!IsClose(outHost[i], expected, 1e-12, 1e-12)) {
            ok = false;
            err = "index=" + std::to_string(i) + " actual=" + std::to_string(outHost[i]) +
                  " expected=" + std::to_string(expected);
            break;
        }
    }
    result.pass = ok;
    result.detail = err;
    return result;
}

CaseResult CaseAddOutAliasOtherProbe(aclrtStream stream)
{
    CaseResult result{"Add_OutAliasOther_Probe", false, ""};
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfHost = {1.0F, -2.0F, 3.5F, 4.0F};
    std::vector<float> otherHost = {2.0F, 1.0F, -1.5F, 0.5F};
    const std::vector<float> otherBefore = otherHost;

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto wsRet = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, other.tensor, &workspaceSize, &executor);
    if (wsRet != ACL_SUCCESS) {
        result.detail = "get workspace failed, status=" + std::to_string(static_cast<int>(wsRet));
        return result;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            result.detail = "malloc workspace failed";
            return result;
        }
    }

    auto runRet = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
    }
    if (runRet != ACL_SUCCESS) {
        result.detail = "run add failed, status=" + std::to_string(static_cast<int>(runRet));
        return result;
    }

    ret = SyncStreamWithTimeout(stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "sync stream failed/timeout, runtime=" + std::to_string(g_lastSyncRuntimeError) +
                        ", timeout_ms=" + std::to_string(ACL_SYNC_TIMEOUT_MS);
        return result;
    }

    ret = CopyTensorToHost(other, &otherHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy other failed";
        return result;
    }

    std::vector<double> expected(otherHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfHost[i]) + static_cast<double>(alphaValue) * static_cast<double>(otherBefore[i]);
    }

    std::string err;
    result.pass = CheckFloatResult(otherHost, expected, 1e-6, 1e-6, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddFp16AlphaHalf(aclrtStream stream)
{
    CaseResult result{"Add_Fp16_AlphaHalf", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.0F};
    std::vector<float> otherFp32 = {2.0F, -1.5F, 8.0F, 0.25F};

    std::vector<uint16_t> selfFp16(shape[0], 0);
    std::vector<uint16_t> otherFp16(shape[0], 0);
    std::vector<uint16_t> outFp16(shape[0], 0);
    for (size_t i = 0; i < selfFp16.size(); ++i) {
        selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
        otherFp16[i] = FloatToFp16Bits(otherFp32[i]);
    }

    int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherFp16, shape, ACL_FLOAT16, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outFp16, shape, ACL_FLOAT16, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 0.5F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outFp16);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            Fp16ToDoubleVector(selfFp16), shape, Fp16ToDoubleVector(otherFp16), shape, shape, static_cast<double>(alphaValue));

    std::string err;
    result.pass = CheckFp16Result(outFp16, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddBf16Alpha1(aclrtStream stream)
{
    CaseResult result{"Add_Bf16_Alpha1", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<float> selfFp32 = {1.0F, -2.5F, 0.5F, 3.0F};
    std::vector<float> otherFp32 = {2.0F, -1.25F, 8.0F, 0.5F};

    std::vector<uint16_t> selfBf16(shape[0], 0);
    std::vector<uint16_t> otherBf16(shape[0], 0);
    std::vector<uint16_t> outBf16(shape[0], 0);
    for (size_t i = 0; i < selfBf16.size(); ++i) {
        selfBf16[i] = FloatToBf16Bits(selfFp32[i]);
        otherBf16[i] = FloatToBf16Bits(otherFp32[i]);
    }

    int ret = CreateAclTensor(selfBf16, shape, ACL_BF16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherBf16, shape, ACL_BF16, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outBf16, shape, ACL_BF16, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "run add failed";
        return result;
    }

    ret = CopyTensorToHost(out, &outBf16);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            Bf16ToDoubleVector(selfBf16), shape, Bf16ToDoubleVector(otherBf16), shape, shape, static_cast<double>(alphaValue));

    std::string err;
    result.pass = CheckBf16Result(outBf16, expected, 1e-2, 1e-2, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddMixFp16Fp32(aclrtStream stream)
{
    CaseResult result{"Add_Mix_Fp16_Fp32", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<float> selfFp32 = {1.5F, -2.25F, 0.125F, 3.0F};
    std::vector<uint16_t> selfFp16(shape[0], 0);
    for (size_t i = 0; i < selfFp16.size(); ++i) {
        selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
    }
    std::vector<float> otherHost = {2.0F, -1.0F, 8.0F, 0.5F};
    std::vector<float> outHost(shape[0], 0.0F);

    int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "run add failed";
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            Fp16ToDoubleVector(selfFp16), shape, ToDoubleVector(otherHost), shape, shape, static_cast<double>(alphaValue));
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddMixFp32Fp16(aclrtStream stream)
{
    CaseResult result{"Add_Mix_Fp32_Fp16", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<float> selfHost = {1.0F, -2.0F, 0.5F, 3.5F};
    std::vector<float> otherFp32 = {2.0F, -1.5F, 4.0F, 0.5F};
    std::vector<uint16_t> otherFp16(shape[0], 0);
    for (size_t i = 0; i < otherFp16.size(); ++i) {
        otherFp16[i] = FloatToFp16Bits(otherFp32[i]);
    }
    std::vector<float> outHost(shape[0], 0.0F);

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherFp16, shape, ACL_FLOAT16, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "run add failed";
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            ToDoubleVector(selfHost), shape, Fp16ToDoubleVector(otherFp16), shape, shape, static_cast<double>(alphaValue));
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddMixFp16Fp32BroadcastAlpha1(aclrtStream stream)
{
    CaseResult result{"Add_Mix_Fp16_Fp32_Broadcast_Alpha1", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {3};
    std::vector<int64_t> outShape = {2, 3};
    std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.0F, -4.0F, 6.0F};
    std::vector<uint16_t> selfFp16(selfFp32.size(), 0);
    for (size_t i = 0; i < selfFp16.size(); ++i) {
        selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
    }
    std::vector<float> otherHost = {0.5F, -1.0F, 2.0F};
    std::vector<float> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0.0F);

    int ret = CreateAclTensor(selfFp16, selfShape, ACL_FLOAT16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, outShape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            Fp16ToDoubleVector(selfFp16), selfShape, ToDoubleVector(otherHost), otherShape, outShape, static_cast<double>(alphaValue));
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddFp16Bf16Alpha1CastPath(aclrtStream stream)
{
    CaseResult result{"Add_Fp16_Bf16_Alpha1_CastPath", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.0F, -4.0F, 6.0F};
    std::vector<float> otherFp32 = {2.0F, 1.0F, -1.5F, 0.5F, 2.5F, -3.0F};
    std::vector<uint16_t> selfFp16(selfFp32.size(), 0);
    std::vector<uint16_t> otherBf16(otherFp32.size(), 0);
    for (size_t i = 0; i < selfFp16.size(); ++i) {
        selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
    }
    for (size_t i = 0; i < otherBf16.size(); ++i) {
        otherBf16[i] = FloatToBf16Bits(otherFp32[i]);
    }
    std::vector<float> outHost(selfFp32.size(), 0.0F);

    int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherBf16, shape, ACL_BF16, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            Fp16ToDoubleVector(selfFp16), shape, Bf16ToDoubleVector(otherBf16), shape, shape, static_cast<double>(alphaValue));
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddBoolAlphaTrue(aclrtStream stream)
{
    CaseResult result{"Add_Bool_AlphaTrue", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 4};
    std::vector<uint8_t> selfHost = {1, 0, 1, 1, 0, 1, 0, 0};
    std::vector<uint8_t> otherHost = {1, 1, 0, 1, 0, 1, 1, 0};
    std::vector<uint8_t> outHost(selfHost.size(), 0);

    int ret = CreateAclTensor(selfHost, shape, ACL_BOOL, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_BOOL, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_BOOL, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    bool alphaValue = true;
    ret = CreateAclScalar(alphaValue, ACL_BOOL, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "run add failed";
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<uint8_t> expected(outHost.size(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        uint8_t value = static_cast<uint8_t>(selfHost[i] + static_cast<uint8_t>(alphaValue) * otherHost[i]);
        expected[i] = static_cast<uint8_t>(value != 0);
    }

    std::string err;
    result.pass = CheckExactResult(outHost, expected, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddFloatNonNdFormat(aclrtStream stream)
{
    return RunAddFloatCase("Add_Float32_NonNdFormat",
                                                 stream,
                                                 {1, 1, 2, 3},
                                                 {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                                 {1, 1, 2, 3},
                                                 {2.0F, 1.0F, -1.0F, 0.5F, 0.0F, 2.0F},
                                                 {1, 1, 2, 3},
                                                 1.0F,
                                                 1e-5,
                                                 1e-5,
                                                 ACL_FORMAT_NCHW);
}

CaseResult CaseAddEmptyTensor(aclrtStream stream)
{
    CaseResult result{"Add_EmptyTensor", false, ""};
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 0, 3};
    std::vector<float> emptyData;

    int ret = CreateAclTensor(emptyData, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(emptyData, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(emptyData, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdd(self.tensor, other.tensor, alpha.scalar, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "run add failed";
        return result;
    }

    std::vector<float> outHost;
    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    result.pass = outHost.empty();
    if (!result.pass) {
        result.detail = "empty output expected, got size=" + std::to_string(outHost.size());
    }
    return result;
}

CaseResult CaseAddsFloat(aclrtStream stream)
{
    CaseResult result{"Adds_Float32", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, -1.0F, -2.0F, 4.0F};
    std::vector<float> outHost(selfHost.size(), 0.0F);

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float otherValue = 2.5F;
    float alphaValue = -0.5F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfHost[i]) + static_cast<double>(alphaValue) * static_cast<double>(otherValue);
    }

    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddsFloatAlphaZero(aclrtStream stream)
{
    CaseResult result{"Adds_Float32_AlphaZero", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 2, 3};
    std::vector<float> selfHost = {1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F,
                                   7.0F, -8.0F, 9.0F, -10.0F, 11.0F, -12.0F};
    std::vector<float> outHost(selfHost.size(), 0.0F);
    const std::vector<float> expectedHost = selfHost;

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float otherValue = 123.5F;
    float alphaValue = 0.0F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::string err;
    result.pass = CheckFloatResult(outHost, ToDoubleVector(expectedHost), 1e-6, 1e-6, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddsFloatPrecisionLargePlusTiny(aclrtStream stream)
{
    CaseResult result{"Adds_Float32_Precision_LargePlusTiny", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {6};
    std::vector<float> selfHost = {1.0e10F, -1.0e10F, 16777216.0F, -16777216.0F, 100000.0F, -100000.0F};
    std::vector<float> outHost(selfHost.size(), 0.0F);

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float otherValue = 1.0e-3F;
    float alphaValue = 1.0F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfHost[i]) + static_cast<double>(alphaValue) * static_cast<double>(otherValue);
    }

    result = MakeFloatCompareResult("Adds_Float32_Precision_LargePlusTiny", outHost, expected, 2.0, 1e-6, 1e-4, 1e-6);
    result.detail = "scenario=scalar_add_large_plus_tiny; " + result.detail;
    return result;
}

CaseResult CaseAddsFloatNonNdFormat(aclrtStream stream)
{
    CaseResult result{"Adds_Float32_NonNdFormat", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {1, 1, 2, 3};
    std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, -1.0F, -2.0F, 4.0F};
    std::vector<float> outHost(selfHost.size(), 0.0F);

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self, ACL_FORMAT_NCHW);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out, ACL_FORMAT_NCHW);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float otherValue = 1.5F;
    float alphaValue = 0.5F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfHost[i]) + static_cast<double>(alphaValue) * static_cast<double>(otherValue);
    }

    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddsInt64(aclrtStream stream)
{
    CaseResult result{"Adds_Int64", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<int64_t> selfHost = {1, -2, 3, -4, 5, -6};
    std::vector<int64_t> outHost(selfHost.size(), 0);

    int ret = CreateAclTensor(selfHost, shape, ACL_INT64, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_INT64, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    int64_t otherValue = 2;
    int64_t alphaValue = 3;
    ret = CreateAclScalar(otherValue, ACL_INT64, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_INT64, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<int64_t> expected(outHost.size(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = selfHost[i] + alphaValue * otherValue;
    }

    std::string err;
    result.pass = CheckExactResult(outHost, expected, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddsDoubleMulPathAiCpu(aclrtStream stream)
{
    CaseResult result{"Adds_Double_MulPath_AiCpu", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 2};
    std::vector<double> selfHost = {1.0, -2.0, 3.5, 4.0};
    std::vector<double> outHost(selfHost.size(), 0.0);

    int ret = CreateAclTensor(selfHost, shape, ACL_DOUBLE, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_DOUBLE, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    double otherValue = 2.0;
    double alphaValue = -0.25;
    ret = CreateAclScalar(otherValue, ACL_DOUBLE, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_DOUBLE, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    bool ok = true;
    std::string err;
    for (size_t i = 0; i < outHost.size(); ++i) {
        double expected = selfHost[i] + alphaValue * otherValue;
        if (!IsClose(outHost[i], expected, 1e-12, 1e-12)) {
            ok = false;
            err = "index=" + std::to_string(i) + " actual=" + std::to_string(outHost[i]) +
                  " expected=" + std::to_string(expected);
            break;
        }
    }

    result.pass = ok;
    result.detail = err;
    return result;
}

CaseResult CaseAddsBoolSpecial(aclrtStream stream)
{
    CaseResult result{"Adds_Bool_SpecialCast", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<uint8_t> selfHost = {1, 0, 0, 1, 0, 1};
    std::vector<float> outHost(selfHost.size(), 0.0F);

    int ret = CreateAclTensor(selfHost, shape, ACL_BOOL, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    bool otherValue = true;
    bool alphaValue = true;
    ret = CreateAclScalar(otherValue, ACL_BOOL, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_BOOL, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    // aclnnAdds has a dedicated bool->non-bool cast path to avoid true + true becoming 2.
    std::vector<double> expected(outHost.size(), 1.0);
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-6, 1e-6, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddsEmptyTensor(aclrtStream stream)
{
    CaseResult result{"Adds_EmptyTensor", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {1, 0, 5};
    std::vector<int32_t> emptyData;

    int ret = CreateAclTensor(emptyData, shape, ACL_INT32, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(emptyData, shape, ACL_INT32, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    int32_t otherValue = 2;
    int32_t alphaValue = 2;
    ret = CreateAclScalar(otherValue, ACL_INT32, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_INT32, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "run adds failed";
        return result;
    }

    std::vector<int32_t> outHost;
    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    result.pass = outHost.empty();
    if (!result.pass) {
        result.detail = "empty output expected, got size=" + std::to_string(outHost.size());
    }
    return result;
}

CaseResult CaseAddsFp16KeepFp16(aclrtStream stream)
{
    CaseResult result{"Adds_Fp16_KeepFp16", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.0F, -4.0F, 6.0F};
    std::vector<uint16_t> selfFp16(selfFp32.size(), 0);
    std::vector<uint16_t> outFp16(selfFp32.size(), 0);
    for (size_t i = 0; i < selfFp16.size(); ++i) {
        selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
    }

    int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outFp16, shape, ACL_FLOAT16, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float otherValue = 0.5F;
    float alphaValue = 1.0F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outFp16);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outFp16.size(), 0.0);
    auto selfD = Fp16ToDoubleVector(selfFp16);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = selfD[i] + static_cast<double>(alphaValue) * static_cast<double>(otherValue);
    }

    std::string err;
    result.pass = CheckFp16Result(outFp16, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddsFp16PromoteFloat(aclrtStream stream)
{
    CaseResult result{"Adds_Fp16_PromoteFloat", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.0F, -4.0F, 6.0F};
    std::vector<uint16_t> selfFp16(selfFp32.size(), 0);
    std::vector<float> outHost(selfFp32.size(), 0.0F);
    for (size_t i = 0; i < selfFp16.size(); ++i) {
        selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
    }

    int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float otherValue = 0.1F;
    float alphaValue = 0.2F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    auto selfD = Fp16ToDoubleVector(selfFp16);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = selfD[i] + static_cast<double>(alphaValue) * static_cast<double>(otherValue);
    }

    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddsBf16KeepBf16(aclrtStream stream)
{
    CaseResult result{"Adds_Bf16_KeepBf16", false, ""};
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.0F, -4.0F, 6.0F};
    std::vector<uint16_t> selfBf16(selfFp32.size(), 0);
    std::vector<uint16_t> outBf16(selfFp32.size(), 0);
    for (size_t i = 0; i < selfBf16.size(); ++i) {
        selfBf16[i] = FloatToBf16Bits(selfFp32[i]);
    }

    int ret = CreateAclTensor(selfBf16, shape, ACL_BF16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(outBf16, shape, ACL_BF16, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float otherValue = 0.5F;
    float alphaValue = 1.0F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAdds(self.tensor, other.scalar, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outBf16);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outBf16.size(), 0.0);
    auto selfD = Bf16ToDoubleVector(selfBf16);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = selfD[i] + static_cast<double>(alphaValue) * static_cast<double>(otherValue);
    }

    std::string err;
    result.pass = CheckBf16Result(outBf16, expected, 1e-2, 1e-2, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddFloatBroadcast(aclrtStream stream)
{
    CaseResult result{"InplaceAdd_Float32_Broadcast", false, ""};
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {3};
    std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> otherHost = {2.0F, -1.0F, 0.5F};

    int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    float alphaValue = -1.5F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAdd(self.tensor, other.tensor, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(self, &selfHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy self failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            selfShape,
            ToDoubleVector(otherHost),
            otherShape,
            selfShape,
            static_cast<double>(alphaValue));
    std::string err;
    result.pass = CheckFloatResult(selfHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddFloatAlphaZero(aclrtStream stream)
{
    CaseResult result{"InplaceAdd_Float32_AlphaZero", false, ""};
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> selfShape = {2, 2, 3};
    std::vector<int64_t> otherShape = {3};
    std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F,
                                   4.0F, 5.0F, 6.0F, -4.0F, -5.0F, -6.0F};
    std::vector<float> otherHost = {100.0F, -200.0F, 300.0F};
    const std::vector<float> expectedHost = selfHost;

    int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    float alphaValue = 0.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAdd(self.tensor, other.tensor, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(self, &selfHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy self failed";
        return result;
    }

    std::string err;
    result.pass = CheckFloatResult(selfHost, ToDoubleVector(expectedHost), 1e-6, 1e-6, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddFloatPrecisionCancellation(aclrtStream stream)
{
    CaseResult result{"InplaceAdd_Float32_Precision_Cancellation", false, ""};
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {6};
    std::vector<float> selfHost = {1.0e10F, -1.0e10F, 100000.0F, -100000.0F, 1.0000001F, -1.0000001F};
    std::vector<float> otherHost = {1.0e-3F, -1.0e-3F, -1000000.0F, 1000000.0F, -1.0F, 1.0F};
    const std::vector<float> selfBefore = selfHost;

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    float alphaValue = 0.1F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAdd(self.tensor, other.tensor, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(self, &selfHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy self failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            ToDoubleVector(selfBefore), shape, ToDoubleVector(otherHost), shape, shape, static_cast<double>(alphaValue));
    result = MakeFloatCompareResult(
            "InplaceAdd_Float32_Precision_Cancellation", selfHost, expected, 5e-3, 5e-6, 1e-4, 1e-5);
    result.detail = "scenario=inplace_cancellation_with_alpha; " + result.detail;
    return result;
}

CaseResult CaseInplaceAddsFloat(aclrtStream stream)
{
    CaseResult result{"InplaceAdds_Float32", false, ""};
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, -1.0F, -2.0F, 4.0F};

    int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }

    float otherValue = -0.5F;
    float alphaValue = 2.0F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAdds(self.tensor, other.scalar, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAdds",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(self, &selfHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy self failed";
        return result;
    }

    std::vector<double> expected(selfHost.size(), 0.0);
    const std::vector<double> selfBefore = {1.0, 2.0, 3.0, -1.0, -2.0, 4.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = selfBefore[i] + static_cast<double>(alphaValue) * static_cast<double>(otherValue);
    }
    std::string err;
    result.pass = CheckFloatResult(selfHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddsBool(aclrtStream stream)
{
    CaseResult result{"InplaceAdds_Bool", false, ""};
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 4};
    std::vector<uint8_t> selfHost = {1, 0, 1, 1, 0, 1, 0, 0};
    std::vector<uint8_t> expected = selfHost;

    int ret = CreateAclTensor(selfHost, shape, ACL_BOOL, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }

    bool otherValue = false;
    bool alphaValue = true;
    ret = CreateAclScalar(otherValue, ACL_BOOL, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_BOOL, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAdds(self.tensor, other.scalar, alpha.scalar, stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "run inplace adds failed";
        return result;
    }

    ret = CopyTensorToHost(self, &selfHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy self failed";
        return result;
    }

    std::string err;
    result.pass = CheckExactResult(selfHost, expected, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddBoolBroadcastKeep(aclrtStream stream)
{
    CaseResult result{"InplaceAdd_Bool_BroadcastKeep", false, ""};
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> selfShape = {2, 4};
    std::vector<int64_t> otherShape = {4};
    std::vector<uint8_t> selfHost = {1, 0, 1, 1, 0, 1, 0, 0};
    std::vector<uint8_t> otherHost = {0, 0, 0, 0};
    const std::vector<uint8_t> expected = selfHost;

    int ret = CreateAclTensor(selfHost, selfShape, ACL_BOOL, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherHost, otherShape, ACL_BOOL, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    bool alphaValue = true;
    ret = CreateAclScalar(alphaValue, ACL_BOOL, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAdd(self.tensor, other.tensor, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(self, &selfHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy self failed";
        return result;
    }

    std::string err;
    result.pass = CheckExactResult(selfHost, expected, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddFp16Broadcast(aclrtStream stream)
{
    CaseResult result{"InplaceAdd_Fp16_Broadcast", false, ""};
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {3};
    std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.0F, -4.0F, 6.0F};
    std::vector<float> otherFp32 = {0.25F, -1.0F, 2.0F};
    std::vector<uint16_t> selfFp16(selfFp32.size(), 0);
    std::vector<uint16_t> otherFp16(otherFp32.size(), 0);
    for (size_t i = 0; i < selfFp16.size(); ++i) {
        selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
    }
    for (size_t i = 0; i < otherFp16.size(); ++i) {
        otherFp16[i] = FloatToFp16Bits(otherFp32[i]);
    }
    const auto selfBefore = selfFp16;

    int ret = CreateAclTensor(selfFp16, selfShape, ACL_FLOAT16, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self failed";
        return result;
    }
    ret = CreateAclTensor(otherFp16, otherShape, ACL_FLOAT16, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    float alphaValue = 0.5F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAdd(self.tensor, other.tensor, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAdd",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(self, &selfFp16);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy self failed";
        return result;
    }

    auto expected = ComputeExpectedBroadcastAdd(
            Fp16ToDoubleVector(selfBefore),
            selfShape,
            Fp16ToDoubleVector(otherFp16),
            otherShape,
            selfShape,
            static_cast<double>(alphaValue));

    std::string err;
    result.pass = CheckFp16Result(selfFp16, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddV3FloatAlpha1(aclrtStream stream)
{
    CaseResult result{"AddV3_Float32_Alpha1", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<float> otherHost = {2.0F, -1.0F, 8.0F, 0.5F};
    std::vector<float> outHost(shape[0], 0.0F);

    int ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float selfValue = 1.25F;
    float alphaValue = 1.0F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfValue) + static_cast<double>(alphaValue) * static_cast<double>(otherHost[i]);
    }
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddV3FloatAlphaNeg(aclrtStream stream)
{
    CaseResult result{"AddV3_Float32_AlphaNeg", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<float> otherHost = {2.0F, -1.0F, 8.0F, 0.5F};
    std::vector<float> outHost(shape[0], 0.0F);

    int ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float selfValue = -1.0F;
    float alphaValue = -0.25F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfValue) + static_cast<double>(alphaValue) * static_cast<double>(otherHost[i]);
    }
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddV3FloatPrecisionCancellation(aclrtStream stream)
{
    CaseResult result{"AddV3_Float32_Precision_Cancellation", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {6};
    std::vector<float> otherHost = {1.0e10F, -1.0e10F, -3.0F, 3.0F, 16777216.0F, -16777216.0F};
    std::vector<float> outHost(shape[0], 0.0F);

    int ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float selfValue = 1.0F;
    float alphaValue = 0.33333334F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfValue) + static_cast<double>(alphaValue) * static_cast<double>(otherHost[i]);
    }

    result = MakeFloatCompareResult("AddV3_Float32_Precision_Cancellation", outHost, expected, 2.0, 1e-7, 1e-4, 1e-6);
    result.detail = "scenario=v3_scalar_plus_alpha_tensor; " + result.detail;
    return result;
}

CaseResult CaseAddV3Fp16AlphaHalf(aclrtStream stream)
{
    CaseResult result{"AddV3_Fp16_AlphaHalf", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> otherFp32 = {2.0F, -1.5F, 8.0F, 0.25F, -4.0F, 0.5F};
    std::vector<uint16_t> otherFp16(otherFp32.size(), 0);
    std::vector<uint16_t> outFp16(otherFp32.size(), 0);
    for (size_t i = 0; i < otherFp16.size(); ++i) {
        otherFp16[i] = FloatToFp16Bits(otherFp32[i]);
    }

    int ret = CreateAclTensor(otherFp16, shape, ACL_FLOAT16, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outFp16, shape, ACL_FLOAT16, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float selfValue = 1.25F;
    float alphaValue = 0.5F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outFp16);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outFp16.size(), 0.0);
    auto otherD = Fp16ToDoubleVector(otherFp16);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfValue) + static_cast<double>(alphaValue) * otherD[i];
    }
    std::string err;
    result.pass = CheckFp16Result(outFp16, expected, 1e-3, 1e-3, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddV3Bf16AlphaNeg(aclrtStream stream)
{
    CaseResult result{"AddV3_Bf16_AlphaNeg", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> otherFp32 = {1.0F, -2.5F, 0.5F, 3.0F, -8.0F, 16.0F};
    std::vector<uint16_t> otherBf16(otherFp32.size(), 0);
    std::vector<uint16_t> outBf16(otherFp32.size(), 0);
    for (size_t i = 0; i < otherBf16.size(); ++i) {
        otherBf16[i] = FloatToBf16Bits(otherFp32[i]);
    }

    int ret = CreateAclTensor(otherBf16, shape, ACL_BF16, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outBf16, shape, ACL_BF16, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float selfValue = -2.0F;
    float alphaValue = -0.25F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outBf16);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outBf16.size(), 0.0);
    auto otherD = Bf16ToDoubleVector(otherBf16);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfValue) + static_cast<double>(alphaValue) * otherD[i];
    }
    std::string err;
    result.pass = CheckBf16Result(outBf16, expected, 1e-2, 1e-2, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddV3SelfFloatOtherInt32(aclrtStream stream)
{
    CaseResult result{"AddV3_SelfFloat_OtherInt32", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<int32_t> otherHost = {2, -1, 8, 0};
    std::vector<float> outHost(shape[0], 0.0F);

    int ret = CreateAclTensor(otherHost, shape, ACL_INT32, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float selfValue = 1.25F;
    float alphaValue = 1.0F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected(outHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfValue) + static_cast<double>(alphaValue) * static_cast<double>(otherHost[i]);
    }
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddV3EmptyTensor(aclrtStream stream)
{
    CaseResult result{"AddV3_EmptyTensor", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {0};
    std::vector<float> otherHost;
    std::vector<float> outHost;

    int ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    float selfValue = 1.0F;
    float alphaValue = 2.0F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<double> expected;
    std::string err;
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseAddV3Int8MulAdd(aclrtStream stream)
{
    CaseResult result{"AddV3_Int8_MulAddPath", false, ""};
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<int8_t> otherHost = {-2, 1, 4, -1};
    std::vector<int8_t> outHost(shape[0], 0);

    int ret = CreateAclTensor(otherHost, shape, ACL_INT8, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }
    ret = CreateAclTensor(outHost, shape, ACL_INT8, &out);
    if (ret != ACL_SUCCESS) {
        result.detail = "create out failed";
        return result;
    }

    int8_t selfValue = 3;
    int8_t alphaValue = 2;
    ret = CreateAclScalar(selfValue, ACL_INT8, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_INT8, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunAddV3(self.scalar, other.tensor, alpha.scalar, out.tensor, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(out, &outHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy out failed";
        return result;
    }

    std::vector<int8_t> expected(outHost.size(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<int8_t>(selfValue + alphaValue * otherHost[i]);
    }

    std::string err;
    result.pass = CheckExactResult(outHost, expected, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddV3Float(aclrtStream stream)
{
    CaseResult result{"InplaceAddV3_Float32", false, ""};
    TensorHolder other;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {4};
    std::vector<float> otherHost = {2.0F, -1.0F, 8.0F, 0.5F};
    const std::vector<float> before = otherHost;

    int ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    float selfValue = 1.5F;
    float alphaValue = 2.0F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAddV3(self.scalar, other.tensor, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(other, &otherHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy other failed";
        return result;
    }

    std::vector<double> expected(otherHost.size(), 0.0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(selfValue) + static_cast<double>(alphaValue) * static_cast<double>(before[i]);
    }
    std::string err;
    result.pass = CheckFloatResult(otherHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
    return result;
}

CaseResult CaseInplaceAddV3Int8MulAdd(aclrtStream stream)
{
    CaseResult result{"InplaceAddV3_Int8_MulAddPath", false, ""};
    TensorHolder other;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {6};
    std::vector<int8_t> otherHost = {-2, 1, 4, -1, 8, -8};
    const std::vector<int8_t> before = otherHost;

    int ret = CreateAclTensor(otherHost, shape, ACL_INT8, &other);
    if (ret != ACL_SUCCESS) {
        result.detail = "create other failed";
        return result;
    }

    int8_t selfValue = 3;
    int8_t alphaValue = 2;
    ret = CreateAclScalar(selfValue, ACL_INT8, &self);
    if (ret != ACL_SUCCESS) {
        result.detail = "create self scalar failed";
        return result;
    }
    ret = CreateAclScalar(alphaValue, ACL_INT8, &alpha);
    if (ret != ACL_SUCCESS) {
        result.detail = "create alpha failed";
        return result;
    }

    ret = RunInplaceAddV3(self.scalar, other.tensor, alpha.scalar, stream);
    if (TryWorkspaceOnlyOnRunFail(
            &result,
            "aclnnInplaceAddV3",
            ret,
            [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, workspaceSize, executor);
            })) {
        return result;
    }

    ret = CopyTensorToHost(other, &otherHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy other failed";
        return result;
    }

    std::vector<int8_t> expected(otherHost.size(), 0);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<int8_t>(selfValue + alphaValue * before[i]);
    }

    std::string err;
    result.pass = CheckExactResult(otherHost, expected, &err);
    result.detail = err;
    return result;
}

CaseResult CaseNegAddNullPtr(aclrtStream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_Add_Nullptr", status);
}

CaseResult CaseNegAddShapeMismatch(aclrtStream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<float> selfHost(6, 1.0F);
    std::vector<float> otherHost(4, 2.0F);
    std::vector<float> outHost(6, 0.0F);

    int ret = CreateAclTensor(selfHost, {2, 3}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_ShapeMismatch", false, "create self failed"};
    }
    ret = CreateAclTensor(otherHost, {2, 2}, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_ShapeMismatch", false, "create other failed"};
    }
    ret = CreateAclTensor(outHost, {2, 3}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_ShapeMismatch", false, "create out failed"};
    }
    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_ShapeMismatch", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_Add_ShapeMismatch", status);
}

CaseResult CaseNegAddRankTooLarge(aclrtStream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> host = {1.0F};

    int ret = CreateAclTensor(host, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_RankTooLarge", false, "create self failed"};
    }
    ret = CreateAclTensor(host, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_RankTooLarge", false, "create other failed"};
    }
    ret = CreateAclTensor(host, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_RankTooLarge", false, "create out failed"};
    }
    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_RankTooLarge", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_Add_RankTooLarge", status);
}

CaseResult CaseNegAddInvalidDtype(aclrtStream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 2};
    std::vector<uint32_t> host = {1, 2, 3, 4};

    int ret = CreateAclTensor(host, shape, ACL_UINT32, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_InvalidDtype", false, "create self failed"};
    }
    ret = CreateAclTensor(host, shape, ACL_UINT32, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_InvalidDtype", false, "create other failed"};
    }
    ret = CreateAclTensor(host, shape, ACL_UINT32, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_InvalidDtype", false, "create out failed"};
    }

    int32_t alphaValue = 1;
    ret = CreateAclScalar(alphaValue, ACL_INT32, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_InvalidDtype", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_Add_InvalidDtype", status);
}

CaseResult CaseNegInplaceAddInvalidBroadcast(aclrtStream)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    std::vector<float> selfHost = {1.0F, 2.0F};
    std::vector<float> otherHost = {1.0F, 2.0F, 3.0F, 4.0F};

    int ret = CreateAclTensor(selfHost, {2, 1}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_InplaceAdd_InvalidBroadcast", false, "create self failed"};
    }
    ret = CreateAclTensor(otherHost, {2, 2}, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_InplaceAdd_InvalidBroadcast", false, "create other failed"};
    }
    float alphaValue = 1.0F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_InplaceAdd_InvalidBroadcast", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_InplaceAdd_InvalidBroadcast", status);
}

CaseResult CaseNegAddsNullAlpha(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;

    std::vector<int64_t> shape = {2, 3};
    std::vector<float> host = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

    int ret = CreateAclTensor(host, shape, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Adds_NullAlpha", false, "create self failed"};
    }
    ret = CreateAclTensor(host, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Adds_NullAlpha", false, "create out failed"};
    }

    float otherValue = 1.0F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Adds_NullAlpha", false, "create other scalar failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, nullptr, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_Adds_NullAlpha", status);
}

CaseResult CaseNegAddsShapeMismatch(aclrtStream)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;

    std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> outHost = {0.0F, 0.0F, 0.0F, 0.0F};

    int ret = CreateAclTensor(selfHost, {2, 3}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Adds_ShapeMismatch", false, "create self failed"};
    }
    ret = CreateAclTensor(outHost, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Adds_ShapeMismatch", false, "create out failed"};
    }

    float otherValue = 1.0F;
    float alphaValue = 1.0F;
    ret = CreateAclScalar(otherValue, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Adds_ShapeMismatch", false, "create other scalar failed"};
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Adds_ShapeMismatch", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_Adds_ShapeMismatch", status);
}

CaseResult CaseNegAddBoolAlphaFloat(aclrtStream)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    std::vector<int64_t> shape = {2, 2};
    std::vector<uint8_t> selfHost = {1, 0, 1, 0};
    std::vector<uint8_t> otherHost = {0, 1, 1, 0};
    std::vector<uint8_t> outHost = {0, 0, 0, 0};

    int ret = CreateAclTensor(selfHost, shape, ACL_BOOL, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_BoolAlphaFloat", false, "create self failed"};
    }
    ret = CreateAclTensor(otherHost, shape, ACL_BOOL, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_BoolAlphaFloat", false, "create other failed"};
    }
    ret = CreateAclTensor(outHost, shape, ACL_BOOL, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_BoolAlphaFloat", false, "create out failed"};
    }

    float alphaValue = 0.5F;
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_Add_BoolAlphaFloat", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_Add_BoolAlphaFloat", status);
}

CaseResult CaseNegAddV3NullPtr(aclrtStream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddV3GetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_AddV3_Nullptr", status);
}

CaseResult CaseNegAddV3ShapeMismatch(aclrtStream)
{
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<float> otherHost(6, 1.0F);
    std::vector<float> outHost(6, 0.0F);

    int ret = CreateAclTensor(otherHost, {2, 3}, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_ShapeMismatch", false, "create other failed"};
    }
    ret = CreateAclTensor(outHost, {3, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_ShapeMismatch", false, "create out failed"};
    }

    float selfValue = 1.0F;
    float alphaValue = 1.0F;
    ret = CreateAclScalar(selfValue, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_ShapeMismatch", false, "create self scalar failed"};
    }
    ret = CreateAclScalar(alphaValue, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_ShapeMismatch", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_AddV3_ShapeMismatch", status);
}

CaseResult CaseNegAddV3UnsupportedDtype(aclrtStream)
{
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;

    std::vector<uint32_t> host = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};

    int ret = CreateAclTensor(host, shape, ACL_UINT32, &other);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_UnsupportedDtype", false, "create other failed"};
    }
    ret = CreateAclTensor(host, shape, ACL_UINT32, &out);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_UnsupportedDtype", false, "create out failed"};
    }

    int32_t selfValue = 1;
    int32_t alphaValue = 1;
    ret = CreateAclScalar(selfValue, ACL_INT32, &self);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_UnsupportedDtype", false, "create self scalar failed"};
    }
    ret = CreateAclScalar(alphaValue, ACL_INT32, &alpha);
    if (ret != ACL_SUCCESS) {
        return {"NEG_AddV3_UnsupportedDtype", false, "create alpha failed"};
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return MakeStatusNotSuccessResult("NEG_AddV3_UnsupportedDtype", status);
}

struct CaseEntry {
    const char* functionName;
    CaseResult (*fn)(aclrtStream);
};

#define ADD_CASE(fn) CaseEntry{#fn, fn}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const bool enableDoubleAicpuCases = IsEnvEnabled("ENABLE_DOUBLE_AICPU_CASES");
    const bool continueOnSyncFail = GetBoolEnv("CONTINUE_ON_SYNC_FAIL", true);
    LOG_PRINT("[INFO] ACL sync timeout = %u ms\n", g_syncTimeoutMs);
    LOG_PRINT("[INFO] CONTINUE_ON_SYNC_FAIL = %s\n", continueOnSyncFail ? "1" : "0");

    std::vector<CaseResult> results;
    std::vector<CaseEntry> caseFuncs = {
            ADD_CASE(CaseAddFloatBasicAlpha1),
            ADD_CASE(CaseAddFloatBroadcastAlphaNeg),
            ADD_CASE(CaseAddFloatAlphaZeroLarge),
            ADD_CASE(CaseAddFloatBroadcast4DAlphaQuarter),
            ADD_CASE(CaseAddFloatPrecisionStress),
            ADD_CASE(CaseAddFloatPrecisionLargePlusTiny),
            ADD_CASE(CaseAddFloatPrecisionCancellationAlpha),
            ADD_CASE(CaseAddFloatNaNInf),
            ADD_CASE(CaseAddInt32Alpha2),
            ADD_CASE(CaseAddInt64Alpha2),
            ADD_CASE(CaseAddUInt8Alpha3),
            ADD_CASE(CaseAddInt32Alpha1),
            ADD_CASE(CaseAddInt64Alpha1),
            ADD_CASE(CaseAddInt8Alpha1),
            ADD_CASE(CaseAddUInt8Alpha1),
            ADD_CASE(CaseAddDoubleAiCpuProbe),
            ADD_CASE(CaseAddDoubleMulPathAlphaHalf),
            ADD_CASE(CaseAddOutAliasOtherProbe),
            ADD_CASE(CaseAddFp16AlphaHalf),
            ADD_CASE(CaseAddBf16Alpha1),
            ADD_CASE(CaseAddMixFp16Fp32),
            ADD_CASE(CaseAddMixFp32Fp16),
            ADD_CASE(CaseAddMixFp16Fp32BroadcastAlpha1),
            ADD_CASE(CaseAddFp16Bf16Alpha1CastPath),
            ADD_CASE(CaseAddBoolAlphaTrue),
            ADD_CASE(CaseAddFloatNonNdFormat),
            ADD_CASE(CaseAddEmptyTensor),
            ADD_CASE(CaseAddsFloat),
            ADD_CASE(CaseAddsFloatAlphaZero),
            ADD_CASE(CaseAddsFloatPrecisionLargePlusTiny),
            ADD_CASE(CaseAddsFloatNonNdFormat),
            ADD_CASE(CaseAddsInt64),
            ADD_CASE(CaseAddsDoubleMulPathAiCpu),
            ADD_CASE(CaseAddsBoolSpecial),
            ADD_CASE(CaseAddsEmptyTensor),
            ADD_CASE(CaseAddsFp16KeepFp16),
            ADD_CASE(CaseAddsFp16PromoteFloat),
            ADD_CASE(CaseAddsBf16KeepBf16),
            ADD_CASE(CaseInplaceAddFloatBroadcast),
            ADD_CASE(CaseInplaceAddFloatAlphaZero),
            ADD_CASE(CaseInplaceAddFloatPrecisionCancellation),
            ADD_CASE(CaseInplaceAddsFloat),
            ADD_CASE(CaseInplaceAddsBool),
            ADD_CASE(CaseInplaceAddBoolBroadcastKeep),
            ADD_CASE(CaseInplaceAddFp16Broadcast),
            ADD_CASE(CaseAddV3FloatAlpha1),
            ADD_CASE(CaseAddV3FloatAlphaNeg),
            ADD_CASE(CaseAddV3FloatPrecisionCancellation),
            ADD_CASE(CaseAddV3Fp16AlphaHalf),
            ADD_CASE(CaseAddV3Bf16AlphaNeg),
            ADD_CASE(CaseAddV3SelfFloatOtherInt32),
            ADD_CASE(CaseAddV3EmptyTensor),
            ADD_CASE(CaseAddV3Int8MulAdd),
            ADD_CASE(CaseInplaceAddV3Float),
            ADD_CASE(CaseInplaceAddV3Int8MulAdd),
            ADD_CASE(CaseNegAddNullPtr),
            ADD_CASE(CaseNegAddShapeMismatch),
            ADD_CASE(CaseNegAddRankTooLarge),
            ADD_CASE(CaseNegAddInvalidDtype),
            ADD_CASE(CaseNegInplaceAddInvalidBroadcast),
            ADD_CASE(CaseNegAddsNullAlpha),
            ADD_CASE(CaseNegAddsShapeMismatch),
            ADD_CASE(CaseNegAddBoolAlphaFloat),
            ADD_CASE(CaseNegAddV3NullPtr),
            ADD_CASE(CaseNegAddV3ShapeMismatch),
            ADD_CASE(CaseNegAddV3UnsupportedDtype),
    };

            if (!enableDoubleAicpuCases) {
            caseFuncs.erase(
                std::remove_if(
                    caseFuncs.begin(),
                    caseFuncs.end(),
                    [](const CaseEntry& entry) {
                        return std::strcmp(entry.functionName, "CaseAddDoubleAiCpuProbe") == 0 ||
                           std::strcmp(entry.functionName, "CaseAddDoubleMulPathAlphaHalf") == 0;
                    }),
                caseFuncs.end());
            LOG_PRINT("[INFO] skip double AICPU probe cases; set ENABLE_DOUBLE_AICPU_CASES=1 to enable.\n");
            }

    auto totalStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < caseFuncs.size(); ++i) {
        const auto& entry = caseFuncs[i];
        LOG_PRINT("[RUN ] %zu/%zu %s\n", i + 1, caseFuncs.size(), entry.functionName);
        fflush(stdout);
        auto start = std::chrono::steady_clock::now();
        CaseResult result = entry.fn(stream);
        auto end = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        PrintCaseResult(result);
        LOG_PRINT("[TIME] %s %lld ms\n", entry.functionName, static_cast<long long>(elapsedMs));
        fflush(stdout);
        results.push_back(result);

        // stream 同步失败后继续复用同一个 stream 风险很高：后续用例可能继续卡住或报级联错误。
        if (!result.pass && result.detail.find("sync stream failed/timeout") != std::string::npos) {
            if (!continueOnSyncFail) {
                LOG_PRINT("[STOP] Stop running remaining cases because stream sync failed/timeout in %s.\n",
                          entry.functionName);
                break;
            }

            LOG_PRINT("[WARN] sync stream failed/timeout in %s, try to recover runtime context and continue.\n",
                      entry.functionName);
            if (!RecoverRuntimeContext(deviceId, &stream)) {
                LOG_PRINT("[STOP] runtime context recovery failed after %s.\n", entry.functionName);
                break;
            }
        }
    }

    int passCount = 0;
    for (const auto& r : results) {
        if (r.pass) {
            ++passCount;
        }
    }
    int failCount = static_cast<int>(results.size()) - passCount;
    auto totalEnd = std::chrono::steady_clock::now();
    auto totalElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();
    LOG_PRINT("\n==== Add Example Summary ====\n");
    LOG_PRINT("Total: %d, Pass: %d, Fail: %d\n", static_cast<int>(results.size()), passCount, failCount);
    LOG_PRINT("Elapsed: %lld ms\n", static_cast<long long>(totalElapsedMs));

    Finalize(deviceId, stream);
    return (failCount == 0) ? 0 : 1;
}

#undef ADD_CASE
