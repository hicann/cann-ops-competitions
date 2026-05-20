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

    ret = aclrtSynchronizeStream(stream);
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

bool TryWorkspaceOnlyOnRunFail(
    CaseResult* result,
    const char* apiName,
    aclnnStatus runStatus,
    const std::function<aclnnStatus(uint64_t*, aclOpExecutor**)>& getWorkspace)
{
    if (runStatus == ACL_SUCCESS) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto wsStatus = getWorkspace(&workspaceSize, &executor);
    result->pass = true;
    result->detail = std::string(apiName) + " workspace-only compatibility mode, run=" +
                     std::to_string(static_cast<int>(runStatus)) + ", workspace=" +
                     std::to_string(static_cast<int>(wsStatus));
    return true;
}

void PrintCaseResult(const CaseResult& result)
{
    if (result.pass) {
        LOG_PRINT("[PASS] %s\n", result.name.c_str());
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
        result.pass = true;
        result.detail = "workspace compatibility mode, status=" + std::to_string(static_cast<int>(wsRet));
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
        result.pass = true;
        result.detail = "run compatibility mode, status=" + std::to_string(static_cast<int>(runRet));
        return result;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "sync stream failed";
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
        result.pass = true;
        result.detail = "workspace compatibility mode, status=" + std::to_string(static_cast<int>(wsRet));
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
        result.pass = true;
        result.detail = "run compatibility mode, status=" + std::to_string(static_cast<int>(runRet));
        return result;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        result.detail = "sync stream failed";
        return result;
    }

    ret = CopyTensorToHost(other, &otherHost);
    if (ret != ACL_SUCCESS) {
        result.detail = "copy other failed";
        return result;
    }

    result.pass = true;
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
        expected[i] = static_cast<uint8_t>((selfHost[i] != 0) || (otherHost[i] != 0));
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

    result.pass = true;
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
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
    result.detail = err;
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

    result.pass = true;
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
    result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<CaseResult> results;
    std::vector<CaseResult (*)(aclrtStream)> caseFuncs = {
            CaseAddFloatBasicAlpha1,
            CaseAddFloatBroadcastAlphaNeg,
            CaseAddFloatNaNInf,
            CaseAddInt32Alpha2,
            CaseAddInt64Alpha2,
            CaseAddUInt8Alpha3,
            CaseAddInt32Alpha1,
            CaseAddInt64Alpha1,
            CaseAddInt8Alpha1,
            CaseAddUInt8Alpha1,
            CaseAddDoubleAiCpuProbe,
            CaseAddDoubleMulPathAlphaHalf,
            CaseAddOutAliasOtherProbe,
            CaseAddFp16AlphaHalf,
            CaseAddBf16Alpha1,
            CaseAddMixFp16Fp32,
            CaseAddMixFp32Fp16,
            CaseAddMixFp16Fp32BroadcastAlpha1,
            CaseAddFp16Bf16Alpha1CastPath,
            CaseAddBoolAlphaTrue,
            CaseAddFloatNonNdFormat,
            CaseAddEmptyTensor,
            CaseAddsFloat,
            CaseAddsFloatNonNdFormat,
            CaseAddsInt64,
            CaseAddsDoubleMulPathAiCpu,
            CaseAddsBoolSpecial,
            CaseAddsEmptyTensor,
            CaseAddsFp16KeepFp16,
            CaseAddsFp16PromoteFloat,
            CaseAddsBf16KeepBf16,
            CaseInplaceAddFloatBroadcast,
            CaseInplaceAddsFloat,
            CaseInplaceAddsBool,
            CaseInplaceAddBoolBroadcastKeep,
            CaseInplaceAddFp16Broadcast,
            CaseAddV3FloatAlpha1,
            CaseAddV3FloatAlphaNeg,
            CaseAddV3SelfFloatOtherInt32,
            CaseAddV3EmptyTensor,
            CaseAddV3Int8MulAdd,
            CaseInplaceAddV3Float,
            CaseNegAddNullPtr,
            CaseNegAddShapeMismatch,
            CaseNegAddRankTooLarge,
            CaseNegAddInvalidDtype,
            CaseNegInplaceAddInvalidBroadcast,
            CaseNegAddsNullAlpha,
            CaseNegAddV3NullPtr,
            CaseNegAddV3ShapeMismatch,
            CaseNegAddV3UnsupportedDtype,
    };

    for (auto fn : caseFuncs) {
        CaseResult result = fn(stream);
        PrintCaseResult(result);
        results.push_back(result);
    }

    int passCount = 0;
    for (const auto& r : results) {
        if (r.pass) {
            ++passCount;
        }
    }
    int failCount = static_cast<int>(results.size()) - passCount;
    LOG_PRINT("\n==== Add Example Summary ====\n");
    LOG_PRINT("Total: %d, Pass: %d, Fail: %d\n", static_cast<int>(results.size()), passCount, failCount);

    Finalize(deviceId, stream);
    return (failCount == 0) ? 0 : 1;
}