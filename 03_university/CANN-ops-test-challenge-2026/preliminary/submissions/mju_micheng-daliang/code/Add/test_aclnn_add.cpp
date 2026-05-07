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
    } while (0)

constexpr int kAclnnErrParamNullptr = 161001;
constexpr int kAclnnErrParamInvalid = 161002;

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;

    ~TensorResource()
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

struct ScalarResource {
    aclScalar* scalar = nullptr;

    ~ScalarResource()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
            scalar = nullptr;
        }
    }
};

struct WorkspaceResource {
    void* addr = nullptr;

    ~WorkspaceResource()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

static inline float Float16ToFloat(uint16_t value)
{
    unsigned int sign = (value >> 15) & 0x1;
    unsigned int exponent = (value >> 10) & 0x1f;
    unsigned int mantissa = value & 0x3ff;

    float result = 0.0f;
    if (exponent == 0) {
        result = mantissa * 0.0000019073486328125f;
    } else if (exponent == 31) {
        result = mantissa == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        result = (1.0f + mantissa * 0.0009765625f) * std::pow(2.0f, static_cast<int>(exponent) - 15);
    }
    return sign ? -result : result;
}

static inline uint16_t FloatToFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;

    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return sign | 0x7c00;
    }
    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

template <typename T>
double ToDoubleValue(const T& value)
{
    return static_cast<double>(value);
}

template <>
double ToDoubleValue<uint16_t>(const uint16_t& value)
{
    return static_cast<double>(Float16ToFloat(value));
}

template <typename T>
std::vector<double> ToDoubleVector(const std::vector<T>& input)
{
    std::vector<double> output;
    output.reserve(input.size());
    for (const auto& value : input) {
        output.push_back(ToDoubleValue(value));
    }
    return output;
}

bool AlmostEqual(double expected, double actual, double atol, double rtol)
{
    if (std::isnan(expected) && std::isnan(actual)) {
        return true;
    }
    if (std::isinf(expected) || std::isinf(actual)) {
        return std::isinf(expected) && std::isinf(actual) && ((expected > 0) == (actual > 0));
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

std::vector<double> ComputeBroadcastBinary(const std::vector<double>& lhsData,
                                           const std::vector<int64_t>& lhsShape,
                                           const std::vector<double>& rhsData,
                                           const std::vector<int64_t>& rhsShape,
                                           const std::vector<int64_t>& outShape,
                                           double alpha)
{
    const auto lhsStrides = GetStrides(lhsShape);
    const auto rhsStrides = GetStrides(rhsShape);
    const auto outStrides = GetStrides(outShape);
    const size_t outRank = outShape.size();
    const size_t lhsOffset = outRank - lhsShape.size();
    const size_t rhsOffset = outRank - rhsShape.size();
    const int64_t outSize = GetShapeSize(outShape);

    std::vector<double> expected(outSize, 0.0);
    for (int64_t flatIndex = 0; flatIndex < outSize; ++flatIndex) {
        int64_t remain = flatIndex;
        std::vector<int64_t> outIndex(outRank, 0);
        for (size_t dim = 0; dim < outRank; ++dim) {
            outIndex[dim] = remain / outStrides[dim];
            remain %= outStrides[dim];
        }

        int64_t lhsFlatIndex = 0;
        for (size_t dim = 0; dim < lhsShape.size(); ++dim) {
            int64_t idx = lhsShape[dim] == 1 ? 0 : outIndex[dim + lhsOffset];
            lhsFlatIndex += idx * lhsStrides[dim];
        }

        int64_t rhsFlatIndex = 0;
        for (size_t dim = 0; dim < rhsShape.size(); ++dim) {
            int64_t idx = rhsShape[dim] == 1 ? 0 : outIndex[dim + rhsOffset];
            rhsFlatIndex += idx * rhsStrides[dim];
        }

        expected[flatIndex] = lhsData[lhsFlatIndex] + alpha * rhsData[rhsFlatIndex];
    }
    return expected;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType, TensorResource* resource)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(&resource->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(resource->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    auto strides = GetStrides(shape);
    resource->tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        resource->deviceAddr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CopyDeviceToHost(const TensorResource& resource, std::vector<T>* hostData)
{
    auto size = hostData->size() * sizeof(T);
    auto ret = aclrtMemcpy(hostData->data(), size, resource.deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CheckNumericResult(
    const std::string& caseName, const std::vector<T>& actual, const std::vector<double>& expected, double atol, double rtol)
{
    if (actual.size() != expected.size()) {
        LOG_PRINT("[FAIL] %s size mismatch: actual=%zu expected=%zu\n", caseName.c_str(), actual.size(), expected.size());
        return 1;
    }

    int mismatchCount = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double actualValue = ToDoubleValue(actual[i]);
        if (!AlmostEqual(expected[i], actualValue, atol, rtol)) {
            LOG_PRINT("  mismatch[%zu]: expected=%.8f actual=%.8f\n", i, expected[i], actualValue);
            ++mismatchCount;
        }
    }

    if (mismatchCount == 0) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s mismatches=%d\n", caseName.c_str(), mismatchCount);
    return 1;
}

template <typename T>
int CreateScalar(T value, aclDataType dataType, ScalarResource* resource)
{
    resource->scalar = aclCreateScalar(&value, dataType);
    CHECK_RET(resource->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename TSelf, typename TOther, typename TOut, typename TAlpha>
int RunAddCase(const std::string& caseName,
               const std::vector<TSelf>& selfData,
               const std::vector<int64_t>& selfShape,
               aclDataType selfType,
               const std::vector<TOther>& otherData,
               const std::vector<int64_t>& otherShape,
               aclDataType otherType,
               TAlpha alphaValue,
               aclDataType alphaType,
               const std::vector<int64_t>& outShape,
               aclDataType outType,
               aclrtStream stream,
               double atol,
               double rtol)
{
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<TOut> outData(GetShapeSize(outShape), static_cast<TOut>(0));
    std::vector<double> expected =
        ComputeBroadcastBinary(ToDoubleVector(selfData), selfShape, ToDoubleVector(otherData), otherShape, outShape, ToDoubleValue(alphaValue));

    int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(otherData, otherShape, otherType, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(outData, outShape, outType, &out);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(alphaValue, alphaType, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnAddGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return 1;
        }
    }

    ret = aclnnAdd(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnAdd failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = CopyDeviceToHost(out, &outData);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    return CheckNumericResult(caseName, outData, expected, atol, rtol);
}

template <typename TSelf, typename TOther, typename TAlpha>
int RunInplaceAddCase(const std::string& caseName,
                      const std::vector<TSelf>& selfData,
                      const std::vector<int64_t>& selfShape,
                      aclDataType selfType,
                      const std::vector<TOther>& otherData,
                      const std::vector<int64_t>& otherShape,
                      aclDataType otherType,
                      TAlpha alphaValue,
                      aclDataType alphaType,
                      aclrtStream stream,
                      double atol,
                      double rtol)
{
    TensorResource self;
    TensorResource other;
    ScalarResource alpha;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto actual = selfData;
    std::vector<double> expected =
        ComputeBroadcastBinary(ToDoubleVector(selfData), selfShape, ToDoubleVector(otherData), otherShape, selfShape, ToDoubleValue(alphaValue));

    int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(otherData, otherShape, otherType, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(alphaValue, alphaType, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return 1;
        }
    }

    ret = aclnnInplaceAdd(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceAdd failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = CopyDeviceToHost(self, &actual);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    return CheckNumericResult(caseName, actual, expected, atol, rtol);
}

template <typename TSelf, typename TOther, typename TAlpha, typename TOut>
int RunAddsCase(const std::string& caseName,
                const std::vector<TSelf>& selfData,
                const std::vector<int64_t>& shape,
                aclDataType selfType,
                TOther otherValue,
                aclDataType otherType,
                TAlpha alphaValue,
                aclDataType alphaType,
                aclDataType outType,
                aclrtStream stream,
                double atol,
                double rtol)
{
    TensorResource self;
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<TOut> outData(GetShapeSize(shape), static_cast<TOut>(0));
    std::vector<double> expected = ToDoubleVector(selfData);
    const double scaledOther = ToDoubleValue(alphaValue) * ToDoubleValue(otherValue);
    for (auto& item : expected) {
        item += scaledOther;
    }

    int ret = CreateAclTensor(selfData, shape, selfType, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(outData, shape, outType, &out);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(otherValue, otherType, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(alphaValue, alphaType, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return 1;
        }
    }

    ret = aclnnAdds(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnAdds failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = CopyDeviceToHost(out, &outData);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    return CheckNumericResult(caseName, outData, expected, atol, rtol);
}

template <typename TSelf, typename TOther, typename TAlpha>
int RunInplaceAddsCase(const std::string& caseName,
                       const std::vector<TSelf>& selfData,
                       const std::vector<int64_t>& shape,
                       aclDataType selfType,
                       TOther otherValue,
                       aclDataType otherType,
                       TAlpha alphaValue,
                       aclDataType alphaType,
                       aclrtStream stream,
                       double atol,
                       double rtol)
{
    TensorResource self;
    ScalarResource other;
    ScalarResource alpha;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto actual = selfData;
    std::vector<double> expected = ToDoubleVector(selfData);
    const double scaledOther = ToDoubleValue(alphaValue) * ToDoubleValue(otherValue);
    for (auto& item : expected) {
        item += scaledOther;
    }

    int ret = CreateAclTensor(selfData, shape, selfType, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(otherValue, otherType, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(alphaValue, alphaType, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return 1;
        }
    }

    ret = aclnnInplaceAdds(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceAdds failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = CopyDeviceToHost(self, &actual);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    return CheckNumericResult(caseName, actual, expected, atol, rtol);
}

template <typename TOther, typename TAlpha, typename TOut, typename TSelfScalar>
int RunAddV3Case(const std::string& caseName,
                 TSelfScalar selfValue,
                 aclDataType selfType,
                 const std::vector<TOther>& otherData,
                 const std::vector<int64_t>& shape,
                 aclDataType otherType,
                 TAlpha alphaValue,
                 aclDataType alphaType,
                 aclDataType outType,
                 aclrtStream stream,
                 double atol,
                 double rtol)
{
    TensorResource other;
    TensorResource out;
    ScalarResource self;
    ScalarResource alpha;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    std::vector<TOut> outData(GetShapeSize(shape), static_cast<TOut>(0));
    std::vector<double> expected = ToDoubleVector(otherData);
    const double selfDouble = ToDoubleValue(selfValue);
    const double alphaDouble = ToDoubleValue(alphaValue);
    for (auto& item : expected) {
        item = selfDouble + alphaDouble * item;
    }

    int ret = CreateAclTensor(otherData, shape, otherType, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(outData, shape, outType, &out);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(selfValue, selfType, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(alphaValue, alphaType, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return 1;
        }
    }

    ret = aclnnAddV3(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnAddV3 failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = CopyDeviceToHost(out, &outData);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    return CheckNumericResult(caseName, outData, expected, atol, rtol);
}

template <typename TOther, typename TAlpha, typename TSelfScalar>
int RunInplaceAddV3Case(const std::string& caseName,
                        TSelfScalar selfValue,
                        aclDataType selfType,
                        const std::vector<TOther>& otherData,
                        const std::vector<int64_t>& shape,
                        aclDataType otherType,
                        TAlpha alphaValue,
                        aclDataType alphaType,
                        aclrtStream stream,
                        double atol,
                        double rtol)
{
    TensorResource other;
    ScalarResource self;
    ScalarResource alpha;
    WorkspaceResource workspace;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto actual = otherData;
    std::vector<double> expected = ToDoubleVector(otherData);
    const double selfDouble = ToDoubleValue(selfValue);
    const double alphaDouble = ToDoubleValue(alphaValue);
    for (auto& item : expected) {
        item = selfDouble + alphaDouble * item;
    }

    int ret = CreateAclTensor(otherData, shape, otherType, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(selfValue, selfType, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(alphaValue, alphaType, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceAddV3GetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[FAIL] %s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
            return 1;
        }
    }

    ret = aclnnInplaceAddV3(workspace.addr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclnnInplaceAddV3 failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
        return 1;
    }
    ret = CopyDeviceToHost(other, &actual);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    return CheckNumericResult(caseName, actual, expected, atol, rtol);
}

int RunAddNullptrCase(const std::string& caseName)
{
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    std::vector<float> tensorData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {2, 2};

    int ret = CreateAclTensor(tensorData, shape, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(tensorData, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int RunAddInvalidShapeCase(const std::string& caseName)
{
    TensorResource self;
    TensorResource other;
    TensorResource out;
    ScalarResource alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    int ret = CreateAclTensor(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(std::vector<float>{7, 8, 9}, {3}, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamInvalid) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamInvalid, ret);
    return 1;
}

int RunAddsNullptrCase(const std::string& caseName)
{
    TensorResource out;
    ScalarResource other;
    ScalarResource alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    int ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(2.0f, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnAddsGetWorkspaceSize(nullptr, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int RunInplaceAddInvalidShapeCase(const std::string& caseName)
{
    TensorResource self;
    TensorResource other;
    ScalarResource alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    int ret = CreateAclTensor(std::vector<float>{1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(std::vector<float>{5, 6, 7, 8, 9, 10}, {2, 3}, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret == kAclnnErrParamInvalid) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamInvalid, ret);
    return 1;
}

int RunInplaceAddsNullptrCase(const std::string& caseName)
{
    ScalarResource other;
    ScalarResource alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    int ret = CreateScalar(3.0f, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnInplaceAddsGetWorkspaceSize(nullptr, other.scalar, alpha.scalar, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int RunAddV3InvalidShapeCase(const std::string& caseName)
{
    TensorResource other;
    TensorResource out;
    ScalarResource self;
    ScalarResource alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    int ret = CreateAclTensor(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateAclTensor(std::vector<float>(4, 0.0f), {2, 2}, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &self);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret == kAclnnErrParamInvalid) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamInvalid, ret);
    return 1;
}

int RunInplaceAddV3NullptrCase(const std::string& caseName)
{
    TensorResource other;
    ScalarResource alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    int ret = CreateAclTensor(std::vector<float>{1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &other);
    if (ret != ACL_SUCCESS) {
        return 1;
    }
    ret = CreateScalar(1.0f, ACL_FLOAT, &alpha);
    if (ret != ACL_SUCCESS) {
        return 1;
    }

    ret = aclnnInplaceAddV3GetWorkspaceSize(nullptr, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret == kAclnnErrParamNullptr) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        return 0;
    }

    LOG_PRINT("[FAIL] %s expect=%d actual=%d\n", caseName.c_str(), kAclnnErrParamNullptr, ret);
    return 1;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int failedCases = 0;

    failedCases += RunAddCase<float, float, float, float>("add_float32_alpha1", {0.0f, 1.0f, 2.0f, 3.0f}, {2, 2},
                                                          ACL_FLOAT, {4.0f, -1.0f, 0.5f, 2.0f}, {2, 2}, ACL_FLOAT, 1.0f,
                                                          ACL_FLOAT, {2, 2}, ACL_FLOAT, stream, 1e-5, 1e-5);

    failedCases += RunAddCase<float, float, float, float>("add_float32_broadcast_alpha1", {1.0f, 2.0f, 3.0f,
                                                          4.0f, -5.0f, 6.0f}, {2, 3}, ACL_FLOAT,
                                                          {0.5f, -1.0f, 2.0f}, {3}, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                          {2, 3}, ACL_FLOAT, stream, 1e-5, 1e-5);

    failedCases += RunAddCase<uint16_t, float, float, float>("add_float16_float32_mix", {FloatToFloat16(1.0f),
                                                                   FloatToFloat16(-2.0f), FloatToFloat16(4.0f),
                                                                   FloatToFloat16(8.0f)}, {2, 2}, ACL_FLOAT16,
                                                                   {0.5f, 1.5f, -1.0f, 2.0f}, {2, 2}, ACL_FLOAT,
                                                                   1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, stream, 1e-3, 1e-3);

    failedCases += RunAddV3Case<float, float, float, float>("add_v3_float32_alpha1", 1.25f, ACL_FLOAT,
                                                            {0.0f, 1.0f, -2.0f, 4.0f}, {2, 2}, ACL_FLOAT, 1.0f,
                                                            ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);

    failedCases += RunAddNullptrCase("add_nullptr_check");
    failedCases += RunAddInvalidShapeCase("add_invalid_shape_check");
    failedCases += RunAddsNullptrCase("adds_nullptr_check");
    failedCases += RunInplaceAddInvalidShapeCase("inplace_add_invalid_shape_check");
    failedCases += RunInplaceAddsNullptrCase("inplace_adds_nullptr_check");
    failedCases += RunAddV3InvalidShapeCase("add_v3_invalid_shape_check");
    failedCases += RunInplaceAddV3NullptrCase("inplace_add_v3_nullptr_check");

    LOG_PRINT("\n=== Summary: %d failed ===\n", failedCases);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failedCases == 0 ? 0 : 1;
}
