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
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

constexpr double FLOAT_ATOL = 1e-5;
constexpr double FLOAT_RTOL = 1e-5;
constexpr double FP16_ATOL = 1e-3;
constexpr double FP16_RTOL = 1e-3;
constexpr double BF16_ATOL = 1e-2;
constexpr double BF16_RTOL = 1e-2;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)              \
    do {                                     \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> CalcStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() >= 2) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }
    return strides;
}

std::vector<int64_t> LeftPadShape(const std::vector<int64_t>& shape, size_t rank)
{
    std::vector<int64_t> padded(rank, 1);
    const size_t offset = rank - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        padded[offset + i] = shape[i];
    }
    return padded;
}

std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> pa = LeftPadShape(a, rank);
    std::vector<int64_t> pb = LeftPadShape(b, rank);
    std::vector<int64_t> out(rank, 1);

    for (size_t i = 0; i < rank; ++i) {
        if (pa[i] == pb[i]) {
            out[i] = pa[i];
        } else if (pa[i] == 1) {
            out[i] = pb[i];
        } else if (pb[i] == 1) {
            out[i] = pa[i];
        } else {
            return {};
        }
    }
    return out;
}

template <typename TA, typename TB, typename TO>
std::vector<TO> CpuBroadcastMul(const std::vector<TA>& a,
                                const std::vector<int64_t>& aShape,
                                const std::vector<TB>& b,
                                const std::vector<int64_t>& bShape)
{
    std::vector<int64_t> outShape = BroadcastShape(aShape, bShape);
    if (outShape.empty()) {
        return {};
    }

    const size_t rank = outShape.size();
    std::vector<int64_t> pa = LeftPadShape(aShape, rank);
    std::vector<int64_t> pb = LeftPadShape(bShape, rank);
    std::vector<int64_t> outStrides = CalcStrides(outShape);
    std::vector<int64_t> aStrides = CalcStrides(pa);
    std::vector<int64_t> bStrides = CalcStrides(pb);

    const int64_t outNum = GetShapeSize(outShape);
    std::vector<TO> out(outNum);

    for (int64_t linear = 0; linear < outNum; ++linear) {
        int64_t remain = linear;
        int64_t aOffset = 0;
        int64_t bOffset = 0;

        for (size_t d = 0; d < rank; ++d) {
            const int64_t coord = remain / outStrides[d];
            remain %= outStrides[d];

            const int64_t aCoord = (pa[d] == 1) ? 0 : coord;
            const int64_t bCoord = (pb[d] == 1) ? 0 : coord;

            aOffset += aCoord * aStrides[d];
            bOffset += bCoord * bStrides[d];
        }
        out[linear] = static_cast<TO>(a[aOffset]) * static_cast<TO>(b[bOffset]);
    }
    return out;
}

template <typename T>
std::vector<T> RepeatPattern(const std::vector<T>& pattern, int64_t total)
{
    std::vector<T> out(total);
    for (int64_t i = 0; i < total; ++i) {
        out[i] = pattern[static_cast<size_t>(i % static_cast<int64_t>(pattern.size()))];
    }
    return out;
}

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
        auto ret = aclrtDestroyStream(stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtDestroyStream failed. ERROR: %d\n", ret);
        }
    }

    auto ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtResetDevice failed. ERROR: %d\n", ret);
    }

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclFinalize failed. ERROR: %d\n", ret);
    }
}

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
};

void ReleaseTensorResource(TensorResource* resource)
{
    if (resource == nullptr) {
        return;
    }
    if (resource->tensor != nullptr) {
        aclDestroyTensor(resource->tensor);
        resource->tensor = nullptr;
    }
    if (resource->deviceAddr != nullptr) {
        aclrtFree(resource->deviceAddr);
        resource->deviceAddr = nullptr;
    }
}

size_t GetAclDataTypeSize(aclDataType dataType)
{
    switch (dataType) {
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_FLOAT16:
            return sizeof(uint16_t);
        case ACL_BF16:
            return sizeof(uint16_t);
        default:
            return 0;
    }
}

template <typename T>
bool CopyDeviceToHost(const TensorResource& res, std::vector<T>* out)
{
    if (out == nullptr || res.deviceAddr == nullptr) {
        return false;
    }
    const size_t bytes = out->size() * sizeof(T);
    auto ret = aclrtMemcpy(out->data(), bytes, res.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return false;
    }
    return true;
}

bool CheckFloatNear(const std::vector<float>& actual,
                    const std::vector<float>& expect,
                    double atol,
                    double rtol)
{
    if (actual.size() != expect.size()) {
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expect[i]));
        const double tol = atol + rtol * std::fabs(static_cast<double>(expect[i]));
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool CheckEqual(const std::vector<T>& out, const std::vector<T>& expect)
{
    if (out.size() != expect.size()) {
        return false;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != expect[i]) {
            return false;
        }
    }
    return true;
}

template <>
bool CheckEqual<float>(const std::vector<float>& out, const std::vector<float>& expect)
{
    return CheckFloatNear(out, expect, FLOAT_ATOL, FLOAT_RTOL);
}

template <typename T>
void PrintVector(const std::vector<T>& data, const std::string& name)
{
    std::cout << name << " = [";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i + 1 != data.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& v)
{
    std::vector<uint8_t> bytes(v.size() * sizeof(T));
    if (!v.empty()) {
        std::memcpy(bytes.data(), v.data(), bytes.size());
    }
    return bytes;
}

static uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16U) & 1U;
    const uint32_t roundingBias = 0x7FFFU + lsb;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16U);
}

static float Bf16BitsToFloat(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16U;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t FloatToHalfBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16U) & 0x8000U;
    uint32_t mantissa = bits & 0x007FFFFFU;
    int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000U) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mantissa + 0x00001000U) >> 13U));
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | ((mantissa + 0x00001000U) >> 13U));
}

static float HalfBitsToFloat(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000U)) << 16U;
    uint32_t exp = (h >> 10U) & 0x1FU;
    uint32_t mantissa = h & 0x03FFU;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --exp;
            }
            mantissa &= 0x03FFU;
            bits = sign | ((exp + 127 - 15) << 23U) | (mantissa << 13U);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exp + 127 - 15) << 23U) | (mantissa << 13U);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static std::vector<uint8_t> FloatVecToFp16Bytes(const std::vector<float>& input, std::vector<float>* quantized)
{
    std::vector<uint16_t> tmp(input.size());
    if (quantized != nullptr) {
        quantized->resize(input.size());
    }

    for (size_t i = 0; i < input.size(); ++i) {
        tmp[i] = FloatToHalfBits(input[i]);
        if (quantized != nullptr) {
            (*quantized)[i] = HalfBitsToFloat(tmp[i]);
        }
    }

    std::vector<uint8_t> bytes(tmp.size() * sizeof(uint16_t));
    if (!tmp.empty()) {
        std::memcpy(bytes.data(), tmp.data(), bytes.size());
    }
    return bytes;
}

static std::vector<uint8_t> FloatVecToBf16Bytes(const std::vector<float>& input, std::vector<float>* quantized)
{
    std::vector<uint16_t> tmp(input.size());
    if (quantized != nullptr) {
        quantized->resize(input.size());
    }

    for (size_t i = 0; i < input.size(); ++i) {
        tmp[i] = FloatToBf16Bits(input[i]);
        if (quantized != nullptr) {
            (*quantized)[i] = Bf16BitsToFloat(tmp[i]);
        }
    }

    std::vector<uint8_t> bytes(tmp.size() * sizeof(uint16_t));
    if (!tmp.empty()) {
        std::memcpy(bytes.data(), tmp.data(), bytes.size());
    }
    return bytes;
}

template <typename T>
int CreateAclTensorEx(const std::vector<T>& storageData,
                      const std::vector<int64_t>& logicalShape,
                      const std::vector<int64_t>& storageShape,
                      const std::vector<int64_t>& strides,
                      aclDataType dataType,
                      TensorResource* resource)
{
    CHECK_RET(resource != nullptr, LOG_PRINT("resource is nullptr\n"); return -1);

    const int64_t storageNumel = GetShapeSize(storageShape);
    CHECK_RET(storageNumel > 0, LOG_PRINT("invalid storage shape\n"); return -1);
    CHECK_RET(static_cast<int64_t>(storageData.size()) == storageNumel,
              LOG_PRINT("storageData size mismatch\n"); return -1);
    CHECK_RET(logicalShape.size() == strides.size() && logicalShape.size() == storageShape.size(),
              LOG_PRINT("shape/stride rank mismatch\n"); return -1);

    const size_t bytes = static_cast<size_t>(storageNumel) * sizeof(T);

    auto ret = aclrtMalloc(&resource->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(resource->deviceAddr, bytes, storageData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy(H2D) failed. ERROR: %d\n", ret); return ret);

    resource->tensor = aclCreateTensor(logicalShape.data(),
                                       logicalShape.size(),
                                       dataType,
                                       strides.data(),
                                       0,
                                       aclFormat::ACL_FORMAT_ND,
                                       storageShape.data(),
                                       storageShape.size(),
                                       resource->deviceAddr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);

    return ACL_SUCCESS;
}

int CreateAclTensorFromBytesEx(const std::vector<uint8_t>& storageBytes,
                               const std::vector<int64_t>& logicalShape,
                               const std::vector<int64_t>& storageShape,
                               const std::vector<int64_t>& strides,
                               aclDataType dataType,
                               TensorResource* resource)
{
    CHECK_RET(resource != nullptr, LOG_PRINT("resource is nullptr\n"); return -1);

    const int64_t storageNumel = GetShapeSize(storageShape);
    const size_t typeSize = GetAclDataTypeSize(dataType);
    CHECK_RET(storageNumel > 0 && typeSize > 0, LOG_PRINT("invalid storage shape or dtype\n"); return -1);
    CHECK_RET(storageBytes.size() == static_cast<size_t>(storageNumel) * typeSize,
              LOG_PRINT("raw bytes size mismatch\n"); return -1);
    CHECK_RET(logicalShape.size() == strides.size() && logicalShape.size() == storageShape.size(),
              LOG_PRINT("shape/stride rank mismatch\n"); return -1);

    auto ret = aclrtMalloc(&resource->deviceAddr, storageBytes.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(resource->deviceAddr,
                      storageBytes.size(),
                      storageBytes.data(),
                      storageBytes.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy(H2D) failed. ERROR: %d\n", ret); return ret);

    resource->tensor = aclCreateTensor(logicalShape.data(),
                                       logicalShape.size(),
                                       dataType,
                                       strides.data(),
                                       0,
                                       aclFormat::ACL_FORMAT_ND,
                                       storageShape.data(),
                                       storageShape.size(),
                                       resource->deviceAddr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);

    return ACL_SUCCESS;
}

template <typename T>
std::vector<T> PackLogicalToStridedStorage(const std::vector<T>& logicalData,
                                           const std::vector<int64_t>& logicalShape,
                                           const std::vector<int64_t>& storageShape,
                                           const std::vector<int64_t>& strides)
{
    const int64_t logicalNumel = GetShapeSize(logicalShape);
    const int64_t storageNumel = GetShapeSize(storageShape);
    std::vector<T> storage(static_cast<size_t>(storageNumel), static_cast<T>(0));

    std::vector<int64_t> logicalStrides = CalcStrides(logicalShape);
    const size_t rank = logicalShape.size();

    for (int64_t linear = 0; linear < logicalNumel; ++linear) {
        int64_t remain = linear;
        int64_t storageOffset = 0;
        for (size_t d = 0; d < rank; ++d) {
            const int64_t coord = remain / logicalStrides[d];
            remain %= logicalStrides[d];
            storageOffset += coord * strides[d];
        }
        storage[static_cast<size_t>(storageOffset)] = logicalData[static_cast<size_t>(linear)];
    }
    return storage;
}

std::vector<uint8_t> PackBytesToStridedStorage(const std::vector<uint8_t>& logicalBytes,
                                               size_t typeSize,
                                               const std::vector<int64_t>& logicalShape,
                                               const std::vector<int64_t>& storageShape,
                                               const std::vector<int64_t>& strides)
{
    const int64_t logicalNumel = GetShapeSize(logicalShape);
    const int64_t storageNumel = GetShapeSize(storageShape);
    std::vector<uint8_t> storage(static_cast<size_t>(storageNumel) * typeSize, 0);

    std::vector<int64_t> logicalStrides = CalcStrides(logicalShape);
    const size_t rank = logicalShape.size();

    for (int64_t linear = 0; linear < logicalNumel; ++linear) {
        int64_t remain = linear;
        int64_t storageOffset = 0;
        for (size_t d = 0; d < rank; ++d) {
            const int64_t coord = remain / logicalStrides[d];
            remain %= logicalStrides[d];
            storageOffset += coord * strides[d];
        }
        std::memcpy(storage.data() + static_cast<size_t>(storageOffset) * typeSize,
                    logicalBytes.data() + static_cast<size_t>(linear) * typeSize,
                    typeSize);
    }
    return storage;
}

template <typename T>
struct MulExecCase {
    std::string name;
    std::vector<int64_t> selfShape;
    std::vector<int64_t> otherShape;
    std::vector<int64_t> outShape;
    std::vector<T> selfData;
    std::vector<T> otherData;
    std::vector<T> expectData;
};

template <typename T>
struct MulsExecCase {
    std::string name;
    std::vector<int64_t> selfShape;
    std::vector<int64_t> outShape;
    std::vector<T> selfData;
    T scalarValue;
    std::vector<T> expectData;
};

template <typename T>
struct InplaceMulExecCase {
    std::string name;
    std::vector<int64_t> selfShape;
    std::vector<int64_t> otherShape;
    std::vector<T> selfInitData;
    std::vector<T> otherData;
    std::vector<T> expectData;
};

template <typename T>
struct InplaceMulsExecCase {
    std::string name;
    std::vector<int64_t> selfShape;
    std::vector<T> selfInitData;
    T scalarValue;
    std::vector<T> expectData;
};

struct MixedMulExecCase {
    std::string name;
    std::vector<int64_t> selfShape;
    std::vector<int64_t> otherShape;
    std::vector<int64_t> outShape;
    aclDataType selfType;
    aclDataType otherType;
    aclDataType outType;
    std::vector<uint8_t> selfBytes;
    std::vector<uint8_t> otherBytes;
    std::vector<float> expectData;
    double atol = FLOAT_ATOL;
    double rtol = FLOAT_RTOL;
};

struct NonContigFloatMulCase {
    std::string name;
    std::vector<int64_t> selfShape;
    std::vector<int64_t> selfStorageShape;
    std::vector<int64_t> selfStrides;
    std::vector<int64_t> otherShape;
    std::vector<int64_t> otherStorageShape;
    std::vector<int64_t> otherStrides;
    std::vector<int64_t> outShape;
    std::vector<float> selfLogicalData;
    std::vector<float> otherLogicalData;
    std::vector<float> expectData;
};

struct NonContigMixedMulCase {
    std::string name;
    std::vector<int64_t> selfShape;
    std::vector<int64_t> selfStorageShape;
    std::vector<int64_t> selfStrides;
    std::vector<int64_t> otherShape;
    std::vector<int64_t> otherStorageShape;
    std::vector<int64_t> otherStrides;
    std::vector<int64_t> outShape;
    aclDataType selfType;
    aclDataType otherType;
    aclDataType outType;
    std::vector<uint8_t> selfLogicalBytes;
    std::vector<uint8_t> otherLogicalBytes;
    std::vector<float> expectData;
    double atol = FLOAT_ATOL;
    double rtol = FLOAT_RTOL;
};

template <typename T>
MulExecCase<T> MakeMulCase(const std::string& name,
                           const std::vector<int64_t>& selfShape,
                           const std::vector<int64_t>& otherShape,
                           const std::vector<T>& selfData,
                           const std::vector<T>& otherData)
{
    MulExecCase<T> tc;
    tc.name = name;
    tc.selfShape = selfShape;
    tc.otherShape = otherShape;
    tc.outShape = BroadcastShape(selfShape, otherShape);
    tc.selfData = selfData;
    tc.otherData = otherData;
    tc.expectData = CpuBroadcastMul<T, T, T>(selfData, selfShape, otherData, otherShape);
    return tc;
}

template <typename T>
MulsExecCase<T> MakeMulsCase(const std::string& name,
                             const std::vector<int64_t>& selfShape,
                             const std::vector<T>& selfData,
                             T scalarValue)
{
    MulsExecCase<T> tc;
    tc.name = name;
    tc.selfShape = selfShape;
    tc.outShape = selfShape;
    tc.selfData = selfData;
    tc.scalarValue = scalarValue;
    tc.expectData.resize(selfData.size());
    for (size_t i = 0; i < selfData.size(); ++i) {
        tc.expectData[i] = selfData[i] * scalarValue;
    }
    return tc;
}

template <typename T>
InplaceMulExecCase<T> MakeInplaceMulCase(const std::string& name,
                                         const std::vector<int64_t>& selfShape,
                                         const std::vector<int64_t>& otherShape,
                                         const std::vector<T>& selfData,
                                         const std::vector<T>& otherData)
{
    InplaceMulExecCase<T> tc;
    tc.name = name;
    tc.selfShape = selfShape;
    tc.otherShape = otherShape;
    tc.selfInitData = selfData;
    tc.otherData = otherData;
    tc.expectData = CpuBroadcastMul<T, T, T>(selfData, selfShape, otherData, otherShape);
    return tc;
}

template <typename T>
InplaceMulsExecCase<T> MakeInplaceMulsCase(const std::string& name,
                                           const std::vector<int64_t>& selfShape,
                                           const std::vector<T>& selfData,
                                           T scalarValue)
{
    InplaceMulsExecCase<T> tc;
    tc.name = name;
    tc.selfShape = selfShape;
    tc.selfInitData = selfData;
    tc.scalarValue = scalarValue;
    tc.expectData.resize(selfData.size());
    for (size_t i = 0; i < selfData.size(); ++i) {
        tc.expectData[i] = selfData[i] * scalarValue;
    }
    return tc;
}

void EncodeFloatValuesForType(const std::vector<float>& input,
                              aclDataType dtype,
                              std::vector<uint8_t>* bytes,
                              std::vector<float>* quantized)
{
    if (dtype == ACL_FLOAT) {
        *bytes = ToBytes(input);
        *quantized = input;
        return;
    }
    if (dtype == ACL_FLOAT16) {
        *bytes = FloatVecToFp16Bytes(input, quantized);
        return;
    }
    if (dtype == ACL_BF16) {
        *bytes = FloatVecToBf16Bytes(input, quantized);
        return;
    }
    bytes->clear();
    quantized->clear();
}

MixedMulExecCase MakeMixedMulCase(const std::string& name,
                                  const std::vector<int64_t>& selfShape,
                                  const std::vector<int64_t>& otherShape,
                                  aclDataType selfType,
                                  aclDataType otherType,
                                  aclDataType outType,
                                  const std::vector<float>& selfVals,
                                  const std::vector<float>& otherVals,
                                  double atol,
                                  double rtol)
{
    MixedMulExecCase tc;
    tc.name = name;
    tc.selfShape = selfShape;
    tc.otherShape = otherShape;
    tc.outShape = BroadcastShape(selfShape, otherShape);
    tc.selfType = selfType;
    tc.otherType = otherType;
    tc.outType = outType;
    tc.atol = atol;
    tc.rtol = rtol;

    std::vector<float> qSelf;
    std::vector<float> qOther;
    EncodeFloatValuesForType(selfVals, selfType, &tc.selfBytes, &qSelf);
    EncodeFloatValuesForType(otherVals, otherType, &tc.otherBytes, &qOther);
    tc.expectData = CpuBroadcastMul<float, float, float>(qSelf, selfShape, qOther, otherShape);
    return tc;
}

NonContigFloatMulCase MakeNonContigFloatCase(const std::string& name,
                                             const std::vector<int64_t>& selfShape,
                                             const std::vector<int64_t>& selfStorageShape,
                                             const std::vector<int64_t>& selfStrides,
                                             const std::vector<int64_t>& otherShape,
                                             const std::vector<int64_t>& otherStorageShape,
                                             const std::vector<int64_t>& otherStrides,
                                             const std::vector<float>& selfLogicalData,
                                             const std::vector<float>& otherLogicalData)
{
    NonContigFloatMulCase tc;
    tc.name = name;
    tc.selfShape = selfShape;
    tc.selfStorageShape = selfStorageShape;
    tc.selfStrides = selfStrides;
    tc.otherShape = otherShape;
    tc.otherStorageShape = otherStorageShape;
    tc.otherStrides = otherStrides;
    tc.outShape = BroadcastShape(selfShape, otherShape);
    tc.selfLogicalData = selfLogicalData;
    tc.otherLogicalData = otherLogicalData;
    tc.expectData = CpuBroadcastMul<float, float, float>(selfLogicalData, selfShape, otherLogicalData, otherShape);
    return tc;
}

NonContigMixedMulCase MakeNonContigMixedCase(const std::string& name,
                                             const std::vector<int64_t>& selfShape,
                                             const std::vector<int64_t>& selfStorageShape,
                                             const std::vector<int64_t>& selfStrides,
                                             const std::vector<int64_t>& otherShape,
                                             const std::vector<int64_t>& otherStorageShape,
                                             const std::vector<int64_t>& otherStrides,
                                             aclDataType selfType,
                                             aclDataType otherType,
                                             aclDataType outType,
                                             const std::vector<float>& selfVals,
                                             const std::vector<float>& otherVals,
                                             double atol,
                                             double rtol)
{
    NonContigMixedMulCase tc;
    tc.name = name;
    tc.selfShape = selfShape;
    tc.selfStorageShape = selfStorageShape;
    tc.selfStrides = selfStrides;
    tc.otherShape = otherShape;
    tc.otherStorageShape = otherStorageShape;
    tc.otherStrides = otherStrides;
    tc.outShape = BroadcastShape(selfShape, otherShape);
    tc.selfType = selfType;
    tc.otherType = otherType;
    tc.outType = outType;
    tc.atol = atol;
    tc.rtol = rtol;

    std::vector<float> qSelf;
    std::vector<float> qOther;
    EncodeFloatValuesForType(selfVals, selfType, &tc.selfLogicalBytes, &qSelf);
    EncodeFloatValuesForType(otherVals, otherType, &tc.otherLogicalBytes, &qOther);
    tc.expectData = CpuBroadcastMul<float, float, float>(qSelf, selfShape, qOther, otherShape);
    return tc;
}

template <typename T>
bool RunMulExecuteCase(const MulExecCase<T>& tc, aclDataType dataType, aclrtStream stream)
{
    LOG_PRINT("\n[ RUN      ] %s\n", tc.name.c_str());

    TensorResource selfRes;
    TensorResource otherRes;
    TensorResource outRes;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
        ReleaseTensorResource(&outRes);
    };

    const int64_t selfNum = GetShapeSize(tc.selfShape);
    const int64_t otherNum = GetShapeSize(tc.otherShape);
    const int64_t outNum = GetShapeSize(tc.outShape);

    if (selfNum <= 0 || otherNum <= 0 || outNum <= 0 ||
        static_cast<int64_t>(tc.selfData.size()) != selfNum ||
        static_cast<int64_t>(tc.otherData.size()) != otherNum ||
        static_cast<int64_t>(tc.expectData.size()) != outNum) {
        LOG_PRINT("[  FAILED  ] %s: size mismatch\n", tc.name.c_str());
        cleanup();
        return false;
    }

    std::vector<T> outHostData(outNum, static_cast<T>(0));

    auto ret = CreateAclTensorEx(tc.selfData, tc.selfShape, tc.selfShape, CalcStrides(tc.selfShape), dataType, &selfRes);
    if (ret != ACL_SUCCESS ||
        CreateAclTensorEx(tc.otherData, tc.otherShape, tc.otherShape, CalcStrides(tc.otherShape), dataType, &otherRes) != ACL_SUCCESS ||
        CreateAclTensorEx(outHostData, tc.outShape, tc.outShape, CalcStrides(tc.outShape), dataType, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: CreateAclTensor failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    ret = aclnnMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, outRes.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s: allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
            cleanup();
            return false;
        }
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMul failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    std::vector<T> resultData(outNum, static_cast<T>(0));
    if (!CopyDeviceToHost(outRes, &resultData)) {
        LOG_PRINT("[  FAILED  ] %s: copy result failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    const bool ok = CheckEqual(resultData, tc.expectData);
    if (!ok) {
        LOG_PRINT("[  FAILED  ] %s: result mismatch\n", tc.name.c_str());
        PrintVector(resultData, "result");
        PrintVector(tc.expectData, "expect");
        cleanup();
        return false;
    }

    LOG_PRINT("[       OK ] %s\n", tc.name.c_str());
    cleanup();
    return true;
}

template <typename T>
bool RunMulsExecuteCase(const MulsExecCase<T>& tc, aclDataType dataType, aclrtStream stream)
{
    LOG_PRINT("\n[ RUN      ] %s\n", tc.name.c_str());

    TensorResource selfRes;
    TensorResource outRes;
    aclScalar* scalar = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
            scalar = nullptr;
        }
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&outRes);
    };

    std::vector<T> outInit(tc.expectData.size(), static_cast<T>(0));

    auto ret = CreateAclTensorEx(tc.selfData, tc.selfShape, tc.selfShape, CalcStrides(tc.selfShape), dataType, &selfRes);
    if (ret != ACL_SUCCESS ||
        CreateAclTensorEx(outInit, tc.outShape, tc.outShape, CalcStrides(tc.outShape), dataType, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: CreateAclTensor failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    T scalarValue = tc.scalarValue;
    scalar = aclCreateScalar(&scalarValue, dataType);
    if (scalar == nullptr) {
        LOG_PRINT("[  FAILED  ] %s: aclCreateScalar failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    ret = aclnnMulsGetWorkspaceSize(selfRes.tensor, scalar, outRes.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s: allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
            cleanup();
            return false;
        }
    }

    ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMuls failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    std::vector<T> resultData(tc.expectData.size(), static_cast<T>(0));
    if (!CopyDeviceToHost(outRes, &resultData)) {
        LOG_PRINT("[  FAILED  ] %s: copy result failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    const bool ok = CheckEqual(resultData, tc.expectData);
    if (!ok) {
        LOG_PRINT("[  FAILED  ] %s: result mismatch\n", tc.name.c_str());
        PrintVector(resultData, "result");
        PrintVector(tc.expectData, "expect");
        cleanup();
        return false;
    }

    LOG_PRINT("[       OK ] %s\n", tc.name.c_str());
    cleanup();
    return true;
}

template <typename T>
bool RunInplaceMulExecuteCase(const InplaceMulExecCase<T>& tc, aclDataType dataType, aclrtStream stream)
{
    LOG_PRINT("\n[ RUN      ] %s\n", tc.name.c_str());

    TensorResource selfRes;
    TensorResource otherRes;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
    };

    auto ret = CreateAclTensorEx(tc.selfInitData, tc.selfShape, tc.selfShape, CalcStrides(tc.selfShape), dataType, &selfRes);
    if (ret != ACL_SUCCESS ||
        CreateAclTensorEx(tc.otherData, tc.otherShape, tc.otherShape, CalcStrides(tc.otherShape), dataType, &otherRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: CreateAclTensor failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    ret = aclnnInplaceMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s: allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
            cleanup();
            return false;
        }
    }

    ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnInplaceMul failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    std::vector<T> resultData(tc.expectData.size(), static_cast<T>(0));
    if (!CopyDeviceToHost(selfRes, &resultData)) {
        LOG_PRINT("[  FAILED  ] %s: copy result failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    const bool ok = CheckEqual(resultData, tc.expectData);
    if (!ok) {
        LOG_PRINT("[  FAILED  ] %s: result mismatch\n", tc.name.c_str());
        PrintVector(resultData, "result");
        PrintVector(tc.expectData, "expect");
        cleanup();
        return false;
    }

    LOG_PRINT("[       OK ] %s\n", tc.name.c_str());
    cleanup();
    return true;
}

template <typename T>
bool RunInplaceMulsExecuteCase(const InplaceMulsExecCase<T>& tc, aclDataType dataType, aclrtStream stream)
{
    LOG_PRINT("\n[ RUN      ] %s\n", tc.name.c_str());

    TensorResource selfRes;
    aclScalar* scalar = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
            scalar = nullptr;
        }
        ReleaseTensorResource(&selfRes);
    };

    auto ret = CreateAclTensorEx(tc.selfInitData, tc.selfShape, tc.selfShape, CalcStrides(tc.selfShape), dataType, &selfRes);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: CreateAclTensor failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    T scalarValue = tc.scalarValue;
    scalar = aclCreateScalar(&scalarValue, dataType);
    if (scalar == nullptr) {
        LOG_PRINT("[  FAILED  ] %s: aclCreateScalar failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    ret = aclnnInplaceMulsGetWorkspaceSize(selfRes.tensor, scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s: allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
            cleanup();
            return false;
        }
    }

    ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnInplaceMuls failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    std::vector<T> resultData(tc.expectData.size(), static_cast<T>(0));
    if (!CopyDeviceToHost(selfRes, &resultData)) {
        LOG_PRINT("[  FAILED  ] %s: copy result failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    const bool ok = CheckEqual(resultData, tc.expectData);
    if (!ok) {
        LOG_PRINT("[  FAILED  ] %s: result mismatch\n", tc.name.c_str());
        PrintVector(resultData, "result");
        PrintVector(tc.expectData, "expect");
        cleanup();
        return false;
    }

    LOG_PRINT("[       OK ] %s\n", tc.name.c_str());
    cleanup();
    return true;
}

bool RunMixedMulExecuteCase(const MixedMulExecCase& tc, aclrtStream stream)
{
    LOG_PRINT("\n[ RUN      ] %s\n", tc.name.c_str());

    TensorResource selfRes;
    TensorResource otherRes;
    TensorResource outRes;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
        ReleaseTensorResource(&outRes);
    };

    std::vector<float> outInit(tc.expectData.size(), 0.0f);

    auto ret = CreateAclTensorFromBytesEx(tc.selfBytes, tc.selfShape, tc.selfShape, CalcStrides(tc.selfShape), tc.selfType, &selfRes);
    if (ret != ACL_SUCCESS ||
        CreateAclTensorFromBytesEx(tc.otherBytes, tc.otherShape, tc.otherShape, CalcStrides(tc.otherShape), tc.otherType, &otherRes) != ACL_SUCCESS ||
        CreateAclTensorEx(outInit, tc.outShape, tc.outShape, CalcStrides(tc.outShape), tc.outType, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: CreateAclTensor failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    ret = aclnnMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, outRes.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s: allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
            cleanup();
            return false;
        }
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMul failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    std::vector<float> resultData(tc.expectData.size(), 0.0f);
    if (!CopyDeviceToHost(outRes, &resultData)) {
        LOG_PRINT("[  FAILED  ] %s: copy result failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    const bool ok = CheckFloatNear(resultData, tc.expectData, tc.atol, tc.rtol);
    if (!ok) {
        LOG_PRINT("[  FAILED  ] %s: result mismatch\n", tc.name.c_str());
        PrintVector(resultData, "result");
        PrintVector(tc.expectData, "expect");
        cleanup();
        return false;
    }

    LOG_PRINT("[       OK ] %s\n", tc.name.c_str());
    cleanup();
    return true;
}

bool RunNonContigFloatMulCase(const NonContigFloatMulCase& tc, aclrtStream stream)
{
    LOG_PRINT("\n[ RUN      ] %s\n", tc.name.c_str());

    TensorResource selfRes;
    TensorResource otherRes;
    TensorResource outRes;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
        ReleaseTensorResource(&outRes);
    };

    std::vector<float> selfStorage = PackLogicalToStridedStorage(tc.selfLogicalData, tc.selfShape, tc.selfStorageShape, tc.selfStrides);
    std::vector<float> otherStorage = PackLogicalToStridedStorage(tc.otherLogicalData, tc.otherShape, tc.otherStorageShape, tc.otherStrides);
    std::vector<float> outInit(tc.expectData.size(), 0.0f);

    auto ret = CreateAclTensorEx(selfStorage, tc.selfShape, tc.selfStorageShape, tc.selfStrides, ACL_FLOAT, &selfRes);
    if (ret != ACL_SUCCESS ||
        CreateAclTensorEx(otherStorage, tc.otherShape, tc.otherStorageShape, tc.otherStrides, ACL_FLOAT, &otherRes) != ACL_SUCCESS ||
        CreateAclTensorEx(outInit, tc.outShape, tc.outShape, CalcStrides(tc.outShape), ACL_FLOAT, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: CreateAclTensor failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    ret = aclnnMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, outRes.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s: allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
            cleanup();
            return false;
        }
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMul failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    std::vector<float> resultData(tc.expectData.size(), 0.0f);
    if (!CopyDeviceToHost(outRes, &resultData)) {
        LOG_PRINT("[  FAILED  ] %s: copy result failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    const bool ok = CheckFloatNear(resultData, tc.expectData, FLOAT_ATOL, FLOAT_RTOL);
    if (!ok) {
        LOG_PRINT("[  FAILED  ] %s: result mismatch\n", tc.name.c_str());
        PrintVector(resultData, "result");
        PrintVector(tc.expectData, "expect");
        cleanup();
        return false;
    }

    LOG_PRINT("[       OK ] %s\n", tc.name.c_str());
    cleanup();
    return true;
}

bool RunNonContigMixedMulCase(const NonContigMixedMulCase& tc, aclrtStream stream)
{
    LOG_PRINT("\n[ RUN      ] %s\n", tc.name.c_str());

    TensorResource selfRes;
    TensorResource otherRes;
    TensorResource outRes;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
        ReleaseTensorResource(&outRes);
    };

    const size_t selfTypeSize = GetAclDataTypeSize(tc.selfType);
    const size_t otherTypeSize = GetAclDataTypeSize(tc.otherType);

    std::vector<uint8_t> selfStorageBytes =
        PackBytesToStridedStorage(tc.selfLogicalBytes, selfTypeSize, tc.selfShape, tc.selfStorageShape, tc.selfStrides);
    std::vector<uint8_t> otherStorageBytes =
        PackBytesToStridedStorage(tc.otherLogicalBytes, otherTypeSize, tc.otherShape, tc.otherStorageShape, tc.otherStrides);
    std::vector<float> outInit(tc.expectData.size(), 0.0f);

    auto ret = CreateAclTensorFromBytesEx(selfStorageBytes, tc.selfShape, tc.selfStorageShape, tc.selfStrides, tc.selfType, &selfRes);
    if (ret != ACL_SUCCESS ||
        CreateAclTensorFromBytesEx(otherStorageBytes, tc.otherShape, tc.otherStorageShape, tc.otherStrides, tc.otherType, &otherRes) != ACL_SUCCESS ||
        CreateAclTensorEx(outInit, tc.outShape, tc.outShape, CalcStrides(tc.outShape), tc.outType, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: CreateAclTensor failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    ret = aclnnMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, outRes.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s: allocate workspace failed. ERROR: %d\n", tc.name.c_str(), ret);
            cleanup();
            return false;
        }
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclnnMul failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s: aclrtSynchronizeStream failed. ERROR: %d\n", tc.name.c_str(), ret);
        cleanup();
        return false;
    }

    std::vector<float> resultData(tc.expectData.size(), 0.0f);
    if (!CopyDeviceToHost(outRes, &resultData)) {
        LOG_PRINT("[  FAILED  ] %s: copy result failed\n", tc.name.c_str());
        cleanup();
        return false;
    }

    const bool ok = CheckFloatNear(resultData, tc.expectData, tc.atol, tc.rtol);
    if (!ok) {
        LOG_PRINT("[  FAILED  ] %s: result mismatch\n", tc.name.c_str());
        PrintVector(resultData, "result");
        PrintVector(tc.expectData, "expect");
        cleanup();
        return false;
    }

    LOG_PRINT("[       OK ] %s\n", tc.name.c_str());
    cleanup();
    return true;
}

bool RunInvalidMulOutputShapeCase()
{
    LOG_PRINT("\n[ RUN      ] invalid_mul_output_shape\n");

    TensorResource selfRes;
    TensorResource otherRes;
    TensorResource outRes;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
        ReleaseTensorResource(&outRes);
    };

    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {2, 3};
    std::vector<int64_t> outShape = {3, 2};
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
    std::vector<float> otherData = {1, 1, 1, 2, 2, 2};
    std::vector<float> outData(6, 0);

    if (CreateAclTensorEx(selfData, selfShape, selfShape, CalcStrides(selfShape), ACL_FLOAT, &selfRes) != ACL_SUCCESS ||
        CreateAclTensorEx(otherData, otherShape, otherShape, CalcStrides(otherShape), ACL_FLOAT, &otherRes) != ACL_SUCCESS ||
        CreateAclTensorEx(outData, outShape, outShape, CalcStrides(outShape), ACL_FLOAT, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] invalid_mul_output_shape: CreateAclTensor failed\n");
        cleanup();
        return false;
    }

    auto ret = aclnnMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, outRes.tensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    LOG_PRINT(ok ? "[       OK ] invalid_mul_output_shape\n" : "[  FAILED  ] invalid_mul_output_shape\n");
    cleanup();
    return ok;
}

bool RunInvalidMulsNullScalarCase()
{
    LOG_PRINT("\n[ RUN      ] invalid_muls_null_scalar\n");

    TensorResource selfRes;
    TensorResource outRes;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&outRes);
    };

    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> outData(4, 0);

    if (CreateAclTensorEx(selfData, shape, shape, CalcStrides(shape), ACL_FLOAT, &selfRes) != ACL_SUCCESS ||
        CreateAclTensorEx(outData, shape, shape, CalcStrides(shape), ACL_FLOAT, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] invalid_muls_null_scalar: CreateAclTensor failed\n");
        cleanup();
        return false;
    }

    auto ret = aclnnMulsGetWorkspaceSize(selfRes.tensor, nullptr, outRes.tensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    LOG_PRINT(ok ? "[       OK ] invalid_muls_null_scalar\n" : "[  FAILED  ] invalid_muls_null_scalar\n");
    cleanup();
    return ok;
}

bool RunInvalidMixedMulCase(const std::string& name,
                            const std::vector<int64_t>& selfShape,
                            const std::vector<int64_t>& otherShape,
                            aclDataType selfType,
                            aclDataType otherType,
                            aclDataType outType,
                            const std::vector<float>& selfVals,
                            const std::vector<float>& otherVals)
{
    LOG_PRINT("\n[ RUN      ] %s\n", name.c_str());

    TensorResource selfRes;
    TensorResource otherRes;
    TensorResource outRes;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
        ReleaseTensorResource(&outRes);
    };

    std::vector<uint8_t> selfBytes;
    std::vector<uint8_t> otherBytes;
    std::vector<float> qSelf;
    std::vector<float> qOther;
    EncodeFloatValuesForType(selfVals, selfType, &selfBytes, &qSelf);
    EncodeFloatValuesForType(otherVals, otherType, &otherBytes, &qOther);

    const std::vector<int64_t> outShape = BroadcastShape(selfShape, otherShape);
    if (outShape.empty()) {
        LOG_PRINT("[  FAILED  ] %s : invalid broadcast shape\n", name.c_str());
        cleanup();
        return false;
    }

    const size_t outBytesSize = static_cast<size_t>(GetShapeSize(outShape)) * GetAclDataTypeSize(outType);
    std::vector<uint8_t> outBytes(outBytesSize, 0);

    if (CreateAclTensorFromBytesEx(selfBytes, selfShape, selfShape, CalcStrides(selfShape), selfType, &selfRes) != ACL_SUCCESS ||
        CreateAclTensorFromBytesEx(otherBytes, otherShape, otherShape, CalcStrides(otherShape), otherType, &otherRes) != ACL_SUCCESS ||
        CreateAclTensorFromBytesEx(outBytes, outShape, outShape, CalcStrides(outShape), outType, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s : CreateAclTensor failed\n", name.c_str());
        cleanup();
        return false;
    }

    auto ret = aclnnMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, outRes.tensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    LOG_PRINT(ok ? "[       OK ] %s\n" : "[  FAILED  ] %s\n", name.c_str());
    cleanup();
    return ok;
}

bool RunInvalidNonContigCase()
{
    LOG_PRINT("\n[ RUN      ] invalid_noncontig_bad_stride\n");

    TensorResource selfRes;
    TensorResource otherRes;
    TensorResource outRes;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto cleanup = [&]() {
        ReleaseTensorResource(&selfRes);
        ReleaseTensorResource(&otherRes);
        ReleaseTensorResource(&outRes);
    };

    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> selfStorageShape = {2, 3};
    std::vector<int64_t> selfStrides = {2, 1}; // 故意让第二行越界/重叠
    std::vector<int64_t> otherShape = {2, 3};
    std::vector<int64_t> outShape = {2, 3};

    std::vector<float> selfLogical = {1, 2, 3, 4, 5, 6};
    std::vector<float> otherLogical = {1, 1, 1, 2, 2, 2};
    std::vector<float> outInit(6, 0);

    std::vector<float> selfStorage = PackLogicalToStridedStorage(selfLogical, selfShape, selfStorageShape, selfStrides);
    std::vector<float> otherStorage = otherLogical;

    if (CreateAclTensorEx(selfStorage, selfShape, selfStorageShape, selfStrides, ACL_FLOAT, &selfRes) != ACL_SUCCESS ||
        CreateAclTensorEx(otherStorage, otherShape, otherShape, CalcStrides(otherShape), ACL_FLOAT, &otherRes) != ACL_SUCCESS ||
        CreateAclTensorEx(outInit, outShape, outShape, CalcStrides(outShape), ACL_FLOAT, &outRes) != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] invalid_noncontig_bad_stride: CreateAclTensor failed\n");
        cleanup();
        return false;
    }

    auto ret = aclnnMulGetWorkspaceSize(selfRes.tensor, otherRes.tensor, outRes.tensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    LOG_PRINT(ok ? "[       OK ] invalid_noncontig_bad_stride\n" : "[  FAILED  ] invalid_noncontig_bad_stride\n");
    cleanup();
    return ok;
}

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const std::vector<MulExecCase<float>> mulFloatCases = {
        MakeMulCase<float>("mul_float_same_shape_basic",
                           {4, 2}, {4, 2},
                           {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
                           {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f}),
        MakeMulCase<float>("mul_float_broadcast_2x3_1x3",
                           {2, 3}, {1, 3},
                           {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                           {10.f, 20.f, 30.f}),
        MakeMulCase<float>("mul_float_negative_zero_mix",
                           {5}, {5},
                           {-3.f, -1.f, 0.f, 2.f, 5.f},
                           {2.f, -2.f, 7.f, 0.5f, -1.f}),
        MakeMulCase<float>("mul_float_large_same_shape_128x128",
                           {128, 128}, {128, 128},
                           RepeatPattern<float>({-2.f, -1.f, 0.f, 1.f, 2.f, 3.f}, 128 * 128),
                           RepeatPattern<float>({0.5f, -1.5f, 2.f, -2.f, 4.f}, 128 * 128)),
        MakeMulCase<float>("mul_float_large_same_shape_192x192",
                           {192, 192}, {192, 192},
                           RepeatPattern<float>({-3.f, -1.f, 0.f, 2.f, 5.f}, 192 * 192),
                           RepeatPattern<float>({0.25f, -2.f, 4.f, 1.5f}, 192 * 192)),
        MakeMulCase<float>("mul_float_tiling_rank3_broadcast_8x1x128_1x16x128",
                           {8, 1, 128}, {1, 16, 128},
                           RepeatPattern<float>({-3.f, -1.f, 0.f, 1.f, 2.f}, 8 * 1 * 128),
                           RepeatPattern<float>({0.5f, -2.0f, 4.0f, 1.5f}, 1 * 16 * 128)),
        MakeMulCase<float>("mul_float_tiling_rank4_broadcast_4x1x32x16_1x8x1x16",
                           {4, 1, 32, 16}, {1, 8, 1, 16},
                           RepeatPattern<float>({-1.f, 0.f, 1.f, 2.f, 3.f}, 4 * 1 * 32 * 16),
                           RepeatPattern<float>({2.f, -3.f, 0.5f, 4.f}, 1 * 8 * 1 * 16)),
        MakeMulCase<float>("mul_float_tiling_rank5_broadcast_2x1x4x1x16_1x3x1x8x16",
                           {2, 1, 4, 1, 16}, {1, 3, 1, 8, 16},
                           RepeatPattern<float>({-2.f, -1.f, 0.f, 1.f, 2.f}, 2 * 1 * 4 * 1 * 16),
                           RepeatPattern<float>({0.25f, -2.f, 4.f, 1.5f}, 1 * 3 * 1 * 8 * 16)),
        MakeMulCase<float>("mul_float_tiling_rank6_broadcast_2x1x2x1x8x8_1x3x1x4x1x8",
                           {2, 1, 2, 1, 8, 8}, {1, 3, 1, 4, 1, 8},
                           RepeatPattern<float>({-4.f, -1.f, 0.f, 2.f, 6.f}, 2 * 1 * 2 * 1 * 8 * 8),
                           RepeatPattern<float>({0.5f, -1.5f, 3.f, 5.f}, 1 * 3 * 1 * 4 * 1 * 8))
    };

    const std::vector<MulExecCase<int32_t>> mulIntCases = {
        MakeMulCase<int32_t>("mul_int32_same_shape_basic",
                             {6}, {6},
                             {-10, -1, 0, 2, 7, 9},
                             {3, -5, 4, 0, -2, 11}),
        MakeMulCase<int32_t>("mul_int32_broadcast_2x3_1x3",
                             {2, 3}, {1, 3},
                             {1, 2, 3, 4, 5, 6},
                             {10, 20, 30}),
        MakeMulCase<int32_t>("mul_int32_large_same_shape_64x64",
                             {64, 64}, {64, 64},
                             RepeatPattern<int32_t>({-5, -1, 0, 1, 2, 7}, 64 * 64),
                             RepeatPattern<int32_t>({3, -2, 4, -6}, 64 * 64))
    };

    const std::vector<MulsExecCase<float>> mulsFloatCases = {
        MakeMulsCase<float>("muls_float_basic", {4}, {1.f, -2.f, 3.5f, -4.f}, 2.f),
        MakeMulsCase<float>("muls_float_zero_scalar", {4}, {1.f, -2.f, 3.5f, -4.f}, 0.f),
        MakeMulsCase<float>("muls_float_large_4096",
                            {4096},
                            RepeatPattern<float>({-3.f, -1.f, 0.f, 2.f, 5.f}, 4096),
                            -1.5f)
    };

    const std::vector<MulsExecCase<int32_t>> mulsIntCases = {
        MakeMulsCase<int32_t>("muls_int32_negative_scalar", {5}, {1, 2, -3, 4, -5}, -3),
        MakeMulsCase<int32_t>("muls_int32_large_2048",
                              {2048},
                              RepeatPattern<int32_t>({-7, -3, 0, 2, 8}, 2048),
                              4)
    };

    const std::vector<InplaceMulExecCase<float>> inplaceMulFloatCases = {
        MakeInplaceMulCase<float>("inplace_mul_float_same_shape",
                                  {4}, {4},
                                  {1.f, 2.f, 3.f, 4.f},
                                  {2.f, 0.5f, -1.f, 3.f}),
        MakeInplaceMulCase<float>("inplace_mul_float_broadcast_2x3_1x3",
                                  {2, 3}, {1, 3},
                                  {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                  {10.f, 20.f, 30.f}),
        MakeInplaceMulCase<float>("inplace_mul_float_large_32x64",
                                  {32, 64}, {32, 64},
                                  RepeatPattern<float>({-2.f, -1.f, 0.f, 1.f, 2.f}, 32 * 64),
                                  RepeatPattern<float>({0.5f, 2.f, -3.f}, 32 * 64))
    };

    const std::vector<InplaceMulsExecCase<float>> inplaceMulsFloatCases = {
        MakeInplaceMulsCase<float>("inplace_muls_float_basic",
                                   {4},
                                   {1.f, -2.f, 3.5f, -4.f},
                                   2.f),
        MakeInplaceMulsCase<float>("inplace_muls_float_large_1024",
                                   {1024},
                                   RepeatPattern<float>({-4.f, -1.f, 0.f, 2.f, 6.f}, 1024),
                                   0.25f)
    };

    const std::vector<MixedMulExecCase> mixedCases = {
        MakeMixedMulCase("mixed_f16_f32_same_shape_to_f32",
                         {2, 4}, {2, 4},
                         ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
                         {0.5f, -1.25f, 2.0f, -3.0f, 4.5f, -5.5f, 6.0f, 7.25f},
                         {2.0f, 3.0f, -1.0f, 0.5f, -2.0f, 1.5f, 4.0f, -0.5f},
                         FP16_ATOL, FP16_RTOL),
        MakeMixedMulCase("mixed_f32_f16_broadcast_to_f32",
                         {2, 3}, {1, 3},
                         ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT,
                         {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, 6.0f},
                         {0.5f, -1.5f, 2.0f},
                         FP16_ATOL, FP16_RTOL),
        MakeMixedMulCase("mixed_bf16_f32_same_shape_to_f32",
                         {4, 2}, {4, 2},
                         ACL_BF16, ACL_FLOAT, ACL_FLOAT,
                         {1.0f, -2.0f, 3.25f, -4.5f, 5.0f, -6.75f, 7.0f, 8.5f},
                         {0.5f, 2.0f, -1.0f, 3.0f, -2.0f, 1.5f, 4.0f, -0.5f},
                         BF16_ATOL, BF16_RTOL),
        MakeMixedMulCase("mixed_f32_bf16_same_shape_to_f32",
                         {4, 2}, {4, 2},
                         ACL_FLOAT, ACL_BF16, ACL_FLOAT,
                         {1.0f, -2.0f, 3.25f, -4.5f, 5.0f, -6.75f, 7.0f, 8.5f},
                         {0.5f, 2.0f, -1.0f, 3.0f, -2.0f, 1.5f, 4.0f, -0.5f},
                         BF16_ATOL, BF16_RTOL),
        MakeMixedMulCase("mixed_f16_f32_rank5_broadcast_to_f32",
                         {2, 1, 4, 1, 16}, {1, 3, 1, 8, 16},
                         ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
                         RepeatPattern<float>({-2.f, -1.f, 0.f, 1.f, 2.f}, 2 * 1 * 4 * 1 * 16),
                         RepeatPattern<float>({0.25f, -2.f, 4.f, 1.5f}, 1 * 3 * 1 * 8 * 16),
                         FP16_ATOL, FP16_RTOL),
        MakeMixedMulCase("mixed_f32_f16_rank6_broadcast_to_f32",
                         {2, 1, 2, 1, 8, 8}, {1, 3, 1, 4, 1, 8},
                         ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT,
                         RepeatPattern<float>({-4.f, -1.f, 0.f, 2.f, 6.f}, 2 * 1 * 2 * 1 * 8 * 8),
                         RepeatPattern<float>({0.5f, -1.5f, 3.f, 5.f}, 1 * 3 * 1 * 4 * 1 * 8),
                         FP16_ATOL, FP16_RTOL),
        MakeMixedMulCase("mixed_bf16_f32_rank5_broadcast_to_f32",
                         {2, 1, 4, 1, 16}, {1, 3, 1, 8, 16},
                         ACL_BF16, ACL_FLOAT, ACL_FLOAT,
                         RepeatPattern<float>({-2.f, -1.f, 0.f, 1.f, 2.f}, 2 * 1 * 4 * 1 * 16),
                         RepeatPattern<float>({0.25f, -2.f, 4.f, 1.5f}, 1 * 3 * 1 * 8 * 16),
                         BF16_ATOL, BF16_RTOL),
        MakeMixedMulCase("mixed_f32_bf16_rank6_broadcast_to_f32",
                         {2, 1, 2, 1, 8, 8}, {1, 3, 1, 4, 1, 8},
                         ACL_FLOAT, ACL_BF16, ACL_FLOAT,
                         RepeatPattern<float>({-4.f, -1.f, 0.f, 2.f, 6.f}, 2 * 1 * 2 * 1 * 8 * 8),
                         RepeatPattern<float>({0.5f, -1.5f, 3.f, 5.f}, 1 * 3 * 1 * 4 * 1 * 8),
                         BF16_ATOL, BF16_RTOL),
        MakeMixedMulCase("mixed_f16_f32_large_96x96_to_f32",
                         {96, 96}, {96, 96},
                         ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
                         RepeatPattern<float>({-3.f, -1.f, 0.f, 2.f, 5.f}, 96 * 96),
                         RepeatPattern<float>({0.25f, -2.f, 4.f, 1.5f}, 96 * 96),
                         FP16_ATOL, FP16_RTOL),
        MakeMixedMulCase("mixed_bf16_f32_large_96x96_to_f32",
                         {96, 96}, {96, 96},
                         ACL_BF16, ACL_FLOAT, ACL_FLOAT,
                         RepeatPattern<float>({-3.f, -1.f, 0.f, 2.f, 5.f}, 96 * 96),
                         RepeatPattern<float>({0.25f, -2.f, 4.f, 1.5f}, 96 * 96),
                         BF16_ATOL, BF16_RTOL)
    };

    const std::vector<NonContigFloatMulCase> nonContigFloatCases = {
        MakeNonContigFloatCase("noncontig_float_2x3_broadcast_1x3",
                               {2, 3}, {2, 4}, {4, 1},
                               {1, 3}, {1, 5}, {5, 1},
                               {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                               {10.f, 20.f, 30.f}),
        MakeNonContigFloatCase("noncontig_float_4x4_same_shape",
                               {4, 4}, {4, 6}, {6, 1},
                               {4, 4}, {4, 5}, {5, 1},
                               RepeatPattern<float>({-2.f, -1.f, 0.f, 1.f, 2.f}, 16),
                               RepeatPattern<float>({0.5f, -1.5f, 3.f}, 16)),
        MakeNonContigFloatCase("noncontig_float_rank4_broadcast",
                               {2, 1, 4, 8}, {2, 2, 4, 8}, {64, 32, 8, 1},
                               {1, 3, 1, 8}, {1, 3, 2, 8}, {48, 16, 8, 1},
                               RepeatPattern<float>({-4.f, -1.f, 0.f, 2.f, 6.f}, 2 * 1 * 4 * 8),
                               RepeatPattern<float>({0.5f, -1.5f, 3.f, 5.f}, 1 * 3 * 1 * 8))
    };

    const std::vector<NonContigMixedMulCase> nonContigMixedCases = {
        MakeNonContigMixedCase("noncontig_mixed_f16_f32_rank4",
                               {2, 1, 4, 8}, {2, 2, 4, 8}, {64, 32, 8, 1},
                               {1, 3, 1, 8}, {1, 3, 2, 8}, {48, 16, 8, 1},
                               ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
                               RepeatPattern<float>({-4.f, -1.f, 0.f, 2.f, 6.f}, 2 * 1 * 4 * 8),
                               RepeatPattern<float>({0.5f, -1.5f, 3.f, 5.f}, 1 * 3 * 1 * 8),
                               FP16_ATOL, FP16_RTOL),
        MakeNonContigMixedCase("noncontig_mixed_bf16_f32_rank4",
                               {2, 1, 4, 8}, {2, 2, 4, 8}, {64, 32, 8, 1},
                               {1, 3, 1, 8}, {1, 3, 2, 8}, {48, 16, 8, 1},
                               ACL_BF16, ACL_FLOAT, ACL_FLOAT,
                               RepeatPattern<float>({-4.f, -1.f, 0.f, 2.f, 6.f}, 2 * 1 * 4 * 8),
                               RepeatPattern<float>({0.5f, -1.5f, 3.f, 5.f}, 1 * 3 * 1 * 8),
                               BF16_ATOL, BF16_RTOL)
    };

    int failCount = 0;

    for (const auto& tc : mulFloatCases) {
        if (!RunMulExecuteCase(tc, ACL_FLOAT, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : mulIntCases) {
        if (!RunMulExecuteCase(tc, ACL_INT32, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : mulsFloatCases) {
        if (!RunMulsExecuteCase(tc, ACL_FLOAT, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : mulsIntCases) {
        if (!RunMulsExecuteCase(tc, ACL_INT32, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : inplaceMulFloatCases) {
        if (!RunInplaceMulExecuteCase(tc, ACL_FLOAT, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : inplaceMulsFloatCases) {
        if (!RunInplaceMulsExecuteCase(tc, ACL_FLOAT, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : mixedCases) {
        if (!RunMixedMulExecuteCase(tc, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : nonContigFloatCases) {
        if (!RunNonContigFloatMulCase(tc, stream)) {
            ++failCount;
        }
    }

    for (const auto& tc : nonContigMixedCases) {
        if (!RunNonContigMixedMulCase(tc, stream)) {
            ++failCount;
        }
    }

    if (!RunInvalidMulOutputShapeCase()) {
        ++failCount;
    }

    if (!RunInvalidMulsNullScalarCase()) {
        ++failCount;
    }

    if (!RunInvalidMixedMulCase("invalid_mixed_f16_f32_to_f16_output",
                                {2, 4}, {2, 4},
                                ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16,
                                {0.5f, -1.25f, 2.0f, -3.0f, 4.5f, -5.5f, 6.0f, 7.25f},
                                {2.0f, 3.0f, -1.0f, 0.5f, -2.0f, 1.5f, 4.0f, -0.5f})) {
        ++failCount;
    }

    if (!RunInvalidMixedMulCase("invalid_mixed_bf16_f32_to_bf16_output",
                                {4, 2}, {4, 2},
                                ACL_BF16, ACL_FLOAT, ACL_BF16,
                                {1.0f, -2.0f, 3.25f, -4.5f, 5.0f, -6.75f, 7.0f, 8.5f},
                                {0.5f, 2.0f, -1.0f, 3.0f, -2.0f, 1.5f, 4.0f, -0.5f})) {
        ++failCount;
    }

    if (!RunInvalidNonContigCase()) {
        ++failCount;
    }

    if (failCount == 0) {
        LOG_PRINT("\nAll Mul execute cases passed.\n");
    } else {
        LOG_PRINT("\nMul execute cases finished with %d failure(s).\n", failCount);
    }

    Finalize(deviceId, stream);
    return failCount == 0 ? 0 : 1;
}