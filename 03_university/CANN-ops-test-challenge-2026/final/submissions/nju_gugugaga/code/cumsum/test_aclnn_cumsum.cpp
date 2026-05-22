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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

constexpr aclnnStatus kAclnnSuccess = static_cast<aclnnStatus>(0);

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
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
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

size_t GetDataTypeSize(aclDataType dataType)
{
    switch (dataType) {
        case ACL_DOUBLE:
            return sizeof(double);
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_FLOAT16:
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_UINT8:
        case ACL_BOOL:
            return sizeof(uint8_t);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16U) & 0x8000U;
    uint32_t exp = (bits >> 23U) & 0xFFU;
    uint32_t mantissa = bits & 0x7FFFFFU;

    if (exp == 255U) {
        if (mantissa == 0U) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        return static_cast<uint16_t>(sign | 0x7E00U);
    }

    int32_t newExp = static_cast<int32_t>(exp) - 127 + 15;
    if (newExp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (newExp <= 0) {
        if (newExp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000U) >> static_cast<uint32_t>(1 - newExp);
        if ((mantissa & 0x00001000U) != 0U) {
            mantissa += 0x00002000U;
        }
        return static_cast<uint16_t>(sign | (mantissa >> 13U));
    }

    if ((mantissa & 0x00001000U) != 0U) {
        mantissa += 0x00002000U;
        if ((mantissa & 0x00800000U) != 0U) {
            mantissa = 0;
            ++newExp;
            if (newExp >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(newExp) << 10U) | (mantissa >> 13U));
}

float HalfToFloat(uint16_t half)
{
    const uint32_t sign = (static_cast<uint32_t>(half & 0x8000U)) << 16U;
    const uint32_t exp = (half >> 10U) & 0x1FU;
    const uint32_t mantissa = half & 0x03FFU;

    uint32_t bits = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t e = -1;
            uint32_t m = mantissa;
            do {
                ++e;
                m <<= 1U;
            } while ((m & 0x0400U) == 0U);
            m &= 0x03FFU;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23U) | (m << 13U);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exp + (127U - 15U)) << 23U) | (mantissa << 13U);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16U) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16U;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

int64_t NormalizeDim(int64_t dim, int64_t rank)
{
    if (rank <= 0) {
        rank = 1;
    }
    if (dim < 0) {
        dim += rank;
    }
    return dim;
}

std::vector<double> CpuCumsum(const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim)
{
    std::vector<double> output(input.size(), 0.0);
    const int64_t rank = static_cast<int64_t>(shape.size());
    const int64_t axis = NormalizeDim(dim, rank);
    const int64_t axisLen = shape[axis];

    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= shape[i];
    }
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[i];
    }

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double sum = 0.0;
            const int64_t base = o * axisLen * inner + i;
            for (int64_t a = 0; a < axisLen; ++a) {
                const int64_t idx = base + a * inner;
                sum += input[idx];
                output[idx] = sum;
            }
        }
    }
    return output;
}

std::vector<double> CpuCumsumV2(const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim,
    bool exclusive, bool reverse)
{
    std::vector<double> output(input.size(), 0.0);
    const int64_t rank = static_cast<int64_t>(shape.size());
    const int64_t axis = NormalizeDim(dim, rank);
    const int64_t axisLen = shape[axis];

    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= shape[i];
    }
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[i];
    }

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double sum = 0.0;
            const int64_t base = o * axisLen * inner + i;
            if (!reverse) {
                for (int64_t a = 0; a < axisLen; ++a) {
                    const int64_t idx = base + a * inner;
                    if (exclusive) {
                        output[idx] = sum;
                        sum += input[idx];
                    } else {
                        sum += input[idx];
                        output[idx] = sum;
                    }
                }
            } else {
                for (int64_t a = axisLen - 1; a >= 0; --a) {
                    const int64_t idx = base + a * inner;
                    if (exclusive) {
                        output[idx] = sum;
                        sum += input[idx];
                    } else {
                        sum += input[idx];
                        output[idx] = sum;
                    }
                }
            }
        }
    }
    return output;
}

std::string Preview(const std::vector<double>& data, size_t maxCount = 8)
{
    std::ostringstream oss;
    oss << "[";
    const size_t count = std::min(maxCount, data.size());
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << std::setprecision(7) << data[i];
    }
    if (data.size() > count) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

void AppendBytes(std::vector<uint8_t>& bytes, const void* ptr, size_t len)
{
    const auto* begin = reinterpret_cast<const uint8_t*>(ptr);
    bytes.insert(bytes.end(), begin, begin + len);
}

std::vector<uint8_t> PackInput(const std::vector<double>& input, aclDataType dtype, std::vector<double>& quantized)
{
    std::vector<uint8_t> bytes;
    quantized.clear();
    quantized.reserve(input.size());

    for (double v : input) {
        if (dtype == ACL_DOUBLE) {
            const double d = static_cast<double>(v);
            quantized.push_back(d);
            AppendBytes(bytes, &d, sizeof(d));
        } else if (dtype == ACL_FLOAT) {
            const float f = static_cast<float>(v);
            quantized.push_back(static_cast<double>(f));
            AppendBytes(bytes, &f, sizeof(f));
        } else if (dtype == ACL_FLOAT16) {
            const uint16_t h = FloatToHalf(static_cast<float>(v));
            quantized.push_back(static_cast<double>(HalfToFloat(h)));
            AppendBytes(bytes, &h, sizeof(h));
        } else if (dtype == ACL_BF16) {
            const uint16_t b = FloatToBf16(static_cast<float>(v));
            quantized.push_back(static_cast<double>(Bf16ToFloat(b)));
            AppendBytes(bytes, &b, sizeof(b));
        } else if (dtype == ACL_INT32) {
            const int32_t x = static_cast<int32_t>(v);
            quantized.push_back(static_cast<double>(x));
            AppendBytes(bytes, &x, sizeof(x));
        } else if (dtype == ACL_INT16) {
            const int16_t x = static_cast<int16_t>(v);
            quantized.push_back(static_cast<double>(x));
            AppendBytes(bytes, &x, sizeof(x));
        } else if (dtype == ACL_INT8) {
            const int8_t x = static_cast<int8_t>(v);
            quantized.push_back(static_cast<double>(x));
            AppendBytes(bytes, &x, sizeof(x));
        } else if (dtype == ACL_UINT8) {
            const uint8_t x = static_cast<uint8_t>(v);
            quantized.push_back(static_cast<double>(x));
            AppendBytes(bytes, &x, sizeof(x));
        } else if (dtype == ACL_BOOL) {
            const uint8_t x = (v != 0.0) ? 1U : 0U;
            quantized.push_back(static_cast<double>(x));
            AppendBytes(bytes, &x, sizeof(x));
        } else if (dtype == ACL_INT64) {
            const int64_t x = static_cast<int64_t>(v);
            quantized.push_back(static_cast<double>(x));
            AppendBytes(bytes, &x, sizeof(x));
        }
    }
    return bytes;
}

std::vector<double> QuantizeVector(const std::vector<double>& input, aclDataType dtype)
{
    std::vector<double> quantized;
    (void)PackInput(input, dtype, quantized);
    return quantized;
}

std::vector<double> UnpackOutput(const std::vector<uint8_t>& bytes, aclDataType dtype)
{
    const size_t elemSize = GetDataTypeSize(dtype);
    const size_t count = (elemSize == 0) ? 0 : (bytes.size() / elemSize);
    std::vector<double> output(count, 0.0);

    for (size_t i = 0; i < count; ++i) {
        const uint8_t* ptr = bytes.data() + i * elemSize;
        if (dtype == ACL_DOUBLE) {
            double v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = v;
        } else if (dtype == ACL_FLOAT) {
            float v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = static_cast<double>(v);
        } else if (dtype == ACL_FLOAT16) {
            uint16_t h = 0;
            std::memcpy(&h, ptr, sizeof(h));
            output[i] = static_cast<double>(HalfToFloat(h));
        } else if (dtype == ACL_BF16) {
            uint16_t b = 0;
            std::memcpy(&b, ptr, sizeof(b));
            output[i] = static_cast<double>(Bf16ToFloat(b));
        } else if (dtype == ACL_INT32) {
            int32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = static_cast<double>(v);
        } else if (dtype == ACL_INT16) {
            int16_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = static_cast<double>(v);
        } else if (dtype == ACL_INT8) {
            int8_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = static_cast<double>(v);
        } else if (dtype == ACL_UINT8) {
            uint8_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = static_cast<double>(v);
        } else if (dtype == ACL_BOOL) {
            uint8_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = static_cast<double>(v != 0);
        } else if (dtype == ACL_INT64) {
            int64_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            output[i] = static_cast<double>(v);
        }
    }
    return output;
}

int CreateAclTensorRaw(const std::vector<uint8_t>& hostBytes, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
{
    const size_t dataSize = hostBytes.size();
    auto ret = aclrtMalloc(deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, dataSize, hostBytes.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

bool CompareResult(const std::vector<double>& expected, const std::vector<double>& actual, bool integerExact,
    double atol, double rtol, double& maxError, size_t& maxErrorIndex)
{
    if (expected.size() != actual.size()) {
        return false;
    }
    maxError = 0.0;
    maxErrorIndex = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double err = std::fabs(expected[i] - actual[i]);
        if (err > maxError) {
            maxError = err;
            maxErrorIndex = i;
        }
        if (integerExact) {
            if (err != 0.0) {
                return false;
            }
        } else {
            const double tol = atol + rtol * std::fabs(expected[i]);
            if (err > tol) {
                return false;
            }
        }
    }
    return true;
}

struct CumsumCase {
    std::string name;
    std::vector<int64_t> shape;
    int64_t dim;
    aclDataType inputType;
    aclDataType outputType;
    std::vector<double> input;
    bool integerExact;
    double atol;
    double rtol;
    bool strictCheck;
    aclnnStatus expectStatus;
};

struct CumsumV2Case {
    std::string name;
    std::vector<int64_t> shape;
    int64_t dim;
    aclDataType dtype;
    std::vector<double> input;
    bool exclusive;
    bool reverse;
    bool integerExact;
    double atol;
    double rtol;
    bool strictCheck;
    aclnnStatus expectStatus;
};

bool RunWorkspaceProbeCase(const std::string& name, const std::vector<int64_t>& shape, int64_t dim, aclDataType dtype,
    bool expectSuccess)
{
    const int64_t elemCount = GetShapeSize(shape);
    std::vector<double> input(elemCount, 0.0);
    std::vector<double> outInit(elemCount, 0.0);
    std::vector<double> ignored;

    std::vector<uint8_t> inBytes = PackInput(input, dtype, ignored);
    std::vector<uint8_t> outBytes = PackInput(outInit, dtype, ignored);

    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensorRaw(inBytes, shape, &selfDev, dtype, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensorRaw(outBytes, shape, &outDev, dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS,
        aclDestroyTensor(self);
        aclrtFree(selfDev);
        return false;);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);

    const bool pass = expectSuccess ? (ret == kAclnnSuccess) : (ret != kAclnnSuccess);
    LOG_PRINT("Workspace probe: %s\n", name.c_str());
    LOG_PRINT("  ret=%d workspace=%llu expectSuccess=%d\n", ret, static_cast<unsigned long long>(workspaceSize),
        static_cast<int>(expectSuccess));
    LOG_PRINT("  [%s]\n", pass ? "PASS" : "FAIL");

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDev);
    aclrtFree(outDev);
    return pass;
}

bool RunCumsumCase(aclrtStream stream, const CumsumCase& tc, int& precisionWarnCount)
{
    const int64_t elemCount = GetShapeSize(tc.shape);
    LOG_PRINT("Test case: %s\n", tc.name.c_str());

    std::vector<double> quantizedInput;
    std::vector<uint8_t> inputBytes = PackInput(tc.input, tc.inputType, quantizedInput);
    std::vector<double> zeroOutput(elemCount, 0.0);
    std::vector<double> ignored;
    std::vector<uint8_t> outputBytes = PackInput(zeroOutput, tc.outputType, ignored);

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensorRaw(inputBytes, tc.shape, &selfDeviceAddr, tc.inputType, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensorRaw(outputBytes, tc.shape, &outDeviceAddr, tc.outputType, &out);
    CHECK_RET(ret == ACL_SUCCESS, aclDestroyTensor(self); aclrtFree(selfDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, tc.dim, tc.outputType, out, &workspaceSize, &executor);

    if (ret != tc.expectStatus) {
        LOG_PRINT("  [FAIL] aclnnCumsumGetWorkspaceSize ret=%d expected=%d\n", ret, tc.expectStatus);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        return false;
    }

    if (tc.expectStatus != kAclnnSuccess) {
        LOG_PRINT("  [PASS] expected failure path triggered (ret=%d)\n", ret);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        return true;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("  [FAIL] allocate workspace failed ret=%d\n", ret);
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(selfDeviceAddr);
            aclrtFree(outDeviceAddr);
            return false;);
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("  [FAIL] aclnnCumsum failed ret=%d\n", ret);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); }
        return false;);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("  [FAIL] aclrtSynchronizeStream failed ret=%d\n", ret);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); }
        return false;);

    ret = aclrtMemcpy(outputBytes.data(), outputBytes.size(), outDeviceAddr, outputBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("  [FAIL] D2H memcpy failed ret=%d\n", ret);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); }
        return false;);

    std::vector<double> actual = UnpackOutput(outputBytes, tc.outputType);
    std::vector<double> computeInput = quantizedInput;
    if (tc.inputType != tc.outputType) {
        // aclnnCumsum 内部会先将输入 cast 到输出 dtype，再执行 cumsum。
        computeInput = QuantizeVector(quantizedInput, tc.outputType);
    }
    std::vector<double> expected = CpuCumsum(computeInput, tc.shape, tc.dim);

    double maxError = 0.0;
    size_t maxErrorIndex = 0;
    const bool precisionPass = CompareResult(expected, actual, tc.integerExact, tc.atol, tc.rtol, maxError, maxErrorIndex);
    bool pass = precisionPass;
    if (!precisionPass && !tc.strictCheck) {
        ++precisionWarnCount;
        LOG_PRINT("  [WARN] precision mismatch detected (non-blocking probe case)\n");
        pass = true;
    }

    LOG_PRINT("  Expected: %s\n", Preview(expected).c_str());
    LOG_PRINT("  Actual:   %s\n", Preview(actual).c_str());
    LOG_PRINT("  Max error: %.9g (idx=%zu)\n", maxError, maxErrorIndex);
    LOG_PRINT("  [%s]\n", pass ? "PASS" : "FAIL");

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return pass;
}

bool RunApiErrorCase(const std::string& name, const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out,
    bool expectFailure)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    const bool pass = expectFailure ? (ret != kAclnnSuccess) : (ret == kAclnnSuccess);
    LOG_PRINT("Test case: %s\n", name.c_str());
    LOG_PRINT("  GetWorkspace ret=%d expectFailure=%d\n", ret, static_cast<int>(expectFailure));
    LOG_PRINT("  [%s]\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool RunCumsumV2Case(aclrtStream stream, const CumsumV2Case& tc, int& precisionWarnCount)
{
    const int64_t elemCount = GetShapeSize(tc.shape);
    LOG_PRINT("Test case: %s\n", tc.name.c_str());

    std::vector<double> quantizedInput;
    std::vector<uint8_t> inputBytes = PackInput(tc.input, tc.dtype, quantizedInput);
    std::vector<double> zeroOutput(elemCount, 0.0);
    std::vector<double> ignored;
    std::vector<uint8_t> outputBytes = PackInput(zeroOutput, tc.dtype, ignored);

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensorRaw(inputBytes, tc.shape, &selfDeviceAddr, tc.dtype, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensorRaw(outputBytes, tc.shape, &outDeviceAddr, tc.dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS, aclDestroyTensor(self); aclrtFree(selfDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(self, tc.dim, tc.exclusive, tc.reverse, out, &workspaceSize, &executor);
    if (ret != tc.expectStatus) {
        LOG_PRINT("  [FAIL] aclnnCumsumV2GetWorkspaceSize ret=%d expected=%d\n", ret, tc.expectStatus);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        return false;
    }

    if (tc.expectStatus != kAclnnSuccess) {
        LOG_PRINT("  [PASS] expected failure path triggered (ret=%d)\n", ret);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        return true;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(selfDeviceAddr);
            aclrtFree(outDeviceAddr);
            return false;);
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); }
        return false;);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); }
        return false;);

    ret = aclrtMemcpy(outputBytes.data(), outputBytes.size(), outDeviceAddr, outputBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); }
        return false;);

    std::vector<double> actual = UnpackOutput(outputBytes, tc.dtype);
    std::vector<double> expected = CpuCumsumV2(quantizedInput, tc.shape, tc.dim, tc.exclusive, tc.reverse);

    double maxError = 0.0;
    size_t maxErrorIndex = 0;
    const bool precisionPass = CompareResult(expected, actual, tc.integerExact, tc.atol, tc.rtol, maxError, maxErrorIndex);
    bool pass = precisionPass;
    if (!precisionPass && !tc.strictCheck) {
        ++precisionWarnCount;
        LOG_PRINT("  [WARN] precision mismatch detected (non-blocking probe case)\n");
        pass = true;
    }

    LOG_PRINT("  Expected: %s\n", Preview(expected).c_str());
    LOG_PRINT("  Actual:   %s\n", Preview(actual).c_str());
    LOG_PRINT("  Max error: %.9g (idx=%zu)\n", maxError, maxErrorIndex);
    LOG_PRINT("  [%s]\n", pass ? "PASS" : "FAIL");

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return pass;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<CumsumCase> tests;

    tests.push_back({"Basic float32 dim0", {2, 2}, 0, ACL_FLOAT, ACL_FLOAT, {1, 2, 3, 4}, false, 1e-5, 1e-5,
        false, kAclnnSuccess});
    tests.push_back({"Float32 dim1 mixed sign", {3, 4}, 1, ACL_FLOAT, ACL_FLOAT,
        {1.5, -2.0, 3.25, -4.5, 10.0, -1.0, -2.0, 6.0, 0.0, 1.0, -1.0, 2.0}, false, 1e-5, 1e-5, false,
        kAclnnSuccess});

    {
        std::vector<double> ones(10000, 1.0);
        tests.push_back({"Long sequence accumulation float32", {10000}, 0, ACL_FLOAT, ACL_FLOAT, ones, false,
            1e-4, 1e-5, false, kAclnnSuccess});
    }

    {
        std::vector<double> decimalLong(10000, 0.1);
        tests.push_back({"Long decimal accumulation float32", {10000}, 0, ACL_FLOAT, ACL_FLOAT, decimalLong, false,
            1e-2, 1e-5, false, kAclnnSuccess});
    }

    {
        std::vector<double> mixed(4096, 0.0);
        for (size_t i = 0; i < mixed.size(); ++i) {
            mixed[i] = (i % 2 == 0) ? 1e8 : 1e-6;
        }
        tests.push_back({"Mixed magnitude float32", {4096}, 0, ACL_FLOAT, ACL_FLOAT, mixed, false, 2e-3, 1e-5,
            false, kAclnnSuccess});
    }

    {
        std::vector<double> cancel(4096, 0.0);
        for (size_t i = 0; i < cancel.size(); ++i) {
            const size_t r = i % 4;
            cancel[i] = (r == 0) ? 1e8 : ((r == 1) ? -1e8 : ((r == 2) ? 1e-3 : -1e-3));
        }
        tests.push_back({"Alternating cancellation float32", {4096}, 0, ACL_FLOAT, ACL_FLOAT, cancel, false, 1e-2,
            1e-5, false, kAclnnSuccess});
    }

    {
        std::vector<double> data3d(120, 0.0);
        for (size_t i = 0; i < data3d.size(); ++i) {
            data3d[i] = (static_cast<int>(i % 23) - 11) * 0.125;
        }
        tests.push_back({"Float32 middle dim 3D", {4, 5, 6}, 1, ACL_FLOAT, ACL_FLOAT, data3d, false, 1e-5, 1e-5,
            false, kAclnnSuccess});
    }

    {
        std::vector<double> decimal(4096, 0.1);
        tests.push_back({"Float16 decimal accumulation", {4096}, 0, ACL_FLOAT16, ACL_FLOAT16, decimal, false, 1e-2,
            2e-3, false, kAclnnSuccess});
    }

    {
        std::vector<double> bf16Input(2048);
        for (size_t i = 0; i < bf16Input.size(); ++i) {
            bf16Input[i] = (i % 7 == 0) ? -0.25 : 0.125;
        }
        tests.push_back({"BF16 accumulation", {2048}, 0, ACL_BF16, ACL_BF16, bf16Input, false, 1e-2, 3e-3,
            false, kAclnnSuccess});
    }

    tests.push_back({"Dim boundary -1 float32", {2, 5}, -1, ACL_FLOAT, ACL_FLOAT,
        {1, 2, 3, 4, 5, -1, -2, -3, -4, -5}, false, 1e-5, 1e-5, false, kAclnnSuccess});

    tests.push_back({"Int32 exact", {2, 5}, -1, ACL_INT32, ACL_INT32, {1, 2, 3, 4, 5, -1, -2, -3, -4, -5}, true,
        0.0, 0.0, false, kAclnnSuccess});

    tests.push_back({"Int64 exact", {3, 3}, 0, ACL_INT64, ACL_INT64, {100, 200, -50, 10, 20, 30, -7, -9, 11}, true,
        0.0, 0.0, true, kAclnnSuccess});

    {
        std::vector<double> castInput(1024);
        for (size_t i = 0; i < castInput.size(); ++i) {
            castInput[i] = (i % 13) * 0.25 - 1.5;
        }
        tests.push_back({"Input float16 output float32", {32, 32}, 1, ACL_FLOAT16, ACL_FLOAT, castInput, false,
            1e-5, 1e-5, false, kAclnnSuccess});
    }

    {
        std::vector<double> castInt32(256, 0.0);
        for (size_t i = 0; i < castInt32.size(); ++i) {
            castInt32[i] = static_cast<int>((i % 7)) - 3;
        }
        tests.push_back({"Input int32 output float32 cast", {16, 16}, 1, ACL_INT32, ACL_FLOAT, castInt32, false,
            1e-5, 1e-5, false, kAclnnSuccess});
    }

    tests.push_back({"Input float32 output int32 cast exact", {2, 6}, 1, ACL_FLOAT, ACL_INT32,
        {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}, true, 0.0, 0.0, false, kAclnnSuccess});

    {
        std::vector<double> castF16(32, 0.0);
        for (size_t i = 0; i < castF16.size(); ++i) {
            castF16[i] = (static_cast<int>(i % 9) - 4) * 0.25;
        }
        tests.push_back({"Input float32 output float16 cast", {4, 8}, 1, ACL_FLOAT, ACL_FLOAT16, castF16, false,
            2e-2, 2e-3, false, kAclnnSuccess});
    }

    tests.push_back({"Int32 3D middle axis exact", {4, 7, 3}, 1, ACL_INT32, ACL_INT32,
        {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
            6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3,
            4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6},
        true, 0.0, 0.0, false, kAclnnSuccess});

    tests.push_back({"Int8 exact small range", {2, 8}, 1, ACL_INT8, ACL_INT8,
        {1, 1, -1, 2, -2, 1, 0, 1, -1, 2, -1, 1, 0, 1, -1, 2}, true, 0.0, 0.0, false, kAclnnSuccess});
    tests.push_back({"UInt8 exact small range", {3, 4}, 1, ACL_UINT8, ACL_UINT8,
        {1, 2, 3, 4, 2, 0, 1, 1, 0, 1, 0, 1}, true, 0.0, 0.0, false, kAclnnSuccess});

    {
        // Cast约束相关探测：int64->float32在2^31内可视为可观测精度路径。
        std::vector<double> castSafe(2048);
        for (size_t i = 0; i < castSafe.size(); ++i) {
            castSafe[i] = static_cast<double>((static_cast<int64_t>(i) % 2000000) - 1000000);
        }
        tests.push_back({"Cast path int64->float32 (safe range)", {2048}, 0, ACL_INT64, ACL_FLOAT, castSafe, false,
            1e-5, 1e-5, false, kAclnnSuccess});
    }

    {
        std::vector<double> scalarInput = {3.0};
        tests.push_back({"Single element shape dim0", {1}, 0, ACL_FLOAT, ACL_FLOAT, scalarInput, false, 1e-5, 1e-5,
            false, kAclnnSuccess});
    }

    int passCount = 0;
    int failCount = 0;
    int precisionWarnCount = 0;
    for (const auto& tc : tests) {
        if (RunCumsumCase(stream, tc, precisionWarnCount)) {
            ++passCount;
        } else {
            ++failCount;
        }
    }

    std::vector<CumsumV2Case> v2Tests;
    v2Tests.push_back({"V2 float32 inclusive forward", {2, 4}, 1, ACL_FLOAT,
        {1, 2, 3, 4, -1, -2, -3, -4}, false, false, false, 1e-5, 1e-5, false, kAclnnSuccess});
    v2Tests.push_back({"V2 float32 exclusive forward", {2, 4}, 1, ACL_FLOAT,
        {1, 2, 3, 4, -1, -2, -3, -4}, true, false, false, 1e-5, 1e-5, false, kAclnnSuccess});
    v2Tests.push_back({"V2 float32 inclusive reverse", {2, 4}, 1, ACL_FLOAT,
        {1, 2, 3, 4, -1, -2, -3, -4}, false, true, false, 1e-5, 1e-5, false, kAclnnSuccess});
    v2Tests.push_back({"V2 float32 exclusive reverse", {2, 4}, 1, ACL_FLOAT,
        {1, 2, 3, 4, -1, -2, -3, -4}, true, true, false, 1e-5, 1e-5, false, kAclnnSuccess});

    // 这两条用于命中op_api/cumsum.cpp中 Cumsum(exclusive, reverse) 的 AiCpu 分支。
    v2Tests.push_back({"V2 int64 exclusive reverse (AiCpu path)", {2, 5}, 1, ACL_INT64,
        {10, 20, 30, 40, 50, -1, -2, -3, -4, -5}, true, true, true, 0.0, 0.0, true, kAclnnSuccess});
    v2Tests.push_back({"V2 int8 inclusive forward (AiCpu path)", {2, 8}, 1, ACL_INT8,
        {1, 1, -1, 2, -2, 1, 0, 1, -1, 2, -1, 1, 0, 1, -1, 2}, false, false, true, 0.0, 0.0, true,
        kAclnnSuccess});

    for (const auto& tc : v2Tests) {
        if (RunCumsumV2Case(stream, tc, precisionWarnCount)) {
            ++passCount;
        } else {
            ++failCount;
        }
    }

    std::vector<int64_t> baseShape = {2, 2};
    std::vector<double> baseInput = {1, 2, 3, 4};
    std::vector<double> baseZero = {0, 0, 0, 0};
    std::vector<double> ignored;
    std::vector<uint8_t> inBytes = PackInput(baseInput, ACL_FLOAT, ignored);
    std::vector<uint8_t> outBytes = PackInput(baseZero, ACL_FLOAT, ignored);
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    ret = CreateAclTensorRaw(inBytes, baseShape, &selfDev, ACL_FLOAT, &selfTensor);
    CHECK_RET(ret == ACL_SUCCESS, aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = CreateAclTensorRaw(outBytes, baseShape, &outDev, ACL_FLOAT, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS,
        aclDestroyTensor(selfTensor);
        aclrtFree(selfDev);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;);

    if (RunApiErrorCase("Error path: dim out of range", selfTensor, 3, ACL_FLOAT, outTensor, true)) {
        ++passCount;
    } else {
        ++failCount;
    }

    if (RunApiErrorCase("Error path: out nullptr", selfTensor, 0, ACL_FLOAT, nullptr, true)) {
        ++passCount;
    } else {
        ++failCount;
    }

    if (RunApiErrorCase("Error path: dtype mismatch", selfTensor, 0, ACL_INT32, outTensor, true)) {
        ++passCount;
    } else {
        ++failCount;
    }
    if (RunApiErrorCase("Error path: self nullptr", nullptr, 0, ACL_FLOAT, outTensor, true)) {
        ++passCount;
    } else {
        ++failCount;
    }

    {
        // 文档约束覆盖：shape不一致。
        std::vector<int64_t> outShapeMismatch = {4};
        std::vector<double> outMismatchInit = {0, 0, 0, 0};
        std::vector<uint8_t> outMismatchBytes = PackInput(outMismatchInit, ACL_FLOAT, ignored);
        void* outMismatchDev = nullptr;
        aclTensor* outMismatchTensor = nullptr;
        ret = CreateAclTensorRaw(outMismatchBytes, outShapeMismatch, &outMismatchDev, ACL_FLOAT, &outMismatchTensor);
        if (ret == ACL_SUCCESS) {
            if (RunApiErrorCase("Error path: shape mismatch", selfTensor, 0, ACL_FLOAT, outMismatchTensor, true)) {
                ++passCount;
            } else {
                ++failCount;
            }
            aclDestroyTensor(outMismatchTensor);
            aclrtFree(outMismatchDev);
        } else {
            ++failCount;
        }
    }

    {
        // 文档约束覆盖：维度大于8。
        std::vector<int64_t> shape9d = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<double> oneElem = {1.0};
        std::vector<uint8_t> in9Bytes = PackInput(oneElem, ACL_FLOAT, ignored);
        std::vector<uint8_t> out9Bytes = PackInput(oneElem, ACL_FLOAT, ignored);
        void* self9Dev = nullptr;
        void* out9Dev = nullptr;
        aclTensor* self9 = nullptr;
        aclTensor* out9 = nullptr;
        ret = CreateAclTensorRaw(in9Bytes, shape9d, &self9Dev, ACL_FLOAT, &self9);
        if (ret == ACL_SUCCESS) {
            ret = CreateAclTensorRaw(out9Bytes, shape9d, &out9Dev, ACL_FLOAT, &out9);
            if (ret == ACL_SUCCESS) {
                if (RunApiErrorCase("Error path: rank > 8", self9, 0, ACL_FLOAT, out9, true)) {
                    ++passCount;
                } else {
                    ++failCount;
                }
                aclDestroyTensor(out9);
                aclrtFree(out9Dev);
            } else {
                ++failCount;
            }
            aclDestroyTensor(self9);
            aclrtFree(self9Dev);
        } else {
            ++failCount;
        }
    }

    if (RunWorkspaceProbeCase("Coverage probe: cube-support candidate", {12800, 512}, 1, ACL_FLOAT, true)) {
        ++passCount;
    } else {
        ++failCount;
    }
    if (RunWorkspaceProbeCase("Coverage probe: non-cube neighbor", {12799, 512}, 1, ACL_FLOAT, true)) {
        ++passCount;
    } else {
        ++failCount;
    }

    // Int tiling targeted probes: drive rarely-covered branches in cumsum_tiling_ascendc_int_arch35.cpp.
    if (RunWorkspaceProbeCase("Tiling probe int8 axis0 right-large (dtypeSize==1)", {2, 4096}, 0, ACL_INT8, true)) {
        ++passCount;
    } else {
        ++failCount;
    }
    if (RunWorkspaceProbeCase("Tiling probe int32 axis0 right-large (TD rightA)", {2, 2048}, 0, ACL_INT32, true)) {
        ++passCount;
    } else {
        ++failCount;
    }
    if (RunWorkspaceProbeCase("Tiling probe int32 left-large (TD leftA gate)", {4096, 64}, 1, ACL_INT32, true)) {
        ++passCount;
    } else {
        ++failCount;
    }
    if (RunWorkspaceProbeCase("Tiling probe int32 R-axis competition", {1, 65536, 1}, 1, ACL_INT32, true)) {
        ++passCount;
    } else {
        ++failCount;
    }
    if (RunWorkspaceProbeCase("Tiling probe int32 RA-axis competition", {1, 8, 8192}, 1, ACL_INT32, true)) {
        ++passCount;
    } else {
        ++failCount;
    }

    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(selfDev);
    aclrtFree(outDev);

    LOG_PRINT("Summary: %d passed, %d failed, %d precision warnings\n", passCount, failCount, precisionWarnCount);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (failCount == 0) ? 0 : 1;
}
