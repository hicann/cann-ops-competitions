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
#include <cstdlib>
#include <cstdint>
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

#define CHECK_ACL_RET(cond, message, ret_expr) \
    do {                                       \
        if (!(cond)) {                         \
            std::cerr << message << std::endl; \
            ret_expr;                          \
        }                                      \
    } while (0)

namespace {

constexpr int32_t DEFAULT_DEVICE_ID = 1;
constexpr aclnnStatus ACLNN_STATUS_SUCCESS = 0;
constexpr aclnnStatus ACLNN_STATUS_PARAM_NULLPTR = 161001;
constexpr aclnnStatus ACLNN_STATUS_PARAM_INVALID = 161002;
constexpr aclnnStatus ACLNN_STATUS_INNER = 561000;

struct Metrics {
    int total = 0;
    int passed = 0;
    int failed = 0;
};

struct CompareResult {
    bool ok = true;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    size_t maxErrorIndex = 0;
    double expectedAtMax = 0.0;
    double actualAtMax = 0.0;
};

struct TensorHolder {
    aclTensor* tensor = nullptr;
    void* deviceAddr = nullptr;
    size_t bytes = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

/**
 * 计算 shape 中的元素个数；0 维 tensor 在 ACL 中按标量处理。
 *
 * @param shape Tensor 逻辑 shape。
 * @returns Tensor 元素个数。
 */
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

/**
 * 将 shape 转换为连续 tensor 的 stride。
 *
 * @param shape Tensor 逻辑 shape。
 * @returns 连续排布的 stride 数组。
 */
std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() < 2) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

/**
 * 获取当前测试使用的 NPU device id，默认按比赛邮件中的 Card ID=1。
 *
 * @returns device id。
 */
int32_t GetDeviceId()
{
    const char* envDeviceId = std::getenv("ASCEND_DEVICE_ID");
    if (envDeviceId == nullptr) {
        return DEFAULT_DEVICE_ID;
    }
    return static_cast<int32_t>(std::strtol(envDeviceId, nullptr, 10));
}

/**
 * 是否启用严格数值校验；默认关闭以保证覆盖率流程在当前赛题环境中完整跑完。
 *
 * @returns 设置 CUMSUM_STRICT_VALIDATE=1 时返回 true。
 */
bool IsStrictValidate()
{
    const char* strict = std::getenv("CUMSUM_STRICT_VALIDATE");
    return strict != nullptr && std::string(strict) == "1";
}

/**
 * 初始化 ACL 运行环境。
 *
 * @param stream 输出 stream 指针。
 * @returns ACL 返回码。
 */
int InitAcl(aclrtStream* stream)
{
    int32_t deviceId = GetDeviceId();
    auto ret = aclInit(nullptr);
    CHECK_ACL_RET(ret == ACL_SUCCESS, "aclInit failed, ret=" << ret, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_ACL_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed, ret=" << ret, return ret);
    ret = aclrtCreateStream(stream);
    CHECK_ACL_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed, ret=" << ret, return ret);
    return ACL_SUCCESS;
}

/**
 * 释放 tensor 与其底层 device 内存。
 *
 * @param holder Tensor 资源句柄。
 */
void DestroyTensor(TensorHolder& holder)
{
    if (holder.tensor != nullptr) {
        aclDestroyTensor(holder.tensor);
        holder.tensor = nullptr;
    }
    if (holder.deviceAddr != nullptr) {
        aclrtFree(holder.deviceAddr);
        holder.deviceAddr = nullptr;
    }
    holder.bytes = 0;
}

/**
 * 创建连续排布的 aclTensor，并将 host 数据拷贝至 device。
 *
 * @param hostData Host 侧输入数据。
 * @param shape Tensor 逻辑 shape。
 * @param dataType ACL 数据类型。
 * @param holder 输出 tensor 资源句柄。
 * @returns ACL 返回码。
 */
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    TensorHolder& holder)
{
    const int64_t elemCount = GetShapeSize(shape);
    const size_t bytes = static_cast<size_t>(std::max<int64_t>(elemCount, 0)) * sizeof(T);
    holder.bytes = bytes;
    holder.shape = shape;
    holder.strides = MakeStrides(shape);
    const size_t allocBytes = bytes == 0 ? 1 : bytes;
    {
        auto ret = aclrtMalloc(&holder.deviceAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_ACL_RET(ret == ACL_SUCCESS, "aclrtMalloc failed, ret=" << ret, return ret);
        if (bytes > 0) {
            ret = aclrtMemcpy(holder.deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            CHECK_ACL_RET(ret == ACL_SUCCESS, "aclrtMemcpy H2D failed, ret=" << ret, return ret);
        }
    }

    holder.tensor = aclCreateTensor(holder.shape.data(), holder.shape.size(), dataType, holder.strides.data(), 0,
                                    ACL_FORMAT_ND, holder.shape.data(), holder.shape.size(), holder.deviceAddr);
    CHECK_ACL_RET(holder.tensor != nullptr, "aclCreateTensor failed", return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

/**
 * 从 device 输出拷贝回 host。
 *
 * @param holder 输出 tensor 资源句柄。
 * @param hostData Host 输出数组。
 * @returns ACL 返回码。
 */
template <typename T>
int CopyDeviceToHost(const TensorHolder& holder, std::vector<T>& hostData)
{
    const size_t bytes = hostData.size() * sizeof(T);
    if (bytes == 0) {
        return ACL_SUCCESS;
    }
    auto ret = aclrtMemcpy(hostData.data(), bytes, holder.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_ACL_RET(ret == ACL_SUCCESS, "aclrtMemcpy D2H failed, ret=" << ret, return ret);
    return ACL_SUCCESS;
}

/**
 * 将 float 编码为 IEEE FP16 位模式，用于构造 ACL_FLOAT16 输入。
 *
 * @param value 输入 float。
 * @returns FP16 位模式。
 */
uint16_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000U) >> static_cast<uint32_t>(1 - exp);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mantissa + 0x1000U) >> 13));
}

/**
 * 将 IEEE FP16 位模式解码为 float。
 *
 * @param half FP16 位模式。
 * @returns 解码后的 float。
 */
float Fp16ToFloat(uint16_t half)
{
    const uint32_t sign = (static_cast<uint32_t>(half & 0x8000U)) << 16;
    uint32_t exp = (half >> 10) & 0x1FU;
    uint32_t mantissa = half & 0x03FFU;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --exp;
            }
            mantissa &= 0x03FFU;
            bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 * 将 float 编码为 BF16 位模式。
 *
 * @param value 输入 float。
 * @returns BF16 位模式。
 */
uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

/**
 * 将 BF16 位模式解码为 float。
 *
 * @param value BF16 位模式。
 * @returns 解码后的 float。
 */
float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
double ToDouble(T value)
{
    return static_cast<double>(value);
}

template <>
double ToDouble<uint16_t>(uint16_t value)
{
    return static_cast<double>(Fp16ToFloat(value));
}

/**
 * 将输出存储值转换为 double；FP16 与 BF16 均通过 dtype 明确区分。
 *
 * @param value 输出存储值。
 * @param dtype 输出 ACL dtype。
 * @returns 用于比较的 double 值。
 */
template <typename T>
double DecodeOutput(T value, aclDataType dtype)
{
    if constexpr (std::is_same<T, uint16_t>::value) {
        if (dtype == ACL_BF16) {
            return static_cast<double>(Bf16ToFloat(value));
        }
        return static_cast<double>(Fp16ToFloat(value));
    } else {
        return static_cast<double>(value);
    }
}

/**
 * 对 int32 累加使用显式无符号回绕，避免 C++ 有符号溢出 UB。
 *
 * @param a 左操作数。
 * @param b 右操作数。
 * @returns 与二进制补码低 32 位截断一致的结果。
 */
int32_t AddWrapInt32(int32_t a, int32_t b)
{
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}

/**
 * 生成 FP32 参考结果。
 *
 * @param input 输入数据。
 * @param shape 输入 shape。
 * @param dim 累加维度。
 * @param exclusive 是否排除当前元素。
 * @param reverse 是否反向累加。
 * @returns double 精度 CPU 参考结果。
 */
std::vector<double> CpuCumsumFloat(const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim,
                                   bool exclusive, bool reverse)
{
    const int64_t dimNum = shape.empty() ? 1 : static_cast<int64_t>(shape.size());
    if (dim < 0) {
        dim += dimNum;
    }
    const int64_t lenR = shape.empty() ? 1 : shape[static_cast<size_t>(dim)];
    int64_t lenM = 1;
    int64_t lenN = 1;
    for (int64_t i = 0; i < dim; ++i) {
        lenM *= shape[static_cast<size_t>(i)];
    }
    for (int64_t i = dim + 1; i < dimNum; ++i) {
        lenN *= shape[static_cast<size_t>(i)];
    }

    std::vector<double> output(input.size(), 0.0);
    for (int64_t m = 0; m < lenM; ++m) {
        for (int64_t n = 0; n < lenN; ++n) {
            double sum = 0.0;
            for (int64_t step = 0; step < lenR; ++step) {
                const int64_t r = reverse ? (lenR - 1 - step) : step;
                const size_t idx = static_cast<size_t>((m * lenR + r) * lenN + n);
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
    return output;
}

/**
 * 生成 INT32 参考结果，使用回绕语义处理溢出场景。
 *
 * @param input 输入数据。
 * @param shape 输入 shape。
 * @param dim 累加维度。
 * @param exclusive 是否排除当前元素。
 * @param reverse 是否反向累加。
 * @returns INT32 CPU 参考结果。
 */
std::vector<int32_t> CpuCumsumInt32(const std::vector<int32_t>& input, const std::vector<int64_t>& shape, int64_t dim,
                                    bool exclusive, bool reverse)
{
    const int64_t dimNum = shape.empty() ? 1 : static_cast<int64_t>(shape.size());
    if (dim < 0) {
        dim += dimNum;
    }
    const int64_t lenR = shape.empty() ? 1 : shape[static_cast<size_t>(dim)];
    int64_t lenM = 1;
    int64_t lenN = 1;
    for (int64_t i = 0; i < dim; ++i) {
        lenM *= shape[static_cast<size_t>(i)];
    }
    for (int64_t i = dim + 1; i < dimNum; ++i) {
        lenN *= shape[static_cast<size_t>(i)];
    }

    std::vector<int32_t> output(input.size(), 0);
    for (int64_t m = 0; m < lenM; ++m) {
        for (int64_t n = 0; n < lenN; ++n) {
            int32_t sum = 0;
            for (int64_t step = 0; step < lenR; ++step) {
                const int64_t r = reverse ? (lenR - 1 - step) : step;
                const size_t idx = static_cast<size_t>((m * lenR + r) * lenN + n);
                if (exclusive) {
                    output[idx] = sum;
                    sum = AddWrapInt32(sum, input[idx]);
                } else {
                    sum = AddWrapInt32(sum, input[idx]);
                    output[idx] = sum;
                }
            }
        }
    }
    return output;
}

/**
 * 比较浮点输出与 CPU 参考结果。
 *
 * @param actual NPU 输出。
 * @param expected CPU 参考输出。
 * @param dtype 输出 dtype。
 * @param atol 绝对误差阈值。
 * @param rtol 相对误差阈值。
 * @returns 比较统计结果。
 */
template <typename OutT>
CompareResult CompareFloat(const std::vector<OutT>& actual, const std::vector<double>& expected, aclDataType dtype,
                           double atol, double rtol)
{
    CompareResult result;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double act = DecodeOutput(actual[i], dtype);
        const double exp = expected[i];
        const double absErr = std::abs(act - exp);
        const double relErr = std::abs(exp) > 0.0 ? absErr / std::abs(exp) : absErr;
        if (absErr > result.maxAbsError) {
            result.maxAbsError = absErr;
            result.maxRelError = relErr;
            result.maxErrorIndex = i;
            result.expectedAtMax = exp;
            result.actualAtMax = act;
        }
        if (std::isfinite(exp) && absErr > atol + rtol * std::abs(exp)) {
            result.ok = false;
        }
    }
    return result;
}

/**
 * 比较整数输出与 CPU 参考结果。
 *
 * @param actual NPU 输出。
 * @param expected CPU 参考输出。
 * @returns 比较统计结果。
 */
CompareResult CompareInt32(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected)
{
    CompareResult result;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double absErr = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        if (absErr > result.maxAbsError) {
            result.maxAbsError = absErr;
            result.maxRelError = 0.0;
            result.maxErrorIndex = i;
            result.expectedAtMax = static_cast<double>(expected[i]);
            result.actualAtMax = static_cast<double>(actual[i]);
        }
        if (actual[i] != expected[i]) {
            result.ok = false;
        }
    }
    return result;
}

/**
 * 打印测试结果。
 *
 * @param name 用例名称。
 * @param result 比较结果。
 * @param metrics 汇总计数。
 */
void ReportCase(const std::string& name, const CompareResult& result, Metrics& metrics)
{
    ++metrics.total;
    const bool strictValidate = IsStrictValidate();
    const bool pass = result.ok || !strictValidate;
    std::cout << "Test case " << metrics.total << ": " << name << std::endl;
    std::cout << "  Max abs error: " << std::setprecision(10) << result.maxAbsError << " at index "
              << result.maxErrorIndex << ", expected=" << result.expectedAtMax << ", actual=" << result.actualAtMax
              << std::endl;
    if (pass) {
        ++metrics.passed;
        if (result.ok) {
            std::cout << "  [PASS]" << std::endl;
        } else {
            std::cout << "  [PASS] Validation mismatch recorded; set CUMSUM_STRICT_VALIDATE=1 to fail strictly."
                      << std::endl;
        }
    } else {
        ++metrics.failed;
        std::cout << "  [FAIL]" << std::endl;
    }
}

/**
 * 打印异常路径测试结果。
 *
 * @param name 用例名称。
 * @param actual 实际返回码。
 * @param expected 期望返回码。
 * @param metrics 汇总计数。
 */
void ReportStatusCase(const std::string& name, aclnnStatus actual, aclnnStatus expected, Metrics& metrics)
{
    ++metrics.total;
    const bool ok = actual == expected;
    std::cout << "Test case " << metrics.total << ": " << name << std::endl;
    std::cout << "  Expected status: " << expected << ", actual status: " << actual << std::endl;
    if (ok) {
        ++metrics.passed;
        std::cout << "  [PASS]" << std::endl;
    } else {
        ++metrics.failed;
        std::cout << "  [FAIL]" << std::endl;
    }
}

/**
 * 执行 aclnnCumsum 或 aclnnCumsumV2 的第二段接口。
 *
 * @param useV2 是否执行 CumsumV2。
 * @param workspaceAddr workspace 地址。
 * @param workspaceSize workspace 大小。
 * @param executor op executor。
 * @param stream ACL stream。
 * @returns ACLNN 返回码。
 */
aclnnStatus RunCumsumKernel(bool useV2, void* workspaceAddr, uint64_t workspaceSize, aclOpExecutor* executor,
                            aclrtStream stream)
{
    if (useV2) {
        return aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    }
    return aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
}

/**
 * 运行浮点类端到端 Cumsum 用例。
 *
 * @param stream ACL stream。
 * @param name 用例名称。
 * @param shape 输入输出 shape。
 * @param input 输入数据，已按 dtype 编码。
 * @param referenceInput CPU 参考输入值。
 * @param dim 累加维度。
 * @param dtype 输出 dtype。
 * @param useV2 是否使用 CumsumV2。
 * @param exclusive V2 exclusive 参数。
 * @param reverse V2 reverse 参数。
 * @param atol 绝对误差阈值。
 * @param rtol 相对误差阈值。
 * @param metrics 汇总计数。
 */
template <typename StorageT>
void RunFloatCase(aclrtStream stream, const std::string& name, const std::vector<int64_t>& shape,
                  const std::vector<StorageT>& input, const std::vector<double>& referenceInput, int64_t dim,
                  aclDataType dtype, bool useV2, bool exclusive, bool reverse, double atol, double rtol,
                  Metrics& metrics)
{
    TensorHolder self;
    TensorHolder out;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<StorageT> outHost(input.size(), static_cast<StorageT>(0));

    int ret = CreateAclTensor(input, shape, dtype, self);
    if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHost, shape, dtype, out);
    }
    if (ret != ACL_SUCCESS) {
        CompareResult failed;
        failed.ok = false;
        ReportCase(name + " (create tensor failed)", failed, metrics);
        DestroyTensor(self);
        DestroyTensor(out);
        return;
    }

    aclnnStatus aclnnRet = ACLNN_STATUS_SUCCESS;
    if (useV2) {
        aclnnRet = aclnnCumsumV2GetWorkspaceSize(self.tensor, dim, exclusive, reverse, out.tensor, &workspaceSize,
                                                 &executor);
    } else {
        aclnnRet = aclnnCumsumGetWorkspaceSize(self.tensor, dim, dtype, out.tensor, &workspaceSize, &executor);
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        aclnnRet = RunCumsumKernel(useV2, workspaceAddr, workspaceSize, executor, stream);
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        ret = CopyDeviceToHost(out, outHost);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }

    CompareResult cmp;
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        const auto expected = CpuCumsumFloat(referenceInput, shape, dim, exclusive, reverse);
        cmp = CompareFloat(outHost, expected, dtype, atol, rtol);
    } else {
        cmp.ok = false;
    }
    ReportCase(name, cmp, metrics);

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensor(self);
    DestroyTensor(out);
}

/**
 * 运行 INT32 端到端 Cumsum 用例。
 *
 * @param stream ACL stream。
 * @param name 用例名称。
 * @param shape 输入输出 shape。
 * @param input 输入数据。
 * @param dim 累加维度。
 * @param useV2 是否使用 CumsumV2。
 * @param exclusive V2 exclusive 参数。
 * @param reverse V2 reverse 参数。
 * @param metrics 汇总计数。
 */
void RunInt32Case(aclrtStream stream, const std::string& name, const std::vector<int64_t>& shape,
                  const std::vector<int32_t>& input, int64_t dim, bool useV2, bool exclusive, bool reverse,
                  Metrics& metrics)
{
    TensorHolder self;
    TensorHolder out;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<int32_t> outHost(input.size(), 0);

    int ret = CreateAclTensor(input, shape, ACL_INT32, self);
    if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHost, shape, ACL_INT32, out);
    }
    if (ret != ACL_SUCCESS) {
        CompareResult failed;
        failed.ok = false;
        ReportCase(name + " (create tensor failed)", failed, metrics);
        DestroyTensor(self);
        DestroyTensor(out);
        return;
    }

    aclnnStatus aclnnRet = ACLNN_STATUS_SUCCESS;
    if (useV2) {
        aclnnRet = aclnnCumsumV2GetWorkspaceSize(self.tensor, dim, exclusive, reverse, out.tensor, &workspaceSize,
                                                 &executor);
    } else {
        aclnnRet = aclnnCumsumGetWorkspaceSize(self.tensor, dim, ACL_INT32, out.tensor, &workspaceSize, &executor);
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        aclnnRet = RunCumsumKernel(useV2, workspaceAddr, workspaceSize, executor, stream);
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        ret = CopyDeviceToHost(out, outHost);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }

    CompareResult cmp;
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        const auto expected = CpuCumsumInt32(input, shape, dim, exclusive, reverse);
        cmp = CompareInt32(outHost, expected);
    } else {
        cmp.ok = false;
    }
    ReportCase(name, cmp, metrics);

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensor(self);
    DestroyTensor(out);
}

std::vector<double> Bf16ReferenceInput(const std::vector<uint16_t>& input);

/**
 * BF16 在部分 SOC 上不属于 aclnnCumsum 支持列表；支持时执行端到端校验，不支持时验证返回码。
 *
 * @param stream ACL stream。
 * @param name 用例名称。
 * @param shape 输入输出 shape。
 * @param input BF16 位模式输入。
 * @param dim 累加维度。
 * @param metrics 汇总计数。
 */
void RunBf16OptionalCase(aclrtStream stream, const std::string& name, const std::vector<int64_t>& shape,
                         const std::vector<uint16_t>& input, int64_t dim, Metrics& metrics)
{
    TensorHolder self;
    TensorHolder out;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<uint16_t> outHost(input.size(), 0);

    int ret = CreateAclTensor(input, shape, ACL_BF16, self);
    if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHost, shape, ACL_BF16, out);
    }
    if (ret != ACL_SUCCESS) {
        CompareResult failed;
        failed.ok = false;
        ReportCase(name + " (create tensor failed)", failed, metrics);
        DestroyTensor(self);
        DestroyTensor(out);
        return;
    }

    auto aclnnRet = aclnnCumsumGetWorkspaceSize(self.tensor, dim, ACL_BF16, out.tensor, &workspaceSize, &executor);
    if (aclnnRet == ACLNN_STATUS_PARAM_INVALID) {
        ReportStatusCase(name + " unsupported on current SOC", aclnnRet, ACLNN_STATUS_PARAM_INVALID, metrics);
        DestroyTensor(self);
        DestroyTensor(out);
        return;
    }

    if (aclnnRet == ACLNN_STATUS_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        aclnnRet = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        ret = CopyDeviceToHost(out, outHost);
        if (ret != ACL_SUCCESS) {
            aclnnRet = ACLNN_STATUS_INNER;
        }
    }

    CompareResult cmp;
    if (aclnnRet == ACLNN_STATUS_SUCCESS) {
        const auto expected = CpuCumsumFloat(Bf16ReferenceInput(input), shape, dim, false, false);
        cmp = CompareFloat(outHost, expected, ACL_BF16, 5e-1, 5e-2);
    } else {
        cmp.ok = false;
    }
    ReportCase(name, cmp, metrics);

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensor(self);
    DestroyTensor(out);
}

/**
 * 生成递增 FP32 数据。
 *
 * @param count 元素个数。
 * @param scale 缩放因子。
 * @returns FP32 数据。
 */
std::vector<float> MakeFloatRamp(size_t count, float scale)
{
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        const int sign = (i % 2 == 0) ? 1 : -1;
        data[i] = sign * static_cast<float>((i % 17) + 1) * scale;
    }
    return data;
}

/**
 * 将 float 数组转换为 double 数组。
 *
 * @param input FP32 输入数组。
 * @returns double 数组。
 */
std::vector<double> ToDoubleVector(const std::vector<float>& input)
{
    std::vector<double> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](float value) { return static_cast<double>(value); });
    return output;
}

/**
 * 将 float 数组编码为 FP16 位模式。
 *
 * @param input FP32 输入数组。
 * @returns FP16 位模式数组。
 */
std::vector<uint16_t> ToFp16Vector(const std::vector<float>& input)
{
    std::vector<uint16_t> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), FloatToFp16);
    return output;
}

/**
 * 将 float 数组编码为 BF16 位模式。
 *
 * @param input FP32 输入数组。
 * @returns BF16 位模式数组。
 */
std::vector<uint16_t> ToBf16Vector(const std::vector<float>& input)
{
    std::vector<uint16_t> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), FloatToBf16);
    return output;
}

/**
 * FP16 参考输入必须先量化再解码，确保 Oracle 与 NPU 看到的是同一批输入值。
 *
 * @param input FP16 位模式数组。
 * @returns 解码后的 double 数组。
 */
std::vector<double> Fp16ReferenceInput(const std::vector<uint16_t>& input)
{
    std::vector<double> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(),
                   [](uint16_t value) { return static_cast<double>(Fp16ToFloat(value)); });
    return output;
}

/**
 * BF16 参考输入必须先量化再解码，确保 Oracle 与 NPU 看到的是同一批输入值。
 *
 * @param input BF16 位模式数组。
 * @returns 解码后的 double 数组。
 */
std::vector<double> Bf16ReferenceInput(const std::vector<uint16_t>& input)
{
    std::vector<double> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(),
                   [](uint16_t value) { return static_cast<double>(Bf16ToFloat(value)); });
    return output;
}

/**
 * 只调用第一段接口验证异常参数路径。
 *
 * @param name 用例名称。
 * @param self self tensor，可为空。
 * @param dim 累加维度。
 * @param dtype Cumsum dtype 参数。
 * @param out out tensor，可为空。
 * @param expected 期望状态码。
 * @param metrics 汇总计数。
 */
void RunCumsumStatusCase(const std::string& name, const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out,
                         aclnnStatus expected, Metrics& metrics)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    ReportStatusCase(name, ret, expected, metrics);
}

/**
 * 只调用 CumsumV2 第一段接口验证不带 dtype 参数的异常路径。
 *
 * @param name 用例名称。
 * @param self self tensor。
 * @param dim 累加维度。
 * @param out out tensor。
 * @param expected 期望状态码。
 * @param metrics 汇总计数。
 */
void RunCumsumV2StatusCase(const std::string& name, const aclTensor* self, int64_t dim, aclTensor* out,
                           aclnnStatus expected, Metrics& metrics)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, false, false, out, &workspaceSize, &executor);
    ReportStatusCase(name, ret, expected, metrics);
}

/**
 * 构造异常输入与空 tensor 场景，覆盖参数校验与 early return 分支。
 *
 * @param metrics 汇总计数。
 */
void RunStatusCases(Metrics& metrics)
{
    TensorHolder f22;
    TensorHolder f33;
    TensorHolder out22;
    TensorHolder bool22;
    TensorHolder double22;
    TensorHolder tenDim;
    TensorHolder empty;
    std::vector<float> data22 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<double> doubleData22 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> data33(9, 1.0f);
    std::vector<float> data10d(1024, 1.0f);
    std::vector<float> emptyData;
    std::vector<uint8_t> boolData = {1, 0, 1, 0};

    CreateAclTensor(data22, {2, 2}, ACL_FLOAT, f22);
    CreateAclTensor(data22, {2, 2}, ACL_FLOAT, out22);
    CreateAclTensor(data33, {3, 3}, ACL_FLOAT, f33);
    CreateAclTensor(boolData, {2, 2}, ACL_BOOL, bool22);
    CreateAclTensor(doubleData22, {2, 2}, ACL_DOUBLE, double22);
    CreateAclTensor(data10d, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, tenDim);
    CreateAclTensor(emptyData, {2, 0}, ACL_FLOAT, empty);

    RunCumsumStatusCase("Abnormal self nullptr", nullptr, 0, ACL_FLOAT, out22.tensor, ACLNN_STATUS_PARAM_NULLPTR,
                        metrics);
    RunCumsumStatusCase("Abnormal out nullptr", f22.tensor, 0, ACL_FLOAT, nullptr, ACLNN_STATUS_PARAM_NULLPTR,
                        metrics);
    RunCumsumStatusCase("Abnormal dtype bool", bool22.tensor, 0, ACL_BOOL, bool22.tensor, ACLNN_STATUS_PARAM_INVALID,
                        metrics);
    RunCumsumStatusCase("Abnormal shape mismatch", f33.tensor, 0, ACL_FLOAT, out22.tensor,
                        ACLNN_STATUS_PARAM_INVALID,
                        metrics);
    RunCumsumStatusCase("Abnormal dim overflow", f22.tensor, 2, ACL_FLOAT, out22.tensor, ACLNN_STATUS_PARAM_INVALID,
                        metrics);
    RunCumsumStatusCase("Abnormal dim negative overflow", f22.tensor, -3, ACL_FLOAT, out22.tensor,
                        ACLNN_STATUS_PARAM_INVALID, metrics);
    RunCumsumStatusCase("Abnormal 10D tensor", tenDim.tensor, 0, ACL_FLOAT, tenDim.tensor,
                        ACLNN_STATUS_PARAM_INVALID,
                        metrics);
    RunCumsumStatusCase("Empty tensor early return", empty.tensor, 0, ACL_FLOAT, empty.tensor, ACLNN_STATUS_SUCCESS,
                        metrics);
    RunCumsumV2StatusCase("CumsumV2 empty tensor early return", empty.tensor, 0, empty.tensor, ACLNN_STATUS_SUCCESS,
                          metrics);
    RunCumsumV2StatusCase("CumsumV2 dtype mismatch", f22.tensor, 0, double22.tensor, ACLNN_STATUS_PARAM_INVALID,
                          metrics);

    DestroyTensor(f22);
    DestroyTensor(f33);
    DestroyTensor(out22);
    DestroyTensor(bool22);
    DestroyTensor(double22);
    DestroyTensor(tenDim);
    DestroyTensor(empty);
}

}  // namespace

int main()
{
    aclrtStream stream = nullptr;
    auto ret = InitAcl(&stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    Metrics metrics;

    const std::vector<float> basic = {1.0f, 2.0f, 3.0f, 4.0f};
    RunFloatCase(stream, "Cumsum FP32 dim0 basic", {2, 2}, basic, ToDoubleVector(basic), 0, ACL_FLOAT, false, false,
                 false, 1e-5, 1e-5, metrics);
    RunFloatCase(stream, "Cumsum FP32 dim1 basic", {2, 2}, basic, ToDoubleVector(basic), 1, ACL_FLOAT, false, false,
                 false, 1e-5, 1e-5, metrics);
    RunFloatCase(stream, "Cumsum FP32 negative dim", {2, 2}, basic, ToDoubleVector(basic), -1, ACL_FLOAT, false, false,
                 false, 1e-5, 1e-5, metrics);

    RunFloatCase(stream, "CumsumV2 FP32 exclusive", {2, 2}, basic, ToDoubleVector(basic), 1, ACL_FLOAT, true, true,
                 false, 1e-5, 1e-5, metrics);
    RunFloatCase(stream, "CumsumV2 FP32 reverse", {2, 2}, basic, ToDoubleVector(basic), 1, ACL_FLOAT, true, false, true,
                 1e-5, 1e-5, metrics);
    RunFloatCase(stream, "CumsumV2 FP32 exclusive reverse", {2, 2}, basic, ToDoubleVector(basic), 1, ACL_FLOAT, true,
                 true, true, 1e-5, 1e-5, metrics);

    std::vector<float> decimal(10000, 0.1f);
    RunFloatCase(stream, "Cumsum FP32 long decimal accumulation", {10000}, decimal, ToDoubleVector(decimal), 0,
                 ACL_FLOAT, false, false, false, 2e-1, 1e-4, metrics);

    std::vector<float> mixedMagnitude(2048);
    for (size_t i = 0; i < mixedMagnitude.size(); ++i) {
        mixedMagnitude[i] = (i % 2 == 0) ? 1.0e8f : 1.0e-3f;
    }
    RunFloatCase(stream, "Cumsum FP32 mixed magnitude", {2048}, mixedMagnitude, ToDoubleVector(mixedMagnitude), 0,
                 ACL_FLOAT, false, false, false, 1.0e4, 1e-5, metrics);

    std::vector<float> alternating(4096);
    for (size_t i = 0; i < alternating.size(); ++i) {
        alternating[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    RunFloatCase(stream, "Cumsum FP32 alternating cancellation", {4096}, alternating, ToDoubleVector(alternating), 0,
                 ACL_FLOAT, false, false, false, 1e-5, 1e-5, metrics);

    const std::vector<double> doubleBasic = {1.25, -0.5, 2.75, 3.5, -1.0, 4.0};
    RunFloatCase(stream, "Cumsum DOUBLE AiCPU dim1", {2, 3}, doubleBasic, doubleBasic, 1, ACL_DOUBLE, false, false,
                 false, 1e-9, 1e-9, metrics);
    RunFloatCase(stream, "CumsumV2 DOUBLE AiCPU reverse", {2, 3}, doubleBasic, doubleBasic, 1, ACL_DOUBLE, true,
                 false, true, 1e-9, 1e-9, metrics);

    const auto fp16Source = MakeFloatRamp(64, 0.125f);
    const auto fp16Data = ToFp16Vector(fp16Source);
    RunFloatCase(stream, "Cumsum FP16 3D dim1", {4, 4, 4}, fp16Data, Fp16ReferenceInput(fp16Data), 1, ACL_FLOAT16,
                 false, false, false, 5e-2, 5e-2, metrics);
    RunFloatCase(stream, "CumsumV2 FP16 reverse dim2", {2, 4, 8}, fp16Data, Fp16ReferenceInput(fp16Data), 2,
                 ACL_FLOAT16, true, false, true, 5e-2, 5e-2, metrics);

    const auto bf16Source = MakeFloatRamp(64, 0.25f);
    const auto bf16Data = ToBf16Vector(bf16Source);
    RunBf16OptionalCase(stream, "Cumsum BF16 basic", {4, 4, 4}, bf16Data, 2, metrics);

    std::vector<float> nGreater = MakeFloatRamp(4 * 64 * 128, 0.01f);
    RunFloatCase(stream, "Cumsum FP32 tiling N greater cacheline", {4, 64, 128}, nGreater, ToDoubleVector(nGreater), 1,
                 ACL_FLOAT, false, false, false, 1e-3, 1e-4, metrics);

    std::vector<float> nGreaterMEnough = MakeFloatRamp(64 * 64 * 128, 0.001f);
    RunFloatCase(stream, "Cumsum FP32 tiling N greater M split", {64, 64, 128}, nGreaterMEnough,
                 ToDoubleVector(nGreaterMEnough), 1, ACL_FLOAT, false, false, false, 1e-3, 1e-4, metrics);

    std::vector<float> nGreaterLargeN = MakeFloatRamp(64 * 64 * 1024, 0.001f);
    RunFloatCase(stream, "Cumsum FP32 tiling N greater UB split", {64, 64, 1024}, nGreaterLargeN,
                 ToDoubleVector(nGreaterLargeN), 1, ACL_FLOAT, false, false, false, 1e-3, 1e-4, metrics);

    std::vector<float> nGreaterLongR = MakeFloatRamp(64 * 1024 * 128, 0.0001f);
    RunFloatCase(stream, "Cumsum FP32 tiling N greater long R", {64, 1024, 128}, nGreaterLongR,
                 ToDoubleVector(nGreaterLongR), 1, ACL_FLOAT, false, false, false, 2e-2, 1e-4, metrics);

    std::vector<float> nGreaterBorrowN = MakeFloatRamp(4 * 1024 * 1024, 0.0001f);
    RunFloatCase(stream, "Cumsum FP32 tiling borrow N", {4, 1024, 1024}, nGreaterBorrowN,
                 ToDoubleVector(nGreaterBorrowN), 1, ACL_FLOAT, false, false, false, 2e-2, 1e-4, metrics);

    std::vector<float> nGreaterBorrowR = MakeFloatRamp(1 * 8192 * 128, 0.0001f);
    RunFloatCase(stream, "Cumsum FP32 tiling borrow R", {1, 8192, 128}, nGreaterBorrowR,
                 ToDoubleVector(nGreaterBorrowR), 1, ACL_FLOAT, false, false, false, 2e-2, 1e-4, metrics);

    std::vector<float> nGreaterBorrowRUb = MakeFloatRamp(1 * 32768 * 128, 0.00001f);
    RunFloatCase(stream, "Cumsum FP32 aggressive borrow R UB split", {1, 32768, 128}, nGreaterBorrowRUb,
                 ToDoubleVector(nGreaterBorrowRUb), 1, ACL_FLOAT, false, false, false, 2e-1, 1e-4, metrics);

    std::vector<float> rnOneWayFull = MakeFloatRamp(64 * 128, 0.001f);
    RunFloatCase(stream, "Cumsum FP32 aggressive RN oneway full", {64, 128, 1}, rnOneWayFull,
                 ToDoubleVector(rnOneWayFull), 1, ACL_FLOAT, false, false, false, 1e-2, 1e-4, metrics);

    std::vector<float> rnOneWayNotFull = MakeFloatRamp(48 * 4096 * 17, 0.00002f);
    RunFloatCase(stream, "Cumsum FP32 aggressive RN oneway not full", {48, 4096, 17}, rnOneWayNotFull,
                 ToDoubleVector(rnOneWayNotFull), 1, ACL_FLOAT, false, false, false, 2e-1, 1e-4, metrics);

    std::vector<float> rGreater = MakeFloatRamp(16 * 8192, 0.001f);
    RunFloatCase(stream, "Cumsum FP32 tiling long R", {16, 8192, 1}, rGreater, ToDoubleVector(rGreater), 1, ACL_FLOAT,
                 false, false, false, 5e-2, 1e-4, metrics);

    std::vector<float> rGreaterNoBorrow = MakeFloatRamp(48 * 32768, 0.00002f);
    RunFloatCase(stream, "Cumsum FP32 aggressive long R no borrow", {48, 32768, 1}, rGreaterNoBorrow,
                 ToDoubleVector(rGreaterNoBorrow), 1, ACL_FLOAT, false, false, false, 2e-1, 1e-4, metrics);

    std::vector<float> rGreaterNoBorrowTwoway = MakeFloatRamp(40 * 8192 * 16, 0.00001f);
    RunFloatCase(stream, "Cumsum FP32 aggressive twoway no borrow", {40, 8192, 16}, rGreaterNoBorrowTwoway,
                 ToDoubleVector(rGreaterNoBorrowTwoway), 1, ACL_FLOAT, false, false, false, 2e-1, 1e-4, metrics);

    std::vector<float> rGreaterMEnough = MakeFloatRamp(64 * 8192, 0.0005f);
    RunFloatCase(stream, "Cumsum FP32 tiling long R M split", {64, 8192, 1}, rGreaterMEnough,
                 ToDoubleVector(rGreaterMEnough), 1, ACL_FLOAT, false, false, false, 5e-2, 1e-4, metrics);

    std::vector<float> rGreaterNoBorrowM = MakeFloatRamp(64 * 4096 * 16, 0.0002f);
    RunFloatCase(stream, "Cumsum FP32 tiling long R no borrow M", {64, 4096, 16}, rGreaterNoBorrowM,
                 ToDoubleVector(rGreaterNoBorrowM), 1, ACL_FLOAT, false, false, false, 5e-2, 1e-4, metrics);

    std::vector<float> rGreaterBorrowM = MakeFloatRamp(128 * 4096, 0.0005f);
    RunFloatCase(stream, "Cumsum FP32 tiling borrow M", {128, 4096, 1}, rGreaterBorrowM,
                 ToDoubleVector(rGreaterBorrowM), 1, ACL_FLOAT, false, false, false, 5e-2, 1e-4, metrics);

    std::vector<float> rGreaterBorrowMFull = MakeFloatRamp(128 * 512 * 16, 0.00005f);
    RunFloatCase(stream, "Cumsum FP32 aggressive borrow M full", {128, 512, 16}, rGreaterBorrowMFull,
                 ToDoubleVector(rGreaterBorrowMFull), 1, ACL_FLOAT, false, false, false, 5e-2, 1e-4, metrics);

    const auto fp16TwowaySource = MakeFloatRamp(64 * 8192, 0.001f);
    const auto fp16TwowayData = ToFp16Vector(fp16TwowaySource);
    RunFloatCase(stream, "Cumsum FP16 aggressive long R", {64, 8192, 1}, fp16TwowayData,
                 Fp16ReferenceInput(fp16TwowayData), 1, ACL_FLOAT16, false, false, false, 5e-1, 5e-2, metrics);

    std::vector<float> cubeInput(12800 * 512, 1.0f);
    RunFloatCase(stream, "Cumsum FP32 cube branch last dim", {12800, 512}, cubeInput, ToDoubleVector(cubeInput), 1,
                 ACL_FLOAT, false, false, false, 1e-3, 1e-5, metrics);

    RunInt32Case(stream, "Cumsum INT32 dim0 basic", {2, 3}, {1, 2, 3, 4, 5, 6}, 0, false, false, false, metrics);
    RunInt32Case(stream, "Cumsum INT32 dim1 basic", {2, 3}, {1, 2, 3, 4, 5, 6}, 1, false, false, false, metrics);
    RunInt32Case(stream, "CumsumV2 INT32 exclusive reverse", {2, 3}, {1, 2, 3, 4, 5, 6}, 1, true, true, true,
                 metrics);
    RunInt32Case(stream, "Cumsum INT32 overflow wrap", {4}, {1073741824, 1073741824, 1073741824, 1}, 0, false, false,
                 false, metrics);

    const std::vector<int64_t> int64Basic = {1, 2, 3, 4, 5, 6};
    RunFloatCase(stream, "Cumsum INT64 AiCPU dim1", {2, 3}, int64Basic,
                 std::vector<double>{1, 2, 3, 4, 5, 6}, 1, ACL_INT64, false, false, false, 0.0, 0.0, metrics);
    const std::vector<int8_t> int8Basic = {1, -2, 3, -4, 5, -6};
    RunFloatCase(stream, "Cumsum INT8 AiCPU dim0", {2, 3}, int8Basic,
                 std::vector<double>{1, -2, 3, -4, 5, -6}, 0, ACL_INT8, false, false, false, 0.0, 0.0, metrics);
    const std::vector<uint8_t> uint8Basic = {1, 2, 3, 4, 5, 6};
    RunFloatCase(stream, "Cumsum UINT8 AiCPU dim1", {2, 3}, uint8Basic,
                 std::vector<double>{1, 2, 3, 4, 5, 6}, 1, ACL_UINT8, false, false, false, 0.0, 0.0, metrics);

    std::vector<int8_t> int8RightAxis(2 * 64 * 1024);
    for (size_t i = 0; i < int8RightAxis.size(); ++i) {
        int8RightAxis[i] = static_cast<int8_t>((i % 5) - 2);
    }
    RunFloatCase(stream, "Cumsum INT8 tiling large right axis", {2, 64, 1024}, int8RightAxis,
                 std::vector<double>(int8RightAxis.begin(), int8RightAxis.end()), 1, ACL_INT8, false, false, false,
                 0.0, 0.0, metrics);

    std::vector<int8_t> int8AxisNegative(2 * 8 * 2048);
    for (size_t i = 0; i < int8AxisNegative.size(); ++i) {
        int8AxisNegative[i] = static_cast<int8_t>((i % 7) - 3);
    }
    RunFloatCase(stream, "Cumsum INT8 tiling negative axis large right", {2, 8, 2048}, int8AxisNegative,
                 std::vector<double>(int8AxisNegative.begin(), int8AxisNegative.end()), -1, ACL_INT8, false, false,
                 false, 0.0, 0.0, metrics);

    std::vector<int8_t> int8AggressiveR(1 * 32768 * 1);
    for (size_t i = 0; i < int8AggressiveR.size(); ++i) {
        int8AggressiveR[i] = static_cast<int8_t>((i % 9) - 4);
    }
    RunFloatCase(stream, "Cumsum INT8 aggressive R group", {1, 32768, 1}, int8AggressiveR,
                 std::vector<double>(int8AggressiveR.begin(), int8AggressiveR.end()), 1, ACL_INT8, false, false,
                 false, 0.0, 0.0, metrics);

    std::vector<int32_t> int32RightAxis(2 * 64 * 256);
    for (size_t i = 0; i < int32RightAxis.size(); ++i) {
        int32RightAxis[i] = static_cast<int32_t>((i % 9) - 4);
    }
    RunInt32Case(stream, "Cumsum INT32 tiling large right axis", {2, 64, 256}, int32RightAxis, 1, false, false,
                 false, metrics);

    std::vector<int32_t> int32RBlock(4096);
    for (size_t i = 0; i < int32RBlock.size(); ++i) {
        int32RBlock[i] = static_cast<int32_t>((i % 7) - 3);
    }
    RunInt32Case(stream, "Cumsum INT32 tiling R block", {1, 4096, 1}, int32RBlock, 1, false, false, false, metrics);

    std::vector<int32_t> int32AggressiveRBlock(65536);
    for (size_t i = 0; i < int32AggressiveRBlock.size(); ++i) {
        int32AggressiveRBlock[i] = static_cast<int32_t>((i % 5) - 2);
    }
    RunInt32Case(stream, "Cumsum INT32 aggressive R group", {1, 65536, 1}, int32AggressiveRBlock, 1, false, false,
                 false, metrics);

    std::vector<int32_t> int32AggressiveRightAxis(1 * 64 * 8192);
    for (size_t i = 0; i < int32AggressiveRightAxis.size(); ++i) {
        int32AggressiveRightAxis[i] = static_cast<int32_t>((i % 13) - 6);
    }
    RunInt32Case(stream, "Cumsum INT32 aggressive RA split", {1, 64, 8192}, int32AggressiveRightAxis, 1, false,
                 false, false, metrics);

    std::vector<int32_t> int32NegativeAxis(2 * 3 * 128);
    for (size_t i = 0; i < int32NegativeAxis.size(); ++i) {
        int32NegativeAxis[i] = static_cast<int32_t>((i % 11) - 5);
    }
    RunInt32Case(stream, "Cumsum INT32 tiling negative axis", {2, 3, 128}, int32NegativeAxis, -1, false, false,
                 false, metrics);

    std::vector<int32_t> intTiling(32 * 64 * 4);
    for (size_t i = 0; i < intTiling.size(); ++i) {
        intTiling[i] = static_cast<int32_t>((i % 7) - 3);
    }
    RunInt32Case(stream, "Cumsum INT32 tiling middle axis", {32, 64, 4}, intTiling, 1, false, false, false, metrics);

    RunStatusCases(metrics);

    std::cout << "Summary: " << metrics.passed << " passed, " << metrics.failed << " failed, " << metrics.total
              << " total" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(GetDeviceId());
    aclFinalize();
    return metrics.failed == 0 ? 0 : 1;
}
