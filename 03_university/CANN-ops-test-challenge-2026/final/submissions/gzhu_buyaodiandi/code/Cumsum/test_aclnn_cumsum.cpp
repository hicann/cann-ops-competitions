/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <algorithm>
#include <string>

#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

// ============================================================================
// 原有基础设施（保持与示例一致）
// ============================================================================
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
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// ============================================================================
// 全局测试统计
// ============================================================================
static int g_total = 0;
static int g_passed = 0;
static int g_failed = 0;

// ============================================================================
// FP16 编解码辅助
// ============================================================================
static uint16_t FloatToFp16(float value)
{
    uint32_t f;
    memcpy(&f, &value, sizeof(f));
    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127;
    uint32_t frac = f & 0x7FFFFF;

    uint16_t h = (uint16_t)(sign << 15);
    if (exp > 15) {
        h |= 0x7C00;
    } else if (exp < -14) {
        int shift = -14 - exp;
        if (shift < 24) {
            frac = (frac | 0x800000) >> (shift + 13);
        } else {
            frac = 0;
        }
        if (frac > 0x3FF) frac = 0x3FF;
        h |= (uint16_t)frac;
    } else {
        uint16_t hExp = (uint16_t)((exp + 15) & 0x1F);
        uint16_t hFrac = (uint16_t)(frac >> 13);
        h |= (hExp << 10) | hFrac;
    }
    return h;
}

static float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;
        } else {
            int e = -14;
            uint32_t m = frac;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            m &= 0x3FF;
            f = (sign << 31) | (uint32_t)((e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        f = (sign << 31) | (uint32_t)((exp - 15 + 127) << 23) | (frac << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

// ============================================================================
// BF16 编解码辅助
// ============================================================================
static uint16_t FloatToBf16(float value)
{
    uint32_t f;
    memcpy(&f, &value, sizeof(f));
    return (uint16_t)(f >> 16);
}

static float Bf16ToFloat(uint16_t h)
{
    uint32_t f = (uint32_t)h << 16;
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

// ============================================================================
// 多维索引辅助
// ============================================================================
static std::vector<int64_t> LinearToCoord(int64_t idx, const std::vector<int64_t>& shape)
{
    int ndim = (int)shape.size();
    std::vector<int64_t> coord(ndim);
    for (int d = ndim - 1; d >= 0; d--) {
        coord[d] = idx % shape[d];
        idx /= shape[d];
    }
    return coord;
}

static int64_t CoordToLinear(const std::vector<int64_t>& coord, const std::vector<int64_t>& shape)
{
    int ndim = (int)shape.size();
    int64_t idx = 0;
    int64_t stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        idx += coord[d] * stride;
        stride *= shape[d];
    }
    return idx;
}

// ============================================================================
// CPU 参考实现 — 多维 Cumsum (double 精度累加)
// ============================================================================
static void CpuCumsumFloat(const float* input, double* output,
                            const std::vector<int64_t>& shape, int dim,
                            bool exclusive, bool reverse)
{
    int64_t numel = GetShapeSize(shape);
    int ndim = (int)shape.size();
    int actualDim = dim;
    if (actualDim < 0) actualDim += ndim;

    int64_t dimSize = shape[actualDim];

    // 标记哪些位置已经处理过
    std::vector<bool> visited(numel, false);

    for (int64_t i = 0; i < numel; i++) {
        if (visited[i]) continue;
        auto coord = LinearToCoord(i, shape);

        // 只处理沿 dim 轴的起始位置
        bool isStart = false;
        if (!reverse && coord[actualDim] == 0) isStart = true;
        if (reverse && coord[actualDim] == dimSize - 1) isStart = true;
        if (!isStart) continue;

        // 提取沿 dim 的 slice
        std::vector<double> sliceVals(dimSize);
        for (int64_t k = 0; k < dimSize; k++) {
            auto c = coord;
            c[actualDim] = k;
            sliceVals[k] = (double)input[CoordToLinear(c, shape)];
        }

        // cumsum
        std::vector<double> cumVals(dimSize, 0.0);
        if (!reverse) {
            double sum = 0.0;
            for (int64_t k = 0; k < dimSize; k++) {
                if (exclusive) {
                    cumVals[k] = sum;
                    sum += sliceVals[k];
                } else {
                    sum += sliceVals[k];
                    cumVals[k] = sum;
                }
            }
        } else {
            double sum = 0.0;
            for (int64_t k = dimSize - 1; k >= 0; k--) {
                if (exclusive) {
                    cumVals[k] = sum;
                    sum += sliceVals[k];
                } else {
                    sum += sliceVals[k];
                    cumVals[k] = sum;
                }
            }
        }

        // 写回并标记
        for (int64_t k = 0; k < dimSize; k++) {
            auto c = coord;
            c[actualDim] = k;
            int64_t idx = CoordToLinear(c, shape);
            output[idx] = cumVals[k];
            visited[idx] = true;
        }
    }
}

// INT32 CPU参考
static void CpuCumsumInt32(const int32_t* input, int64_t* output,
                             const std::vector<int64_t>& shape, int dim,
                             bool exclusive, bool reverse)
{
    int64_t numel = GetShapeSize(shape);
    int ndim = (int)shape.size();
    int actualDim = dim;
    if (actualDim < 0) actualDim += ndim;
    int64_t dimSize = shape[actualDim];

    std::vector<bool> visited(numel, false);
    for (int64_t i = 0; i < numel; i++) {
        if (visited[i]) continue;
        auto coord = LinearToCoord(i, shape);
        bool isStart = (!reverse && coord[actualDim] == 0) ||
                       (reverse && coord[actualDim] == dimSize - 1);
        if (!isStart) continue;

        std::vector<int64_t> sliceVals(dimSize);
        for (int64_t k = 0; k < dimSize; k++) {
            auto c = coord;
            c[actualDim] = k;
            sliceVals[k] = (int64_t)input[CoordToLinear(c, shape)];
        }

        std::vector<int64_t> cumVals(dimSize, 0);
        if (!reverse) {
            int64_t sum = 0;
            for (int64_t k = 0; k < dimSize; k++) {
                if (exclusive) { cumVals[k] = sum; sum += sliceVals[k]; }
                else { sum += sliceVals[k]; cumVals[k] = sum; }
            }
        } else {
            int64_t sum = 0;
            for (int64_t k = dimSize - 1; k >= 0; k--) {
                if (exclusive) { cumVals[k] = sum; sum += sliceVals[k]; }
                else { sum += sliceVals[k]; cumVals[k] = sum; }
            }
        }

        for (int64_t k = 0; k < dimSize; k++) {
            auto c = coord;
            c[actualDim] = k;
            int64_t idx = CoordToLinear(c, shape);
            output[idx] = cumVals[k];
            visited[idx] = true;
        }
    }
}

// INT64 CPU参考
static void CpuCumsumInt64(const int64_t* input, int64_t* output,
                             const std::vector<int64_t>& shape, int dim,
                             bool exclusive, bool reverse)
{
    int64_t numel = GetShapeSize(shape);
    int ndim = (int)shape.size();
    int actualDim = dim;
    if (actualDim < 0) actualDim += ndim;
    int64_t dimSize = shape[actualDim];

    std::vector<bool> visited(numel, false);
    for (int64_t i = 0; i < numel; i++) {
        if (visited[i]) continue;
        auto coord = LinearToCoord(i, shape);
        bool isStart = (!reverse && coord[actualDim] == 0) ||
                       (reverse && coord[actualDim] == dimSize - 1);
        if (!isStart) continue;

        std::vector<int64_t> sliceVals(dimSize);
        for (int64_t k = 0; k < dimSize; k++) {
            auto c = coord;
            c[actualDim] = k;
            sliceVals[k] = input[CoordToLinear(c, shape)];
        }

        std::vector<int64_t> cumVals(dimSize, 0);
        if (!reverse) {
            int64_t sum = 0;
            for (int64_t k = 0; k < dimSize; k++) {
                if (exclusive) { cumVals[k] = sum; sum += sliceVals[k]; }
                else { sum += sliceVals[k]; cumVals[k] = sum; }
            }
        } else {
            int64_t sum = 0;
            for (int64_t k = dimSize - 1; k >= 0; k--) {
                if (exclusive) { cumVals[k] = sum; sum += sliceVals[k]; }
                else { sum += sliceVals[k]; cumVals[k] = sum; }
            }
        }

        for (int64_t k = 0; k < dimSize; k++) {
            auto c = coord;
            c[actualDim] = k;
            int64_t idx = CoordToLinear(c, shape);
            output[idx] = cumVals[k];
            visited[idx] = true;
        }
    }
}

// ============================================================================
// 封装: 运行 aclnnCumsum
// ============================================================================
static int RunCumsum(aclrtStream stream, aclTensor* self, int64_t dim,
                      aclDataType dtype, aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCumsumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumsum failed. ERROR: %d\n", ret));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

// ============================================================================
// 封装: 运行 aclnnCumsumV2
// ============================================================================
static int RunCumsumV2(aclrtStream stream, aclTensor* self, int64_t dim,
                        aclDataType dtype, bool exclusive, bool reverse,
                        aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    // 【修改点】：去掉了 dtype 参数
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse,
                                              out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCumsumV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumsumV2 failed. ERROR: %d\n", ret));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

// ============================================================================
// 释放辅助
// ============================================================================
static void FreeTensorAndDevice(aclTensor* tensor, void* devAddr)
{
    if (tensor) aclDestroyTensor(tensor);
    if (devAddr) aclrtFree(devAddr);
}

// ============================================================================
// 测试 FP32 Cumsum
// ============================================================================
static bool TestCumsumFloat32(aclrtStream stream, const std::string& name,
                               const std::vector<int64_t>& shape,
                               const std::vector<float>& hostData,
                               int64_t dim, float atol, float rtol,
                               bool useV2 = false, bool exclusive = false,
                               bool reverse = false)
{
    g_total++;
    LOG_PRINT("\nTest %d: %s\n", g_total, name.c_str());

    int64_t numel = GetShapeSize(shape);

    // CPU 参考
    std::vector<double> expected(numel, 0.0);
    CpuCumsumFloat(hostData.data(), expected.data(), shape, (int)dim, exclusive, reverse);

    // 创建输入 tensor
    void* selfDevAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclTensor(hostData, shape, &selfDevAddr, ACL_FLOAT, &self);
    if (ret != 0) {
        LOG_PRINT("  [FAIL] create input tensor failed\n");
        g_failed++;
        return false;
    }

    // 创建输出 tensor
    std::vector<float> outHost(numel, 0.0f);
    void* outDevAddr = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(outHost, shape, &outDevAddr, ACL_FLOAT, &out);
    if (ret != 0) {
        LOG_PRINT("  [FAIL] create output tensor failed\n");
        FreeTensorAndDevice(self, selfDevAddr);
        g_failed++;
        return false;
    }

    // 运行算子
    if (useV2) {
        ret = RunCumsumV2(stream, self, dim, ACL_FLOAT, exclusive, reverse, out);
    } else {
        ret = RunCumsum(stream, self, dim, ACL_FLOAT, out);
    }

    if (ret != 0) {
        LOG_PRINT("  [FAIL] operator execution failed\n");
        FreeTensorAndDevice(self, selfDevAddr);
        FreeTensorAndDevice(out, outDevAddr);
        g_failed++;
        return false;
    }

    // 拷贝结果
    std::vector<float> actual(numel);
    aclrtMemcpy(actual.data(), numel * sizeof(float), outDevAddr,
                numel * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 比对
    bool allPass = true;
    double maxErr = 0.0;
    int64_t maxPos = 0;
    for (int64_t i = 0; i < numel; i++) {
        double diff = std::abs((double)actual[i] - expected[i]);
        double tol = (double)atol + (double)rtol * std::abs(expected[i]);
        if (diff > maxErr) {
            maxErr = diff;
            maxPos = i;
        }
        if (diff > tol) {
            allPass = false;
        }
    }

    // 打印部分结果
    int64_t printCnt = (numel < 8) ? numel : 8;
    LOG_PRINT("  Shape: [");
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) LOG_PRINT(",");
        LOG_PRINT("%ld", (long)shape[i]);
    }
    LOG_PRINT("], dim=%ld", (long)dim);
    if (useV2) LOG_PRINT(", exclusive=%d, reverse=%d", (int)exclusive, (int)reverse);
    LOG_PRINT("\n");

    LOG_PRINT("  Expected: [");
    for (int64_t i = 0; i < printCnt; i++) {
        if (i > 0) LOG_PRINT(", ");
        LOG_PRINT("%.4f", expected[i]);
    }
    if (numel > printCnt) LOG_PRINT(", ...");
    LOG_PRINT("]\n");

    LOG_PRINT("  Actual:   [");
    for (int64_t i = 0; i < printCnt; i++) {
        if (i > 0) LOG_PRINT(", ");
        LOG_PRINT("%.4f", (double)actual[i]);
    }
    if (numel > printCnt) LOG_PRINT(", ...");
    LOG_PRINT("]\n");

    LOG_PRINT("  Max error: %.6e (at position %ld)\n", maxErr, (long)maxPos);

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
        g_passed++;
    } else {
        LOG_PRINT("  [FAIL] tolerance exceeded (atol=%.1e, rtol=%.1e)\n",
                  (double)atol, (double)rtol);
        g_failed++;
    }

    FreeTensorAndDevice(self, selfDevAddr);
    FreeTensorAndDevice(out, outDevAddr);
    return allPass;
}

// ============================================================================
// 测试 FP16 Cumsum
// ============================================================================
static bool TestCumsumFp16(aclrtStream stream, const std::string& name,
                            const std::vector<int64_t>& shape,
                            const std::vector<float>& inputFloats,
                            int64_t dim, float atol, float rtol,
                            bool useV2 = false, bool exclusive = false,
                            bool reverse = false)
{
    g_total++;
    LOG_PRINT("\nTest %d: %s\n", g_total, name.c_str());

    int64_t numel = GetShapeSize(shape);

    // 转为 fp16
    std::vector<uint16_t> fp16Data(numel);
    for (int64_t i = 0; i < numel; i++) {
        fp16Data[i] = FloatToFp16(inputFloats[i]);
    }

    // CPU 参考：先将 fp16 解码为 float，再用 double 累加
    std::vector<float> decodedInput(numel);
    for (int64_t i = 0; i < numel; i++) {
        decodedInput[i] = Fp16ToFloat(fp16Data[i]);
    }
    std::vector<double> expected(numel, 0.0);
    CpuCumsumFloat(decodedInput.data(), expected.data(), shape, (int)dim, exclusive, reverse);

    // 创建输入 tensor (用 uint16_t 作为 fp16)
    void* selfDevAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclTensor(fp16Data, shape, &selfDevAddr, ACL_FLOAT16, &self);
    if (ret != 0) {
        LOG_PRINT("  [FAIL] create input tensor failed\n");
        g_failed++;
        return false;
    }

    std::vector<uint16_t> outHost(numel, 0);
    void* outDevAddr = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(outHost, shape, &outDevAddr, ACL_FLOAT16, &out);
    if (ret != 0) {
        FreeTensorAndDevice(self, selfDevAddr);
        LOG_PRINT("  [FAIL] create output tensor failed\n");
        g_failed++;
        return false;
    }

    if (useV2) {
        ret = RunCumsumV2(stream, self, dim, ACL_FLOAT16, exclusive, reverse, out);
    } else {
        ret = RunCumsum(stream, self, dim, ACL_FLOAT16, out);
    }

    if (ret != 0) {
        LOG_PRINT("  [FAIL] operator execution failed\n");
        FreeTensorAndDevice(self, selfDevAddr);
        FreeTensorAndDevice(out, outDevAddr);
        g_failed++;
        return false;
    }

    std::vector<uint16_t> actualFp16(numel);
    aclrtMemcpy(actualFp16.data(), numel * sizeof(uint16_t), outDevAddr,
                numel * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    bool allPass = true;
    double maxErr = 0.0;
    int64_t maxPos = 0;
    for (int64_t i = 0; i < numel; i++) {
        float actualF = Fp16ToFloat(actualFp16[i]);
        double diff = std::abs((double)actualF - expected[i]);
        double tol = (double)atol + (double)rtol * std::abs(expected[i]);
        if (diff > maxErr) { maxErr = diff; maxPos = i; }
        if (diff > tol) allPass = false;
    }

    int64_t printCnt = (numel < 8) ? numel : 8;
    LOG_PRINT("  Shape: [");
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) LOG_PRINT(",");
        LOG_PRINT("%ld", (long)shape[i]);
    }
    LOG_PRINT("], dim=%ld, dtype=FP16\n", (long)dim);
    LOG_PRINT("  Max error: %.6e (at position %ld)\n", maxErr, (long)maxPos);

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
        g_passed++;
    } else {
        LOG_PRINT("  [FAIL] tolerance exceeded (atol=%.1e, rtol=%.1e)\n",
                  (double)atol, (double)rtol);
        g_failed++;
    }

    FreeTensorAndDevice(self, selfDevAddr);
    FreeTensorAndDevice(out, outDevAddr);
    return allPass;
}

// ============================================================================
// 测试 BF16 Cumsum
// ============================================================================
static bool TestCumsumBf16(aclrtStream stream, const std::string& name,
                            const std::vector<int64_t>& shape,
                            const std::vector<float>& inputFloats,
                            int64_t dim, float atol, float rtol,
                            bool useV2 = false, bool exclusive = false,
                            bool reverse = false)
{
    g_total++;
    LOG_PRINT("\nTest %d: %s\n", g_total, name.c_str());

    int64_t numel = GetShapeSize(shape);

    std::vector<uint16_t> bf16Data(numel);
    for (int64_t i = 0; i < numel; i++) {
        bf16Data[i] = FloatToBf16(inputFloats[i]);
    }

    std::vector<float> decodedInput(numel);
    for (int64_t i = 0; i < numel; i++) {
        decodedInput[i] = Bf16ToFloat(bf16Data[i]);
    }
    std::vector<double> expected(numel, 0.0);
    CpuCumsumFloat(decodedInput.data(), expected.data(), shape, (int)dim, exclusive, reverse);

    void* selfDevAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclTensor(bf16Data, shape, &selfDevAddr, ACL_BF16, &self);
    if (ret != 0) {
        LOG_PRINT("  [FAIL] create input tensor failed\n");
        g_failed++;
        return false;
    }

    std::vector<uint16_t> outHost(numel, 0);
    void* outDevAddr = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(outHost, shape, &outDevAddr, ACL_BF16, &out);
    if (ret != 0) {
        FreeTensorAndDevice(self, selfDevAddr);
        LOG_PRINT("  [FAIL] create output tensor failed\n");
        g_failed++;
        return false;
    }

    if (useV2) {
        ret = RunCumsumV2(stream, self, dim, ACL_BF16, exclusive, reverse, out);
    } else {
        ret = RunCumsum(stream, self, dim, ACL_BF16, out);
    }

    if (ret != 0) {
        LOG_PRINT("  [FAIL] operator execution failed\n");
        FreeTensorAndDevice(self, selfDevAddr);
        FreeTensorAndDevice(out, outDevAddr);
        g_failed++;
        return false;
    }

    std::vector<uint16_t> actualBf16(numel);
    aclrtMemcpy(actualBf16.data(), numel * sizeof(uint16_t), outDevAddr,
                numel * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    bool allPass = true;
    double maxErr = 0.0;
    int64_t maxPos = 0;
    for (int64_t i = 0; i < numel; i++) {
        float actualF = Bf16ToFloat(actualBf16[i]);
        double diff = std::abs((double)actualF - expected[i]);
        double tol = (double)atol + (double)rtol * std::abs(expected[i]);
        if (diff > maxErr) { maxErr = diff; maxPos = i; }
        if (diff > tol) allPass = false;
    }

    LOG_PRINT("  Shape: [");
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) LOG_PRINT(",");
        LOG_PRINT("%ld", (long)shape[i]);
    }
    LOG_PRINT("], dim=%ld, dtype=BF16\n", (long)dim);
    LOG_PRINT("  Max error: %.6e (at position %ld)\n", maxErr, (long)maxPos);

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
        g_passed++;
    } else {
        LOG_PRINT("  [FAIL] tolerance exceeded (atol=%.1e, rtol=%.1e)\n",
                  (double)atol, (double)rtol);
        g_failed++;
    }

    FreeTensorAndDevice(self, selfDevAddr);
    FreeTensorAndDevice(out, outDevAddr);
    return allPass;
}

// ============================================================================
// 测试 INT32 Cumsum
// ============================================================================
static bool TestCumsumInt32(aclrtStream stream, const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<int32_t>& hostData,
                             int64_t dim,
                             bool useV2 = false, bool exclusive = false,
                             bool reverse = false)
{
    g_total++;
    LOG_PRINT("\nTest %d: %s\n", g_total, name.c_str());

    int64_t numel = GetShapeSize(shape);

    std::vector<int64_t> expected(numel, 0);
    CpuCumsumInt32(hostData.data(), expected.data(), shape, (int)dim, exclusive, reverse);

    void* selfDevAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclTensor(hostData, shape, &selfDevAddr, ACL_INT32, &self);
    if (ret != 0) {
        LOG_PRINT("  [FAIL] create input tensor failed\n");
        g_failed++;
        return false;
    }

    std::vector<int32_t> outHost(numel, 0);
    void* outDevAddr = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(outHost, shape, &outDevAddr, ACL_INT32, &out);
    if (ret != 0) {
        FreeTensorAndDevice(self, selfDevAddr);
        LOG_PRINT("  [FAIL] create output tensor failed\n");
        g_failed++;
        return false;
    }

    if (useV2) {
        ret = RunCumsumV2(stream, self, dim, ACL_INT32, exclusive, reverse, out);
    } else {
        ret = RunCumsum(stream, self, dim, ACL_INT32, out);
    }

    if (ret != 0) {
        LOG_PRINT("  [FAIL] operator execution failed\n");
        FreeTensorAndDevice(self, selfDevAddr);
        FreeTensorAndDevice(out, outDevAddr);
        g_failed++;
        return false;
    }

    std::vector<int32_t> actual(numel);
    aclrtMemcpy(actual.data(), numel * sizeof(int32_t), outDevAddr,
                numel * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    bool allPass = true;
    int64_t maxPos = 0;
    for (int64_t i = 0; i < numel; i++) {
        int32_t exp32 = (int32_t)expected[i];
        if (expected[i] > INT32_MAX || expected[i] < INT32_MIN) {
            // 溢出场景：用无符号截断比较
            exp32 = (int32_t)(uint32_t)(uint64_t)expected[i];
        }
        if (actual[i] != exp32) {
            allPass = false;
            maxPos = i;
        }
    }

    int64_t printCnt = (numel < 8) ? numel : 8;
    LOG_PRINT("  Shape: [");
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) LOG_PRINT(",");
        LOG_PRINT("%ld", (long)shape[i]);
    }
    LOG_PRINT("], dim=%ld, dtype=INT32\n", (long)dim);

    LOG_PRINT("  Expected: [");
    for (int64_t i = 0; i < printCnt; i++) {
        if (i > 0) LOG_PRINT(", ");
        LOG_PRINT("%ld", (long)expected[i]);
    }
    if (numel > printCnt) LOG_PRINT(", ...");
    LOG_PRINT("]\n");

    LOG_PRINT("  Actual:   [");
    for (int64_t i = 0; i < printCnt; i++) {
        if (i > 0) LOG_PRINT(", ");
        LOG_PRINT("%d", actual[i]);
    }
    if (numel > printCnt) LOG_PRINT(", ...");
    LOG_PRINT("]\n");

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
        g_passed++;
    } else {
        LOG_PRINT("  [FAIL] mismatch at position %ld\n", (long)maxPos);
        g_failed++;
    }

    FreeTensorAndDevice(self, selfDevAddr);
    FreeTensorAndDevice(out, outDevAddr);
    return allPass;
}

// ============================================================================
// 测试 INT64 Cumsum
// ============================================================================
static bool TestCumsumInt64(aclrtStream stream, const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& hostData,
                             int64_t dim,
                             bool useV2 = false, bool exclusive = false,
                             bool reverse = false)
{
    g_total++;
    LOG_PRINT("\nTest %d: %s\n", g_total, name.c_str());

    int64_t numel = GetShapeSize(shape);

    std::vector<int64_t> expected(numel, 0);
    CpuCumsumInt64(hostData.data(), expected.data(), shape, (int)dim, exclusive, reverse);

    void* selfDevAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclTensor(hostData, shape, &selfDevAddr, ACL_INT64, &self);
    if (ret != 0) {
        LOG_PRINT("  [FAIL] create input tensor failed\n");
        g_failed++;
        return false;
    }

    std::vector<int64_t> outHost(numel, 0);
    void* outDevAddr = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(outHost, shape, &outDevAddr, ACL_INT64, &out);
    if (ret != 0) {
        FreeTensorAndDevice(self, selfDevAddr);
        LOG_PRINT("  [FAIL] create output tensor failed\n");
        g_failed++;
        return false;
    }

    if (useV2) {
        ret = RunCumsumV2(stream, self, dim, ACL_INT64, exclusive, reverse, out);
    } else {
        ret = RunCumsum(stream, self, dim, ACL_INT64, out);
    }

    if (ret != 0) {
        LOG_PRINT("  [FAIL] operator execution failed\n");
        FreeTensorAndDevice(self, selfDevAddr);
        FreeTensorAndDevice(out, outDevAddr);
        g_failed++;
        return false;
    }

    std::vector<int64_t> actual(numel);
    aclrtMemcpy(actual.data(), numel * sizeof(int64_t), outDevAddr,
                numel * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    bool allPass = true;
    int64_t maxPos = 0;
    for (int64_t i = 0; i < numel; i++) {
        if (actual[i] != expected[i]) {
            allPass = false;
            maxPos = i;
        }
    }

    int64_t printCnt = (numel < 8) ? numel : 8;
    LOG_PRINT("  Shape: [");
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) LOG_PRINT(",");
        LOG_PRINT("%ld", (long)shape[i]);
    }
    LOG_PRINT("], dim=%ld, dtype=INT64\n", (long)dim);

    LOG_PRINT("  Expected: [");
    for (int64_t i = 0; i < printCnt; i++) {
        if (i > 0) LOG_PRINT(", ");
        LOG_PRINT("%ld", (long)expected[i]);
    }
    if (numel > printCnt) LOG_PRINT(", ...");
    LOG_PRINT("]\n");

    LOG_PRINT("  Actual:   [");
    for (int64_t i = 0; i < printCnt; i++) {
        if (i > 0) LOG_PRINT(", ");
        LOG_PRINT("%ld", (long)actual[i]);
    }
    if (numel > printCnt) LOG_PRINT(", ...");
    LOG_PRINT("]\n");

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
        g_passed++;
    } else {
        LOG_PRINT("  [FAIL] mismatch at position %ld\n", (long)maxPos);
        g_failed++;
    }

    FreeTensorAndDevice(self, selfDevAddr);
    FreeTensorAndDevice(out, outDevAddr);
    return allPass;
}

// ============================================================================
// 测试 INT8 -> INT32 Cumsum（类型提升）
// ============================================================================
static bool TestCumsumInt8ToInt32(aclrtStream stream, const std::string& name,
                                   const std::vector<int64_t>& shape,
                                   const std::vector<int8_t>& hostData,
                                   int64_t dim,
                                   bool useV2 = false, bool exclusive = false,
                                   bool reverse = false)
{
    g_total++;
    LOG_PRINT("\nTest %d: %s\n", g_total, name.c_str());

    int64_t numel = GetShapeSize(shape);

    // CPU 参考：int8 -> int32 累加
    std::vector<int32_t> int32Input(numel);
    for (int64_t i = 0; i < numel; i++) {
        int32Input[i] = (int32_t)hostData[i];
    }
    std::vector<int64_t> expected(numel, 0);
    CpuCumsumInt32(int32Input.data(), expected.data(), shape, (int)dim, exclusive, reverse);

    // 创建 INT8 输入
    void* selfDevAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclTensor(hostData, shape, &selfDevAddr, ACL_INT8, &self);
    if (ret != 0) {
        LOG_PRINT("  [FAIL] create input tensor failed\n");
        g_failed++;
        return false;
    }

    // 创建 INT32 输出
    std::vector<int32_t> outHost(numel, 0);
    void* outDevAddr = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(outHost, shape, &outDevAddr, ACL_INT32, &out);
    if (ret != 0) {
        FreeTensorAndDevice(self, selfDevAddr);
        LOG_PRINT("  [FAIL] create output tensor failed\n");
        g_failed++;
        return false;
    }

    if (useV2) {
        ret = RunCumsumV2(stream, self, dim, ACL_INT32, exclusive, reverse, out);
    } else {
        ret = RunCumsum(stream, self, dim, ACL_INT32, out);
    }

    if (ret != 0) {
        LOG_PRINT("  [FAIL] operator execution failed\n");
        FreeTensorAndDevice(self, selfDevAddr);
        FreeTensorAndDevice(out, outDevAddr);
        g_failed++;
        return false;
    }

    std::vector<int32_t> actual(numel);
    aclrtMemcpy(actual.data(), numel * sizeof(int32_t), outDevAddr,
                numel * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    bool allPass = true;
    for (int64_t i = 0; i < numel; i++) {
        if (actual[i] != (int32_t)expected[i]) {
            allPass = false;
        }
    }

    LOG_PRINT("  Shape: [");
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) LOG_PRINT(",");
        LOG_PRINT("%ld", (long)shape[i]);
    }
    LOG_PRINT("], dim=%ld, dtype=INT8->INT32\n", (long)dim);

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
        g_passed++;
    } else {
        LOG_PRINT("  [FAIL] mismatch\n");
        g_failed++;
    }

    FreeTensorAndDevice(self, selfDevAddr);
    FreeTensorAndDevice(out, outDevAddr);
    return allPass;
}

// ============================================================================
// main — 全部测试用例
// ============================================================================
int main()
{
    // 1. 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("============================================================\n");
    LOG_PRINT("     Cumsum Operator Comprehensive Test Suite\n");
    LOG_PRINT("============================================================\n");

    // ================================================================
    // Part 1: 数据类型覆盖 — 基础功能
    // ================================================================
    LOG_PRINT("\n\n===== Part 1: Data Type Coverage =====\n");

    // TC01: FP32 1D
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        TestCumsumFloat32(stream, "FP32 1D dim=0 [5]", {5}, data, 0, 1e-5f, 1e-5f);
    }

    // TC02: FP16 1D
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        TestCumsumFp16(stream, "FP16 1D dim=0 [8]", {8}, data, 0, 1e-2f, 1e-2f);
    }

    // TC03: BF16 1D
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        TestCumsumBf16(stream, "BF16 1D dim=0 [4]", {4}, data, 0, 5e-2f, 5e-2f);
    }

    // TC04: INT32 1D
    {
        std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TestCumsumInt32(stream, "INT32 1D dim=0 [10]", {10}, data, 0);
    }

    // TC05: INT64 1D
    {
        std::vector<int64_t> data = {10, 20, 30, 40, 50};
        TestCumsumInt64(stream, "INT64 1D dim=0 [5]", {5}, data, 0);
    }

    // TC06: INT8 -> INT32
    {
        std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
        TestCumsumInt8ToInt32(stream, "INT8->INT32 1D dim=0 [8]", {8}, data, 0);
    }

    // ================================================================
    // Part 2: 多维张量 + 不同 dim 值
    // ================================================================
    LOG_PRINT("\n\n===== Part 2: Multi-dim & Different Dims =====\n");

    // TC07: 2D [3,4] dim=0
    {
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        TestCumsumFloat32(stream, "FP32 2D [3,4] dim=0", {3,4}, data, 0, 1e-5f, 1e-5f);
    }

    // TC08: 2D [3,4] dim=1
    {
        std::vector<float> data = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        TestCumsumFloat32(stream, "FP32 2D [3,4] dim=1", {3,4}, data, 1, 1e-5f, 1e-5f);
    }

    // TC09: 2D [2,3] dim=-1 (负维度)
    {
        std::vector<float> data = {1,2,3, 4,5,6};
        TestCumsumFloat32(stream, "FP32 2D [2,3] dim=-1", {2,3}, data, -1, 1e-5f, 1e-5f);
    }

    // TC10: 3D [2,3,2] dim=0
    {
        std::vector<float> data(12);
        for (int i = 0; i < 12; i++) data[i] = (float)(i + 1);
        TestCumsumFloat32(stream, "FP32 3D [2,3,2] dim=0", {2,3,2}, data, 0, 1e-5f, 1e-5f);
    }

    // TC11: 3D [2,3,2] dim=1
    {
        std::vector<float> data(12);
        for (int i = 0; i < 12; i++) data[i] = (float)(i + 1);
        TestCumsumFloat32(stream, "FP32 3D [2,3,2] dim=1", {2,3,2}, data, 1, 1e-5f, 1e-5f);
    }

    // TC12: 3D [2,3,2] dim=2
    {
        std::vector<float> data(12);
        for (int i = 0; i < 12; i++) data[i] = (float)(i + 1);
        TestCumsumFloat32(stream, "FP32 3D [2,3,2] dim=2", {2,3,2}, data, 2, 1e-5f, 1e-5f);
    }

    // TC13: 4D [2,2,3,2] dim=2
    {
        int64_t n = 2*2*3*2;
        std::vector<float> data(n);
        for (int64_t i = 0; i < n; i++) data[i] = (float)(i % 7 + 1);
        TestCumsumFloat32(stream, "FP32 4D [2,2,3,2] dim=2", {2,2,3,2}, data, 2, 1e-5f, 1e-5f);
    }

    // TC14: 5D [2,2,2,2,2] dim=3
    {
        std::vector<float> data(32);
        for (int i = 0; i < 32; i++) data[i] = (float)(i + 1);
        TestCumsumFloat32(stream, "FP32 5D [2,2,2,2,2] dim=3", {2,2,2,2,2}, data, 3, 1e-5f, 1e-5f);
    }

    // TC15: 5D [2,2,2,2,2] dim=4
    {
        std::vector<float> data(32);
        for (int i = 0; i < 32; i++) data[i] = (float)(i % 5 + 1);
        TestCumsumFloat32(stream, "FP32 5D [2,2,2,2,2] dim=4", {2,2,2,2,2}, data, 4, 1e-5f, 1e-5f);
    }

    // TC16: 2D INT32 dim=0
    {
        std::vector<int32_t> data = {1,2,3, 4,5,6};
        TestCumsumInt32(stream, "INT32 2D [2,3] dim=0", {2,3}, data, 0);
    }

    // TC17: 2D INT32 dim=1
    {
        std::vector<int32_t> data = {1,2,3, 4,5,6};
        TestCumsumInt32(stream, "INT32 2D [2,3] dim=1", {2,3}, data, 1);
    }

    // TC18: 2D FP16 dim=0
    {
        std::vector<float> data = {1,2,3, 4,5,6};
        TestCumsumFp16(stream, "FP16 2D [2,3] dim=0", {2,3}, data, 0, 1e-2f, 1e-2f);
    }

    // TC19: 2D FP16 dim=1
    {
        std::vector<float> data = {1,2,3, 4,5,6};
        TestCumsumFp16(stream, "FP16 2D [2,3] dim=1", {2,3}, data, 1, 1e-2f, 1e-2f);
    }

    // TC20: 3D INT32 dim=1
    {
        std::vector<int32_t> data(24);
        for (int i = 0; i < 24; i++) data[i] = i + 1;
        TestCumsumInt32(stream, "INT32 3D [2,4,3] dim=1", {2,4,3}, data, 1);
    }

    // ================================================================
    // Part 3: 序列长度覆盖
    // ================================================================
    LOG_PRINT("\n\n===== Part 3: Sequence Length =====\n");

    // TC21: 短序列 10
    {
        std::vector<float> data(10, 1.0f);
        TestCumsumFloat32(stream, "FP32 len=10 all 1.0", {10}, data, 0, 1e-5f, 1e-5f);
    }

    // TC22: 中等序列 500
    {
        std::vector<float> data(500, 1.0f);
        TestCumsumFloat32(stream, "FP32 len=500 all 1.0", {500}, data, 0, 1e-4f, 1e-4f);
    }

    // TC23: 长序列 2000
    {
        std::vector<float> data(2000, 1.0f);
        TestCumsumFloat32(stream, "FP32 len=2000 all 1.0", {2000}, data, 0, 1e-3f, 1e-3f);
    }

    // TC24: 超长序列 10000
    {
        int n = 10000;
        std::vector<float> data(n, 1.0f);
        TestCumsumFloat32(stream, "FP32 len=10000 all 1.0 (precision)", {(int64_t)n}, data, 0, 1.0f, 1e-3f);
        LOG_PRINT("  [Precision] cumsum(10000 x 1.0f): error bound ~ n*eps = 10000*5.96e-8 = 5.96e-4\n");
    }

    // TC25: 大 2D [100,100] dim=1
    {
        int64_t n = 100*100;
        std::vector<float> data(n, 1.0f);
        TestCumsumFloat32(stream, "FP32 2D [100,100] dim=1", {100,100}, data, 1, 1e-4f, 1e-4f);
    }

    // TC26: 大 2D [100,100] dim=0
    {
        int64_t n = 100*100;
        std::vector<float> data(n, 0.5f);
        TestCumsumFloat32(stream, "FP32 2D [100,100] dim=0", {100,100}, data, 0, 1e-4f, 1e-4f);
    }

    // TC27: 大 1D FP32 len=4096 (tiling 路径)
    {
        int n = 4096;
        std::vector<float> data(n);
        for (int i = 0; i < n; i++) data[i] = (float)(i % 100) * 0.01f;
        TestCumsumFloat32(stream, "FP32 len=4096 mixed", {(int64_t)n}, data, 0, 1e-2f, 1e-3f);
    }

    // TC28: 大 1D INT32 len=4096
    {
        int n = 4096;
        std::vector<int32_t> data(n);
        for (int i = 0; i < n; i++) data[i] = i % 10;
        TestCumsumInt32(stream, "INT32 len=4096", {(int64_t)n}, data, 0);
    }

    // TC29: 大 1D INT64 len=2048
    {
        int n = 2048;
        std::vector<int64_t> data(n);
        for (int i = 0; i < n; i++) data[i] = (int64_t)(i % 50);
        TestCumsumInt64(stream, "INT64 len=2048", {(int64_t)n}, data, 0);
    }

    // TC30: 大 1D FP16 len=1024
    {
        int n = 1024;
        std::vector<float> data(n, 0.5f);
        TestCumsumFp16(stream, "FP16 len=1024", {(int64_t)n}, data, 0, 5.0f, 0.05f);
    }

    // TC31: 大 2D FP32 [256,16] dim=1
    {
        std::vector<float> data(256*16, 1.0f);
        TestCumsumFloat32(stream, "FP32 2D [256,16] dim=1", {256,16}, data, 1, 1e-4f, 1e-4f);
    }

    // TC32: 大 2D FP32 [16,256] dim=0
    {
        std::vector<float> data(16*256, 0.5f);
        TestCumsumFloat32(stream, "FP32 2D [16,256] dim=0", {16,256}, data, 0, 1e-4f, 1e-4f);
    }

    // ================================================================
    // Part 4: 数值特征覆盖
    // ================================================================
    LOG_PRINT("\n\n===== Part 4: Numerical Characteristics =====\n");

    // TC33: 全正
    {
        std::vector<float> data = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        TestCumsumFloat32(stream, "FP32 all positive", {5}, data, 0, 1e-5f, 1e-5f);
    }

    // TC34: 全负
    {
        std::vector<float> data = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
        TestCumsumFloat32(stream, "FP32 all negative", {5}, data, 0, 1e-5f, 1e-5f);
    }

    // TC35: 正负混合
    {
        std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
        TestCumsumFloat32(stream, "FP32 mixed pos/neg", {8}, data, 0, 1e-5f, 1e-5f);
    }

    // TC36: 含零
    {
        std::vector<float> data = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f};
        TestCumsumFloat32(stream, "FP32 with zeros", {6}, data, 0, 1e-5f, 1e-5f);
    }

    // TC37: 全零
    {
        std::vector<float> data(8, 0.0f);
        TestCumsumFloat32(stream, "FP32 all zeros", {8}, data, 0, 1e-5f, 1e-5f);
    }

    // TC38: 单元素
    {
        std::vector<float> data = {42.0f};
        TestCumsumFloat32(stream, "FP32 single element", {1}, data, 0, 1e-5f, 1e-5f);
    }

    // TC39: shape [1,1,1] dim=0
    {
        std::vector<float> data = {7.5f};
        TestCumsumFloat32(stream, "FP32 [1,1,1] dim=0", {1,1,1}, data, 0, 1e-5f, 1e-5f);
    }

    // TC40: dim轴长度为1
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        TestCumsumFloat32(stream, "FP32 [1,3] dim=0 (dim_size=1)", {1,3}, data, 0, 1e-5f, 1e-5f);
    }

    // TC41: 大值 FP32
    {
        std::vector<float> data = {1e30f, 1e30f, 1e30f, 1e30f};
        TestCumsumFloat32(stream, "FP32 large values 1e30", {4}, data, 0, 1e26f, 1e-3f);
    }

    // TC42: 负大值
    {
        std::vector<float> data = {-1e10f, -1e10f, -1e10f, -1e10f};
        TestCumsumFloat32(stream, "FP32 negative large", {4}, data, 0, 1e6f, 1e-3f);
    }

    // TC43: 递减序列
    {
        std::vector<float> data(20);
        for (int i = 0; i < 20; i++) data[i] = 20.0f - (float)i;
        TestCumsumFloat32(stream, "FP32 descending [20]", {20}, data, 0, 1e-4f, 1e-4f);
    }

    // TC44: 非方形 2D
    {
        std::vector<float> data(21);
        for (size_t i = 0; i < 21; i++) data[i] = (float)(i) * 0.5f + 0.1f;
        TestCumsumFloat32(stream, "FP32 2D [3,7] dim=1", {3,7}, data, 1, 1e-4f, 1e-4f);
    }

    // ================================================================
    // Part 5: CumsumV2 — exclusive / reverse
    // ================================================================
    LOG_PRINT("\n\n===== Part 5: CumsumV2 =====\n");

    // TC45: V2 exclusive=true, 1D FP32
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        TestCumsumFloat32(stream, "V2 FP32 exclusive=true 1D", {5}, data, 0, 1e-5f, 1e-5f,
                           true, true, false);
    }

    // TC46: V2 reverse=true, 1D FP32
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        TestCumsumFloat32(stream, "V2 FP32 reverse=true 1D", {5}, data, 0, 1e-5f, 1e-5f,
                           true, false, true);
    }

    // TC47: V2 exclusive+reverse, 1D FP32
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        TestCumsumFloat32(stream, "V2 FP32 excl+rev 1D", {5}, data, 0, 1e-5f, 1e-5f,
                           true, true, true);
    }

    // TC48: V2 normal (same as Cumsum)
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        TestCumsumFloat32(stream, "V2 FP32 excl=false rev=false 1D", {5}, data, 0, 1e-5f, 1e-5f,
                           true, false, false);
    }

    // TC49: V2 2D exclusive dim=1
    {
        std::vector<float> data = {1,2,3,4, 5,6,7,8};
        TestCumsumFloat32(stream, "V2 FP32 2D [2,4] dim=1 exclusive", {2,4}, data, 1, 1e-5f, 1e-5f,
                           true, true, false);
    }

    // TC50: V2 2D reverse dim=0
    {
        std::vector<float> data = {1,2,3, 4,5,6, 7,8,9};
        TestCumsumFloat32(stream, "V2 FP32 2D [3,3] dim=0 reverse", {3,3}, data, 0, 1e-5f, 1e-5f,
                           true, false, true);
    }

    // TC51: V2 2D exclusive+reverse dim=1
    {
        std::vector<float> data(20);
        for (size_t i = 0; i < 20; i++) data[i] = (float)(i + 1);
        TestCumsumFloat32(stream, "V2 FP32 2D [4,5] dim=1 excl+rev", {4,5}, data, 1, 1e-5f, 1e-5f,
                           true, true, true);
    }

    // TC52: V2 INT32 exclusive
    {
        std::vector<int32_t> data = {1, 2, 3, 4, 5};
        TestCumsumInt32(stream, "V2 INT32 exclusive 1D", {5}, data, 0, true, true, false);
    }

    // TC53: V2 INT32 reverse
    {
        std::vector<int32_t> data = {1, 2, 3, 4, 5};
        TestCumsumInt32(stream, "V2 INT32 reverse 1D", {5}, data, 0, true, false, true);
    }

    // TC54: V2 INT64 excl+rev
    {
        std::vector<int64_t> data = {10, 20, 30, 40, 50};
        TestCumsumInt64(stream, "V2 INT64 excl+rev 1D", {5}, data, 0, true, true, true);
    }

    // TC55: V2 FP16 exclusive
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        TestCumsumFp16(stream, "V2 FP16 exclusive 1D", {4}, data, 0, 1e-2f, 1e-2f,
                        true, true, false);
    }

    // TC56: V2 FP16 reverse
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        TestCumsumFp16(stream, "V2 FP16 reverse 1D", {4}, data, 0, 1e-2f, 1e-2f,
                        true, false, true);
    }

    // TC57: V2 BF16 exclusive
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        TestCumsumBf16(stream, "V2 BF16 exclusive 1D", {4}, data, 0, 5e-2f, 5e-2f,
                        true, true, false);
    }

    // TC58: V2 BF16 reverse
    {
        std::vector<float> data = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
        TestCumsumBf16(stream, "V2 BF16 reverse 1D", {5}, data, 0, 5e-2f, 5e-2f,
                        true, false, true);
    }

    // TC59: V2 BF16 excl+rev
    {
        std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
        TestCumsumBf16(stream, "V2 BF16 excl+rev 1D", {4}, data, 0, 5e-2f, 5e-2f,
                        true, true, true);
    }

    // TC60: V2 FP32 large exclusive
    {
        int n = 1000;
        std::vector<float> data(n, 2.0f);
        TestCumsumFloat32(stream, "V2 FP32 exclusive len=1000", {(int64_t)n}, data, 0,
                           1e-3f, 1e-3f, true, true, false);
    }

    // TC61: V2 FP32 large reverse
    {
        int n = 1000;
        std::vector<float> data(n, 3.0f);
        TestCumsumFloat32(stream, "V2 FP32 reverse len=1000", {(int64_t)n}, data, 0,
                           1e-3f, 1e-3f, true, false, true);
    }

    // TC62: V2 INT32 2D exclusive dim=0
    {
        std::vector<int32_t> data(100);
        for (int i = 0; i < 100; i++) data[i] = i % 7 + 1;
        TestCumsumInt32(stream, "V2 INT32 2D [10,10] dim=0 exclusive", {10,10}, data, 0,
                         true, true, false);
    }

    // ================================================================
    // Part 6: 精度分析专项
    // ================================================================
    LOG_PRINT("\n\n===== Part 6: Precision Analysis =====\n");

    // TC63: 误差累积 — 10000 x 1.0f FP32
    {
        int n = 10000;
        std::vector<float> data(n, 1.0f);
        TestCumsumFloat32(stream, "Precision: 10000x1.0 FP32", {(int64_t)n}, data, 0, 1.0f, 1e-3f);
        LOG_PRINT("  [Analysis] Error accumulation effect:\n");
        LOG_PRINT("    Theoretical last: 10000.0\n");
        LOG_PRINT("    Error bound ~ n * eps/2 = 10000 * 5.96e-8 = 5.96e-4\n");
        LOG_PRINT("    Error grows linearly with sequence length\n");
    }

    // TC64: FP16 累积误差
    {
        int n = 2048;
        std::vector<float> data(n, 1.0f);
        TestCumsumFp16(stream, "Precision: 2048x1.0 FP16", {(int64_t)n}, data, 0, 50.0f, 0.1f);
        LOG_PRINT("  [Analysis] FP16 error accumulation:\n");
        LOG_PRINT("    FP16 eps ~ 9.77e-4\n");
        LOG_PRINT("    Error bound ~ 2048 * 9.77e-4 ~ 2.0\n");
        LOG_PRINT("    FP16 accumulates error ~1000x faster than FP32\n");
    }

    // TC65: 大小数混合 — catastrophic absorption
    {
        int n = 100;
        std::vector<float> data(n);
        for (int i = 0; i < n; i++) data[i] = (i % 2 == 0) ? 1e8f : 1e-6f;
        TestCumsumFloat32(stream, "Precision: large/small alternating", {(int64_t)n}, data, 0,
                           100.0f, 1e-3f);
        LOG_PRINT("  [Analysis] Catastrophic absorption:\n");
        LOG_PRINT("    Pattern: [1e8, 1e-6, 1e8, 1e-6, ...]\n");
        LOG_PRINT("    FP32 has ~7 decimal digits; 1e-6 added to ~1e8 is absorbed\n");
        LOG_PRINT("    Small values contribute 0 to cumulative sum\n");
    }

    // TC66: 小数不精确表示 — 10000 x 0.1f
    {
        int n = 10000;
        std::vector<float> data(n, 0.1f);
        TestCumsumFloat32(stream, "Precision: 10000x0.1 FP32", {(int64_t)n}, data, 0, 2.0f, 1e-3f);
        LOG_PRINT("  [Analysis] Inexact decimal representation:\n");
        LOG_PRINT("    0.1 cannot be exactly represented in binary float\n");
        LOG_PRINT("    0.1f actual = %.20f\n", 0.1f);
        LOG_PRINT("    Theoretical last: 1000.0\n");
        LOG_PRINT("    Each addition accumulates 0.1's representation error\n");
    }

    // TC67: 正负交替抵消
    {
        int n = 1000;
        std::vector<float> data(n);
        for (int i = 0; i < n; i++) data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        TestCumsumFloat32(stream, "Precision: pos/neg alternating cancellation", {(int64_t)n}, data, 0,
                           1e-5f, 1e-5f);
        LOG_PRINT("  [Analysis] Alternating cancellation:\n");
        LOG_PRINT("    Pattern: [1, -1, 1, -1, ...]\n");
        LOG_PRINT("    Cumsum alternates between 1 and 0\n");
        LOG_PRINT("    No significant error accumulation (exact cancellation)\n");
    }

    // TC68: FP32 vs FP16 对比
    {
        int n = 100;
        std::vector<float> data(n, 1.5f);
        LOG_PRINT("\n  --- Precision: FP32 vs FP16 comparison (100 x 1.5) ---\n");
        TestCumsumFloat32(stream, "Precision: FP32 100x1.5", {(int64_t)n}, data, 0, 1e-4f, 1e-4f);
        TestCumsumFp16(stream, "Precision: FP16 100x1.5", {(int64_t)n}, data, 0, 1.0f, 0.05f);
        LOG_PRINT("  [Analysis] dtype error comparison:\n");
        LOG_PRINT("    FP16 error rate ~1000x that of FP32\n");
        LOG_PRINT("    FP16 suited for low-precision scenarios only\n");
    }

    // TC69: INT32 溢出
    {
        std::vector<int32_t> data = {2000000000, 1000000000, 500000000, 100000000};
        TestCumsumInt32(stream, "Precision: INT32 overflow", {4}, data, 0);
        LOG_PRINT("  [Analysis] INT32 overflow:\n");
        LOG_PRINT("    cumsum[1] = 3000000000 > INT32_MAX (2147483647) -> wraps\n");
        LOG_PRINT("    Overflow uses two's complement truncation\n");
    }

    // TC70: V2 exclusive+reverse precision
    {
        int n = 5000;
        std::vector<float> data(n, 1.0f);
        TestCumsumFloat32(stream, "Precision: V2 excl+rev len=5000", {(int64_t)n}, data, 0,
                           1.0f, 1e-3f, true, true, true);
        LOG_PRINT("  [Analysis] V2 exclusive+reverse precision:\n");
        LOG_PRINT("    Same error accumulation as forward, but in reverse direction\n");
    }

    // ================================================================
    // Summary
    // ================================================================
    LOG_PRINT("\n\n============================================================\n");
    LOG_PRINT("                    TEST SUMMARY\n");
    LOG_PRINT("============================================================\n");
    LOG_PRINT("Total:  %d\n", g_total);
    LOG_PRINT("Passed: %d\n", g_passed);
    LOG_PRINT("Failed: %d\n", g_failed);
    LOG_PRINT("============================================================\n");

    // 释放资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (g_failed > 0) ? 1 : 0;
}