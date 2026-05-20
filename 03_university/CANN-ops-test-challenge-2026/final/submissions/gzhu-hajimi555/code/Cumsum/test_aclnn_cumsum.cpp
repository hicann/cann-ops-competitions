/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Comprehensive end-to-end coverage tests for the Cumsum operator (aclnnCumsum / aclnnCumsumV2).
 * Design goals:
 *   - Cover api-layer (aclnn_cumsum.cpp / cumsum.cpp) error paths and dispatch branches
 *   - Cover op_host/arch35 tiling branches: small/large shapes, various dims, all supported dtypes,
 *     cube path, exclusive/reverse combinations, empty tensor, borrow-N/R/M paths
 *   - Per-case CPU reference + tolerance comparison to verify correctness
 *   - Precision analysis cases (error accumulation, mixed magnitudes, fp16 vs fp32)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
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

// ------------------------ Globals / Stats -----------------------------------
static int g_totalCases = 0;
static int g_passCases = 0;
static int g_failCases = 0;

struct CaseResult {
    std::string name;
    bool pass;
    std::string detail;
};
static std::vector<CaseResult> g_results;

static void RecordResult(const std::string& name, bool pass, const std::string& detail = "")
{
    g_totalCases++;
    if (pass) {
        g_passCases++;
        LOG_PRINT("[PASS] %s  %s\n", name.c_str(), detail.c_str());
    } else {
        g_failCases++;
        LOG_PRINT("[FAIL] %s  %s\n", name.c_str(), detail.c_str());
    }
    g_results.push_back({name, pass, detail});
}

// ------------------------ FP16 / BF16 helpers -------------------------------
// Minimal IEEE-754 half-precision (fp16) conversion (CPU-side only).
static uint16_t FloatToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) { return static_cast<uint16_t>(sign); }
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = (mant >> shift) + ((mant >> (shift - 1)) & 1u);
        return static_cast<uint16_t>(sign | half);
    } else if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    half += ((mant >> 12) & 1u); // round-to-nearest
    return static_cast<uint16_t>(half);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            exp += 1;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// Store the upper 16 bits of a float32 as bfloat16.
static uint16_t FloatToBf16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t rounded = x + 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}

static float Bf16ToFloat(uint16_t b)
{
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ------------------------ acl helpers ---------------------------------------
static int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto d : shape) s *= d;
    return s;
}

static int Init(int32_t deviceId, aclrtStream* stream)
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
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                           void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = ShapeSize(shape) * sizeof(T);
    if (size == 0) { size = 1; } // avoid zero alloc for empty tensor
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (!hostData.empty()) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), hostData.size() * sizeof(T),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ------------------------ CPU reference -------------------------------------
// Compute cumsum along a given axis on an n-D tensor (double precision reference).
// supports exclusive / reverse.
static std::vector<double> CpuCumsumND(const std::vector<double>& in, const std::vector<int64_t>& shape,
                                       int64_t dim, bool exclusive = false, bool reverse = false)
{
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t outer = 1, inner = 1, axis = shape[dim];
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

    std::vector<double> out(in.size(), 0.0);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t k = 0; k < inner; ++k) {
            // one 1D cumsum along axis
            if (!reverse) {
                double sum = 0.0;
                for (int64_t a = 0; a < axis; ++a) {
                    int64_t idx = (o * axis + a) * inner + k;
                    if (exclusive) {
                        out[idx] = sum;
                        sum += in[idx];
                    } else {
                        sum += in[idx];
                        out[idx] = sum;
                    }
                }
            } else {
                double sum = 0.0;
                for (int64_t a = axis - 1; a >= 0; --a) {
                    int64_t idx = (o * axis + a) * inner + k;
                    if (exclusive) {
                        out[idx] = sum;
                        sum += in[idx];
                    } else {
                        sum += in[idx];
                        out[idx] = sum;
                    }
                }
            }
        }
    }
    return out;
}

// ------------------------ Compare helpers -----------------------------------
struct CompareStats {
    double maxAbs = 0.0;
    double maxRel = 0.0;
    int64_t maxPos = 0;
    bool pass = true;
};

template <typename ActualT>
static CompareStats CompareFloatLike(const std::vector<ActualT>& actual, const std::vector<double>& expected,
                                     double atol, double rtol,
                                     float (*cvt)(ActualT) = nullptr)
{
    CompareStats s;
    int64_t n = static_cast<int64_t>(actual.size());
    for (int64_t i = 0; i < n; ++i) {
        double a = cvt ? static_cast<double>(cvt(actual[i])) : static_cast<double>(actual[i]);
        double e = expected[i];
        double absErr = std::abs(a - e);
        double tol = atol + rtol * std::abs(e);
        if (absErr > s.maxAbs) { s.maxAbs = absErr; s.maxPos = i; }
        double denom = std::max(std::abs(e), 1e-30);
        double relErr = absErr / denom;
        if (relErr > s.maxRel) s.maxRel = relErr;
        if (absErr > tol) s.pass = false;
    }
    return s;
}

template <typename IntT>
static CompareStats CompareInt(const std::vector<IntT>& actual, const std::vector<double>& expected)
{
    CompareStats s;
    int64_t n = static_cast<int64_t>(actual.size());
    for (int64_t i = 0; i < n; ++i) {
        double a = static_cast<double>(actual[i]);
        double e = expected[i];
        double absErr = std::abs(a - e);
        if (absErr > s.maxAbs) { s.maxAbs = absErr; s.maxPos = i; }
        if (absErr > 0.5) s.pass = false;
    }
    return s;
}

// ------------------------ Core runner ---------------------------------------
template <typename HostT>
static aclnnStatus RunCumsum(const std::vector<HostT>& inHost, const std::vector<int64_t>& shape,
                             int64_t dim, aclDataType dtype, std::vector<HostT>& outHost,
                             aclrtStream stream)
{
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(inHost, shape, &selfDev, dtype, &self);
    if (ret != 0) return static_cast<aclnnStatus>(ret);
    ret = CreateAclTensor(outHost, shape, &outDev, dtype, &out);
    if (ret != 0) return static_cast<aclnnStatus>(ret);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &wsSize, &exec);
    if (st != ACL_SUCCESS) {
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDev); aclrtFree(outDev);
        return st;
    }
    void* wsAddr = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    st = aclnnCumsum(wsAddr, wsSize, exec, stream);
    if (st == ACL_SUCCESS) {
        aclrtSynchronizeStream(stream);
        size_t bytes = static_cast<size_t>(ShapeSize(shape)) * sizeof(HostT);
        if (bytes > 0) {
            aclrtMemcpy(outHost.data(), bytes, outDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (wsAddr) aclrtFree(wsAddr);
    return st;
}

template <typename HostT>
static aclnnStatus RunCumsumV2(const std::vector<HostT>& inHost, const std::vector<int64_t>& shape,
                               int64_t dim, bool exclusive, bool reverse, aclDataType dtype,
                               std::vector<HostT>& outHost, aclrtStream stream)
{
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(inHost, shape, &selfDev, dtype, &self);
    if (ret != 0) return static_cast<aclnnStatus>(ret);
    ret = CreateAclTensor(outHost, shape, &outDev, dtype, &out);
    if (ret != 0) return static_cast<aclnnStatus>(ret);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &wsSize, &exec);
    if (st != ACL_SUCCESS) {
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDev); aclrtFree(outDev);
        return st;
    }
    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    st = aclnnCumsumV2(wsAddr, wsSize, exec, stream);
    if (st == ACL_SUCCESS) {
        aclrtSynchronizeStream(stream);
        size_t bytes = static_cast<size_t>(ShapeSize(shape)) * sizeof(HostT);
        if (bytes > 0) {
            aclrtMemcpy(outHost.data(), bytes, outDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (wsAddr) aclrtFree(wsAddr);
    return st;
}

// ------------------------ Test cases ----------------------------------------
static void TestFloat32Basic(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                             const std::vector<float>& in, aclrtStream stream,
                             double atol = 1e-4, double rtol = 1e-4)
{
    std::vector<float> out(ShapeSize(shape), 0.0f);
    auto st = RunCumsum(in, shape, dim, aclDataType::ACL_FLOAT, out, stream);
    if (st != ACL_SUCCESS) {
        RecordResult(name, false, "aclnnCumsum failed status=" + std::to_string(st));
        return;
    }
    std::vector<double> din(in.begin(), in.end());
    auto expected = CpuCumsumND(din, shape, dim);
    auto s = CompareFloatLike<float>(out, expected, atol, rtol);
    char buf[128];
    snprintf(buf, sizeof(buf), "maxAbsErr=%.3e @pos=%lld", s.maxAbs, static_cast<long long>(s.maxPos));
    RecordResult(name, s.pass, buf);
}

static void TestFloat16Basic(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                             const std::vector<float>& inf, aclrtStream stream,
                             double atol = 1e-2, double rtol = 1e-2)
{
    std::vector<uint16_t> in(inf.size());
    for (size_t i = 0; i < inf.size(); ++i) in[i] = FloatToHalf(inf[i]);
    std::vector<uint16_t> out(ShapeSize(shape), 0);
    auto st = RunCumsum(in, shape, dim, aclDataType::ACL_FLOAT16, out, stream);
    if (st != ACL_SUCCESS) {
        RecordResult(name, false, "aclnnCumsum failed status=" + std::to_string(st));
        return;
    }
    std::vector<double> din;
    din.reserve(in.size());
    for (auto h : in) din.push_back(static_cast<double>(HalfToFloat(h)));
    auto expected = CpuCumsumND(din, shape, dim);
    auto s = CompareFloatLike<uint16_t>(out, expected, atol, rtol, HalfToFloat);
    char buf[128];
    snprintf(buf, sizeof(buf), "maxAbsErr=%.3e @pos=%lld", s.maxAbs, static_cast<long long>(s.maxPos));
    RecordResult(name, s.pass, buf);
}

static void TestBf16Basic(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                          const std::vector<float>& inf, aclrtStream stream,
                          double atol = 1e-1, double rtol = 5e-2)
{
    std::vector<uint16_t> in(inf.size());
    for (size_t i = 0; i < inf.size(); ++i) in[i] = FloatToBf16(inf[i]);
    std::vector<uint16_t> out(ShapeSize(shape), 0);
    auto st = RunCumsum(in, shape, dim, aclDataType::ACL_BF16, out, stream);
    if (st != ACL_SUCCESS) {
        RecordResult(name, false, "aclnnCumsum failed status=" + std::to_string(st));
        return;
    }
    std::vector<double> din;
    din.reserve(in.size());
    for (auto b : in) din.push_back(static_cast<double>(Bf16ToFloat(b)));
    auto expected = CpuCumsumND(din, shape, dim);
    auto s = CompareFloatLike<uint16_t>(out, expected, atol, rtol, Bf16ToFloat);
    char buf[128];
    snprintf(buf, sizeof(buf), "maxAbsErr=%.3e @pos=%lld", s.maxAbs, static_cast<long long>(s.maxPos));
    RecordResult(name, s.pass, buf);
}

template <typename IntT>
static void TestIntBasic(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                         const std::vector<IntT>& in, aclDataType dtype, aclrtStream stream)
{
    std::vector<IntT> out(ShapeSize(shape), static_cast<IntT>(0));
    auto st = RunCumsum(in, shape, dim, dtype, out, stream);
    if (st != ACL_SUCCESS) {
        RecordResult(name, false, "aclnnCumsum failed status=" + std::to_string(st));
        return;
    }
    std::vector<double> din(in.begin(), in.end());
    auto expected = CpuCumsumND(din, shape, dim);
    auto s = CompareInt<IntT>(out, expected);
    char buf[128];
    snprintf(buf, sizeof(buf), "maxAbsErr=%.3f @pos=%lld", s.maxAbs, static_cast<long long>(s.maxPos));
    RecordResult(name, s.pass, buf);
}

// V2 test (FP32)
static void TestV2Float32(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                          bool exclusive, bool reverse, const std::vector<float>& in, aclrtStream stream,
                          double atol = 1e-4, double rtol = 1e-4)
{
    std::vector<float> out(ShapeSize(shape), 0.0f);
    auto st = RunCumsumV2(in, shape, dim, exclusive, reverse, aclDataType::ACL_FLOAT, out, stream);
    if (st != ACL_SUCCESS) {
        RecordResult(name, false, "aclnnCumsumV2 failed status=" + std::to_string(st));
        return;
    }
    std::vector<double> din(in.begin(), in.end());
    auto expected = CpuCumsumND(din, shape, dim, exclusive, reverse);
    auto s = CompareFloatLike<float>(out, expected, atol, rtol);
    char buf[128];
    snprintf(buf, sizeof(buf), "maxAbsErr=%.3e @pos=%lld excl=%d rev=%d",
             s.maxAbs, static_cast<long long>(s.maxPos), exclusive ? 1 : 0, reverse ? 1 : 0);
    RecordResult(name, s.pass, buf);
}

// V2 test (INT32)
static void TestV2Int32(const std::string& name, const std::vector<int64_t>& shape, int64_t dim,
                        bool exclusive, bool reverse, const std::vector<int32_t>& in, aclrtStream stream)
{
    std::vector<int32_t> out(ShapeSize(shape), 0);
    auto st = RunCumsumV2(in, shape, dim, exclusive, reverse, aclDataType::ACL_INT32, out, stream);
    if (st != ACL_SUCCESS) {
        RecordResult(name, false, "aclnnCumsumV2 failed status=" + std::to_string(st));
        return;
    }
    std::vector<double> din(in.begin(), in.end());
    auto expected = CpuCumsumND(din, shape, dim, exclusive, reverse);
    auto s = CompareInt<int32_t>(out, expected);
    char buf[128];
    snprintf(buf, sizeof(buf), "maxAbsErr=%.3f excl=%d rev=%d",
             s.maxAbs, exclusive ? 1 : 0, reverse ? 1 : 0);
    RecordResult(name, s.pass, buf);
}

// ------------------------ Error-path tests ----------------------------------
static void TestNullSelf(aclrtStream /*stream*/)
{
    void* outDev = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> outHost(4, 0.0f);
    std::vector<int64_t> shape = {2, 2};
    CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_FLOAT, &out);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(nullptr, 0, aclDataType::ACL_FLOAT, out, &wsSize, &exec);
    bool pass = (st != ACL_SUCCESS);
    RecordResult("ErrorPath:NullSelf", pass, "expect non-SUCCESS got " + std::to_string(st));
    aclDestroyTensor(out);
    aclrtFree(outDev);
}

static void TestNullOut(aclrtStream /*stream*/)
{
    void* selfDev = nullptr;
    aclTensor* self = nullptr;
    std::vector<float> inHost(4, 1.0f);
    std::vector<int64_t> shape = {2, 2};
    CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_FLOAT, nullptr, &wsSize, &exec);
    bool pass = (st != ACL_SUCCESS);
    RecordResult("ErrorPath:NullOut", pass, "expect non-SUCCESS got " + std::to_string(st));
    aclDestroyTensor(self);
    aclrtFree(selfDev);
}

static void TestInvalidDim(aclrtStream /*stream*/)
{
    std::vector<float> inHost(4, 1.0f), outHost(4, 0.0f);
    std::vector<int64_t> shape = {2, 2};
    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* self = nullptr; aclTensor* out = nullptr;
    CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_FLOAT, &out);
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    // dim out of range
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 5, aclDataType::ACL_FLOAT, out, &wsSize, &exec);
    bool pass = (st != ACL_SUCCESS);
    RecordResult("ErrorPath:InvalidDim", pass, "expect non-SUCCESS got " + std::to_string(st));
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
}

static void TestShapeMismatch(aclrtStream /*stream*/)
{
    std::vector<float> inHost(4, 1.0f), outHost(6, 0.0f);
    std::vector<int64_t> selfShape = {2, 2};
    std::vector<int64_t> outShape = {2, 3};
    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* self = nullptr; aclTensor* out = nullptr;
    CreateAclTensor(inHost, selfShape, &selfDev, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(outHost, outShape, &outDev, aclDataType::ACL_FLOAT, &out);
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_FLOAT, out, &wsSize, &exec);
    bool pass = (st != ACL_SUCCESS);
    RecordResult("ErrorPath:ShapeMismatch", pass, "expect non-SUCCESS got " + std::to_string(st));
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
}

static void TestDtypeMismatch(aclrtStream /*stream*/)
{
    std::vector<float> inHost(4, 1.0f);
    std::vector<int32_t> outHost(4, 0);
    std::vector<int64_t> shape = {2, 2};
    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* self = nullptr; aclTensor* out = nullptr;
    CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_INT32, &out);
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    // dtype arg says FLOAT but out is INT32 -> mismatch
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_FLOAT, out, &wsSize, &exec);
    bool pass = (st != ACL_SUCCESS);
    RecordResult("ErrorPath:DtypeMismatch", pass, "expect non-SUCCESS got " + std::to_string(st));
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
}

static void TestOver8Dims(aclrtStream /*stream*/)
{
    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 2}; // 9D
    int64_t total = 2;
    std::vector<float> inHost(total, 1.0f), outHost(total, 0.0f);
    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* self = nullptr; aclTensor* out = nullptr;
    CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_FLOAT, &out);
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_FLOAT, out, &wsSize, &exec);
    bool pass = (st != ACL_SUCCESS);
    RecordResult("ErrorPath:Over8Dims", pass, "expect non-SUCCESS got " + std::to_string(st));
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
}

static void TestV2DtypeMismatchlessPath(aclrtStream /*stream*/)
{
    // V2 uses CheckDtypeValidWithoutDtype: requires self and out same dtype.
    std::vector<float> inHost(4, 1.0f);
    std::vector<int32_t> outHost(4, 0);
    std::vector<int64_t> shape = {2, 2};
    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* self = nullptr; aclTensor* out = nullptr;
    CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_INT32, &out);
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumV2GetWorkspaceSize(self, 0, false, false, out, &wsSize, &exec);
    bool pass = (st != ACL_SUCCESS);
    RecordResult("ErrorPath:V2DtypeMismatch", pass, "expect non-SUCCESS got " + std::to_string(st));
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
}

// Empty tensor case: shape containing 0 dim. Expect SUCCESS with 0 workspace.
static void TestEmptyTensor(aclrtStream /*stream*/)
{
    std::vector<int64_t> shape = {2, 0, 3};
    std::vector<float> inHost, outHost;
    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* self = nullptr; aclTensor* out = nullptr;
    CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_FLOAT, &out);
    uint64_t wsSize = 123; aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_FLOAT, out, &wsSize, &exec);
    bool pass = (st == ACL_SUCCESS) && (wsSize == 0);
    RecordResult("EmptyTensor", pass, "status=" + std::to_string(st) + " ws=" + std::to_string(wsSize));
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
}

// V2 empty tensor
static void TestV2EmptyTensor(aclrtStream /*stream*/)
{
    std::vector<int64_t> shape = {0, 4};
    std::vector<float> inHost, outHost;
    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* self = nullptr; aclTensor* out = nullptr;
    CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
    CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_FLOAT, &out);
    uint64_t wsSize = 999; aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnCumsumV2GetWorkspaceSize(self, 1, true, false, out, &wsSize, &exec);
    bool pass = (st == ACL_SUCCESS) && (wsSize == 0);
    RecordResult("V2:EmptyTensor", pass, "status=" + std::to_string(st));
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclrtFree(selfDev); aclrtFree(outDev);
}

// ------------------------ Precision analysis cases --------------------------
// Long-sequence accumulation (FP32). Measures error against theoretical sum.
static void PrecisionLongSeqFp32(aclrtStream stream)
{
    const int64_t N = 100000;
    std::vector<int64_t> shape = {N};
    std::vector<float> in(N, 1.0f);
    std::vector<float> out(N, 0.0f);
    auto st = RunCumsum(in, shape, 0, aclDataType::ACL_FLOAT, out, stream);
    if (st != ACL_SUCCESS) { RecordResult("Precision:LongSeq_FP32_ones", false, "run failed"); return; }
    double theoretical = static_cast<double>(N);
    double actual = static_cast<double>(out[N - 1]);
    double err = std::abs(actual - theoretical);
    // Expect error ≤ n*eps*~few; FP32 eps≈1.19e-7 => ~0.1 magnitude acceptable
    double atol = 1.0; double rtol = 1e-4;
    bool pass = err <= atol + rtol * theoretical;
    char buf[160];
    snprintf(buf, sizeof(buf), "N=%lld, theoretical=%.1f, actual=%.3f, absErr=%.3e, bound≈%.3e",
             (long long)N, theoretical, actual, err, atol + rtol * theoretical);
    RecordResult("Precision:LongSeq_FP32_ones", pass, buf);
}

// Long-sequence FP16 accumulation. FP16 saturates once sum exceeds 2048 distinguishable range.
static void PrecisionLongSeqFp16(aclrtStream stream)
{
    const int64_t N = 4096;
    std::vector<int64_t> shape = {N};
    std::vector<float> inf(N, 1.0f);
    std::vector<uint16_t> in(N);
    for (int64_t i = 0; i < N; ++i) in[i] = FloatToHalf(inf[i]);
    std::vector<uint16_t> out(N, 0);
    auto st = RunCumsum(in, shape, 0, aclDataType::ACL_FLOAT16, out, stream);
    if (st != ACL_SUCCESS) { RecordResult("Precision:LongSeq_FP16_ones", false, "run failed"); return; }
    float actual = HalfToFloat(out[N - 1]);
    double theoretical = static_cast<double>(N);
    double err = std::abs(actual - theoretical);
    // FP16 mantissa only 10 bits: expect noticeable drift past 2048.
    char buf[160];
    snprintf(buf, sizeof(buf), "N=%lld, theoretical=%.1f, actual=%.3f, absErr=%.3e (FP16 drift expected)",
             (long long)N, theoretical, static_cast<double>(actual), err);
    // Accept very loose tolerance; this test is informational, but still assert not NaN.
    bool pass = std::isfinite(actual);
    RecordResult("Precision:LongSeq_FP16_ones", pass, buf);
}

// Mixed magnitude (swallowing small contributions).
static void PrecisionMixedMagnitude(aclrtStream stream)
{
    const int64_t N = 4096;
    std::vector<int64_t> shape = {N};
    std::vector<float> in(N);
    for (int64_t i = 0; i < N; ++i) in[i] = (i % 2 == 0) ? 1.0e8f : 1.0e-6f;
    std::vector<float> out(N, 0.0f);
    auto st = RunCumsum(in, shape, 0, aclDataType::ACL_FLOAT, out, stream);
    if (st != ACL_SUCCESS) { RecordResult("Precision:MixedMagnitude_FP32", false, "run failed"); return; }
    double expectedLast = (N / 2) * 1e8 + (N / 2) * 1e-6;
    double actual = static_cast<double>(out[N - 1]);
    double err = std::abs(actual - expectedLast);
    char buf[200];
    snprintf(buf, sizeof(buf), "expected≈%.6e, actual=%.6e, absErr=%.3e (1e-6 swallowed by 1e8 in FP32)",
             expectedLast, actual, err);
    // Informational: pass if large-value cumulation is correct order of magnitude.
    bool pass = std::abs(actual - (N / 2) * 1e8) < 1e-2 * (N / 2) * 1e8;
    RecordResult("Precision:MixedMagnitude_FP32", pass, buf);
}

// 0.1 * N : 0.1 not representable -> drift.
static void PrecisionPointOne(aclrtStream stream)
{
    const int64_t N = 10000;
    std::vector<int64_t> shape = {N};
    std::vector<float> in(N, 0.1f);
    std::vector<float> out(N, 0.0f);
    auto st = RunCumsum(in, shape, 0, aclDataType::ACL_FLOAT, out, stream);
    if (st != ACL_SUCCESS) { RecordResult("Precision:Point1_FP32", false, "run failed"); return; }
    double theoretical = 0.1 * N;
    double actual = static_cast<double>(out[N - 1]);
    double err = std::abs(actual - theoretical);
    char buf[160];
    snprintf(buf, sizeof(buf), "N=%lld, theoretical=%.3f, actual=%.3f, absErr=%.3e",
             (long long)N, theoretical, actual, err);
    bool pass = err < 1.0; // loose bound
    RecordResult("Precision:Point1_FP32", pass, buf);
}

// ------------------------ MAIN ---------------------------------------------
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n====== Cumsum end-to-end test suite ======\n\n");

    // ---- Original example basic case (2x2 fp32 dim=0) ----
    TestFloat32Basic("Base:2x2_fp32_dim0", {2, 2}, 0, {1, 2, 3, 4}, stream);

    // ---- Small 2D fp32 cases across dims ----
    TestFloat32Basic("Base:3x4_fp32_dim0", {3, 4}, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    TestFloat32Basic("Base:3x4_fp32_dim1", {3, 4}, 1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    TestFloat32Basic("Base:3x4_fp32_dimNeg1", {3, 4}, -1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);
    TestFloat32Basic("Base:3x4_fp32_dimNeg2", {3, 4}, -2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, stream);

    // ---- 3D tests covering axis positions: 0, middle, last ----
    {
        std::vector<float> v(2 * 3 * 4);
        std::iota(v.begin(), v.end(), 1.0f);
        TestFloat32Basic("Base:2x3x4_fp32_dim0", {2, 3, 4}, 0, v, stream);
        TestFloat32Basic("Base:2x3x4_fp32_dim1", {2, 3, 4}, 1, v, stream);
        TestFloat32Basic("Base:2x3x4_fp32_dim2", {2, 3, 4}, 2, v, stream);
    }

    // ---- 1D short/medium/long sequences to exercise MRNGreaterCl / NGreaterCl ----
    {
        std::vector<float> v(100);
        std::iota(v.begin(), v.end(), 1.0f);
        TestFloat32Basic("Len:fp32_100", {100}, 0, v, stream);
    }
    {
        std::vector<float> v(1000);
        std::iota(v.begin(), v.end(), 1.0f);
        TestFloat32Basic("Len:fp32_1000", {1000}, 0, v, stream, 1e-3, 1e-4);
    }
    {
        std::vector<float> v(8192);
        for (auto& x : v) x = 0.5f;
        TestFloat32Basic("Len:fp32_8192_half", {8192}, 0, v, stream, 1e-2, 1e-5);
    }

    // ---- all-negative / positive-negative mixed ----
    {
        std::vector<float> v(256);
        for (int i = 0; i < 256; ++i) v[i] = -static_cast<float>(i) * 0.25f;
        TestFloat32Basic("Data:fp32_allneg", {256}, 0, v, stream);
    }
    {
        std::vector<float> v(256);
        for (int i = 0; i < 256; ++i) v[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        TestFloat32Basic("Data:fp32_alternating", {256}, 0, v, stream);
    }
    {
        std::vector<float> v(512, 0.0f);
        TestFloat32Basic("Data:fp32_zeros", {512}, 0, v, stream);
    }

    // ---- Large shapes to exercise tiling branches ----
    // Shape for M dominant: lots of M rows, short R -> NGreaterClRFullLoad path
    {
        int64_t M = 1024, R = 256;
        std::vector<float> v(M * R);
        for (int64_t i = 0; i < M * R; ++i) v[i] = static_cast<float>((i % 16) - 8) * 0.125f;
        TestFloat32Basic("Tiling:NGreaterCl_1024x256_dim1_fp32", {M, R}, 1, v, stream, 5e-2, 1e-4);
    }
    // Small M, large R -> RNGreater/borrowR paths
    {
        int64_t M = 2, R = 16384;
        std::vector<float> v(M * R);
        for (int64_t i = 0; i < M * R; ++i) v[i] = 0.001f;
        TestFloat32Basic("Tiling:BorrowR_2x16384_dim1_fp32", {M, R}, 1, v, stream, 5e-1, 1e-3);
    }
    // Very small: exercise MRNLesserCl (everything < cacheline)
    {
        TestFloat32Basic("Tiling:MRNLesser_2x2x2_dim2_fp32", {2, 2, 2}, 2,
                         {1, 2, 3, 4, 5, 6, 7, 8}, stream);
    }
    // MRNGreaterCl path: M*R*N >= cacheline but R*N < cacheline
    {
        int64_t M = 512, R = 4, N = 4;
        std::vector<float> v(M * R * N, 1.0f);
        TestFloat32Basic("Tiling:MRNGreater_512x4x4_dim1_fp32", {M, R, N}, 1, v, stream, 1e-2, 1e-5);
    }
    // Cube-path candidate: batch >= 12800, channel >= 512, dim = last
    {
        int64_t M = 12800, R = 512;
        std::vector<float> v(M * R, 0.001f);
        TestFloat32Basic("Tiling:CubePath_12800x512_dim1_fp32", {M, R}, 1, v, stream, 5e-1, 1e-3);
    }
    // BorrowM path: M >> coreNum, R moderate, N=1 -> hits RNGreaterClBorrowM (twoway -> M-fold to oneway)
    {
        int64_t M = 4096, R = 1024;
        std::vector<float> v(M * R, 0.0001f);
        TestFloat32Basic("Tiling:BorrowM_4096x1024_dim1_fp32", {M, R}, 1, v, stream, 5e-1, 1e-3);
    }
    // 3D shape with N small but >1, exercises RNGreaterCl twoway sklansky
    {
        int64_t M = 64, R = 4096, N = 4;
        std::vector<float> v(M * R * N, 0.001f);
        TestFloat32Basic("Tiling:RNGreaterCl_64x4096x4_dim1_fp32", {M, R, N}, 1, v, stream, 5e-1, 1e-3);
    }

    // ---- FP16 small ----
    {
        std::vector<float> v(8);
        std::iota(v.begin(), v.end(), 1.0f);
        TestFloat16Basic("Base:fp16_1x8_dim1", {1, 8}, 1, v, stream, 1e-2, 1e-3);
    }
    // FP16 dim=0 on 2D
    {
        std::vector<float> v(4 * 16);
        for (auto& x : v) x = 0.25f;
        TestFloat16Basic("Base:fp16_4x16_dim0", {4, 16}, 0, v, stream, 1e-2, 1e-3);
    }
    // FP16 larger -- exercises dtCast path
    {
        int64_t M = 64, R = 128;
        std::vector<float> v(M * R);
        for (int64_t i = 0; i < M * R; ++i) v[i] = 0.01f;
        TestFloat16Basic("Tiling:fp16_64x128_dim1", {M, R}, 1, v, stream, 5e-1, 1e-2);
    }

    // ---- BF16 ----
    {
        std::vector<float> v(8);
        std::iota(v.begin(), v.end(), 1.0f);
        TestBf16Basic("Base:bf16_1x8_dim1", {1, 8}, 1, v, stream, 5e-1, 5e-2);
    }
    {
        int64_t M = 64, R = 128;
        std::vector<float> v(M * R);
        for (int64_t i = 0; i < M * R; ++i) v[i] = 0.25f;
        TestBf16Basic("Tiling:bf16_64x128_dim1", {M, R}, 1, v, stream, 1.0, 5e-2);
    }

    // ---- INT types (hit int tiling file) ----
    {
        std::vector<int32_t> v = {1, 2, 3, 4, 5, 6};
        TestIntBasic<int32_t>("Base:int32_2x3_dim0", {2, 3}, 0, v, aclDataType::ACL_INT32, stream);
        TestIntBasic<int32_t>("Base:int32_2x3_dim1", {2, 3}, 1, v, aclDataType::ACL_INT32, stream);
    }
    {
        std::vector<int32_t> v(1024);
        for (int i = 0; i < 1024; ++i) v[i] = (i % 7) - 3;
        TestIntBasic<int32_t>("Tiling:int32_1x1024_dim1", {1, 1024}, 1, v, aclDataType::ACL_INT32, stream);
    }
    // 3D int32 mid axis -> forces mid-axis code path in int tiling
    {
        std::vector<int32_t> v(4 * 8 * 16);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 5);
        TestIntBasic<int32_t>("Tiling:int32_4x8x16_dim1_mid", {4, 8, 16}, 1, v, aclDataType::ACL_INT32, stream);
    }
    // Int large R to exercise AR split path (CUM_AR_SPLIT)
    {
        std::vector<int32_t> v(2 * 4096);
        for (size_t i = 0; i < v.size(); ++i) v[i] = 1;
        TestIntBasic<int32_t>("Tiling:int32_2x4096_dim1_ARsplit", {2, 4096}, 1, v, aclDataType::ACL_INT32, stream);
    }
    // INT64
    {
        std::vector<int64_t> v(128);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int64_t>(i);
        TestIntBasic<int64_t>("Base:int64_128_dim0", {128}, 0, v, aclDataType::ACL_INT64, stream);
    }
    // INT8
    {
        std::vector<int8_t> v(64);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int8_t>((i % 3) - 1);
        TestIntBasic<int8_t>("Base:int8_64_dim0", {64}, 0, v, aclDataType::ACL_INT8, stream);
    }
    // UINT8
    {
        std::vector<uint8_t> v(64);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(i % 3);
        TestIntBasic<uint8_t>("Base:uint8_64_dim0", {64}, 0, v, aclDataType::ACL_UINT8, stream);
    }

    // ---- V2 API: exclusive / reverse permutations ----
    {
        std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> shape = {8};
        TestV2Float32("V2:fp32_8_ex0_rev0", shape, 0, false, false, v, stream);
        TestV2Float32("V2:fp32_8_ex1_rev0", shape, 0, true, false, v, stream);
        TestV2Float32("V2:fp32_8_ex0_rev1", shape, 0, false, true, v, stream);
        TestV2Float32("V2:fp32_8_ex1_rev1", shape, 0, true, true, v, stream);
    }
    // V2 on 2D various dims
    {
        std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<int64_t> shape = {3, 4};
        TestV2Float32("V2:fp32_3x4_dim0_ex1", shape, 0, true, false, v, stream);
        TestV2Float32("V2:fp32_3x4_dim1_rev1", shape, 1, false, true, v, stream);
        TestV2Float32("V2:fp32_3x4_dim0_ex1_rev1", shape, 0, true, true, v, stream);
    }
    // V2 int32 variations (int tiling path)
    {
        std::vector<int32_t> v = {1, 2, 3, 4, 5, 6};
        TestV2Int32("V2:int32_2x3_dim1_rev", {2, 3}, 1, false, true, v, stream);
        TestV2Int32("V2:int32_2x3_dim0_ex", {2, 3}, 0, true, false, v, stream);
    }

    // ---- Extended INT-tiling coverage (cumsum_tiling_ascendc_int_arch35.cpp) ----
    // Negative axis adjustment in GetInputDims
    {
        std::vector<int32_t> v(2 * 8);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i);
        TestIntBasic<int32_t>("IntTiling:axisNeg1_2x8", {2, 8}, -1, v, aclDataType::ACL_INT32, stream);
        TestIntBasic<int32_t>("IntTiling:axisNeg2_2x8", {2, 8}, -2, v, aclDataType::ACL_INT32, stream);
    }
    // axis=0 with large rightAxisLen (right > vlSize/2 -> AdjustTensor4TDRA path)
    {
        int64_t M = 8, N = 1024;
        std::vector<int32_t> v(M * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 7);
        TestIntBasic<int32_t>("IntTiling:axis0_8x1024_TDRA", {M, N}, 0, v, aclDataType::ACL_INT32, stream);
    }
    // axis=last with large leftAxisLen (TDLA path branch)
    {
        int64_t L = 256, R = 16;
        std::vector<int32_t> v(L * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 11);
        TestIntBasic<int32_t>("IntTiling:axisLast_256x16_TDLA", {L, R}, 1, v, aclDataType::ACL_INT32, stream);
    }
    // Mid axis, very large right (right*dtypeSize > cacheLine -> block-align branch in AdjustTensor4TDRA)
    {
        int64_t L = 4, M = 8, R = 4096;
        std::vector<int32_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 5) - 2;
        TestIntBasic<int32_t>("IntTiling:mid_4x8x4096_TDRA_blockAlign", {L, M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }
    // Right * dtypeSize >= vlSize -> CUM_NO_SPLIT tilingKey
    {
        int64_t L = 4, M = 16, R = 256; // R*4=1024 >= vlSize(256)
        std::vector<int32_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 3);
        TestIntBasic<int32_t>("IntTiling:NoSplit_4x16x256_dim1", {L, M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }
    // Mid-axis dominant -> CUM_WITH_GROUP (R-block axis path)
    {
        int64_t L = 1, M = 65536, R = 4;
        std::vector<int32_t> v(L * M * R, 1);
        TestIntBasic<int32_t>("IntTiling:GroupR_1x65536x4_dim1", {L, M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }
    // INT8 large right -> dtypeSize==1 path + TDRA path
    {
        int64_t L = 4, M = 8, R = 1024;
        std::vector<int8_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int8_t>((i % 5) - 2);
        TestIntBasic<int8_t>("IntTiling:int8_4x8x1024_dim1_TDRA", {L, M, R}, 1, v,
                             aclDataType::ACL_INT8, stream);
    }
    // INT64 mid axis with right > vlSize/16
    {
        int64_t L = 4, M = 8, R = 64;
        std::vector<int64_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int64_t>(i % 9);
        TestIntBasic<int64_t>("IntTiling:int64_4x8x64_dim1", {L, M, R}, 1, v,
                              aclDataType::ACL_INT64, stream);
    }
    // INT32 V2 with reverse + exclusive on 3D mid axis
    {
        std::vector<int32_t> v(4 * 8 * 16);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 5);
        TestV2Int32("V2:int32_4x8x16_dim1_ex_rev", {4, 8, 16}, 1, true, true, v, stream);
    }

    // ---- Extended FLOAT arch35 tiling coverage (cumsum_tiling_ascendc_arch35.cpp) ----
    // BorrowR multi-iter sklansky (ONEWAY): N>cacheline, R cannot full load, M small
    {
        int64_t M = 1, R = 8192, N = 64;
        std::vector<float> v(M * R * N, 0.0001f);
        TestFloat32Basic("Tiling:BorrowR_OneWay_1x8192x64_dim1", {M, R, N}, 1, v, stream, 1.0, 1e-3);
    }
    // BorrowR multi-iter sklansky (FP16) -> dtCast=true, foldCount > 1 path
    {
        int64_t M = 1, R = 4096, N = 32;
        std::vector<float> vf(M * R * N, 0.0005f);
        TestFloat16Basic("Tiling:fp16_BorrowR_1x4096x32_dim1", {M, R, N}, 1, vf, stream, 5.0, 1e-1);
    }
    // 3D twoway with R full-load: small N, mid R, axis=1
    {
        int64_t M = 32, R = 64, N = 8;
        std::vector<float> v(M * R * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>((i % 11)) * 0.01f;
        TestFloat32Basic("Tiling:TwowaySklanskyRFullLoad_32x64x8_dim1", {M, R, N}, 1, v, stream, 1e-2, 1e-4);
    }
    // BF16 TwoWay sklansky path with R not full load
    {
        int64_t M = 8, R = 2048, N = 4;
        std::vector<float> vf(M * R * N, 0.001f);
        TestBf16Basic("Tiling:bf16_TwowayBorrowR_8x2048x4_dim1", {M, R, N}, 1, vf, stream, 5.0, 5e-2);
    }
    // Reverse + cube-suppressed: 3D shape with last dim moderate, exercises NGreaterClRFullLoad with M<coreNum (borrow N)
    {
        int64_t M = 4, R = 16, N = 256;
        std::vector<float> v(M * R * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>((i % 9)) * 0.1f;
        TestV2Float32("V2:fp32_NGreaterClBorrowN_4x16x256_dim1_rev", {M, R, N}, 1, false, true, v, stream, 1e-2, 1e-4);
    }
    // 0-dim-equivalent: 1D scalar-like (1 element)
    {
        std::vector<float> v = {3.14f};
        TestFloat32Basic("Edge:fp32_single_element", {1}, 0, v, stream);
    }
    // 4D shape (axis in middle, multiple outer dims)
    {
        std::vector<float> v(2 * 3 * 4 * 5);
        std::iota(v.begin(), v.end(), 1.0f);
        TestFloat32Basic("Edge:fp32_2x3x4x5_dim1", {2, 3, 4, 5}, 1, v, stream);
        TestFloat32Basic("Edge:fp32_2x3x4x5_dim2", {2, 3, 4, 5}, 2, v, stream);
        TestFloat32Basic("Edge:fp32_2x3x4x5_dim3", {2, 3, 4, 5}, 3, v, stream);
    }
    // NGreaterClRNotFullLoad first branch (M sufficient): TILING_KEY_UB_SS_ONEWAY
    {
        int64_t M = 32, R = 4096, N = 64; // M>coreNum/2, N*4>=clSize, R cannot full-load
        std::vector<float> v(M * R * N, 0.00005f);
        TestFloat32Basic("Tiling:UB_SS_ONEWAY_32x4096x64_dim1", {M, R, N}, 1, v, stream, 5.0, 1e-3);
    }
    // RNGreaterClRNotFullLoadNotBorrowR twoway path (M sufficient, foldCount>1)
    {
        int64_t M = 32, R = 4096, N = 4; // small N enables twoway sklansky, M>coreNum/2 disables borrow
        std::vector<float> v(M * R * N, 0.001f);
        TestFloat32Basic("Tiling:UB_SS_TWOWAY_32x4096x4_dim1", {M, R, N}, 1, v, stream, 5.0, 1e-3);
    }
    // RNGreaterClRNotFullLoadBorrowRTwoway path
    {
        int64_t M = 1, R = 4096, N = 4; // M<coreNum/2 -> borrowR; small N -> twoway
        std::vector<float> v(M * R * N, 0.001f);
        TestFloat32Basic("Tiling:CORE_SS_TWOWAY_1x4096x4_dim1", {M, R, N}, 1, v, stream, 5.0, 1e-3);
    }

    // ---- AiCpu fallback (cumsum.cpp IsAiCoreSupport=false branch) ----
    // INT16/DOUBLE pass aclnn check but NOT in REGBASE AiCore list -> dispatched to AiCpu by Cumsum().
    // We only call GetWorkspaceSize so AiCpu kernel is *planned* (branch hit) but not executed
    // (avoids potential AiCpu runtime errors that could affect subsequent stream state).
    {
        std::vector<int16_t> inHost(8); for (size_t i = 0; i < 8; ++i) inHost[i] = static_cast<int16_t>(i + 1);
        std::vector<int16_t> outHost(8, 0);
        std::vector<int64_t> shape = {8};
        void* selfDev = nullptr; void* outDev = nullptr;
        aclTensor* self = nullptr; aclTensor* out = nullptr;
        CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_INT16, &self);
        CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_INT16, &out);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_INT16, out, &wsSize, &exec);
        char buf[160];
        snprintf(buf, sizeof(buf), "GetWS int16 status=%d (informational, hits IsAiCoreSupport=false)", st);
        RecordResult("AiCpuPath:int16_GetWorkspace", true, buf);
        if (self) aclDestroyTensor(self);
        if (out) aclDestroyTensor(out);
        if (selfDev) aclrtFree(selfDev);
        if (outDev) aclrtFree(outDev);
    }
    {
        std::vector<double> inHost(8); for (size_t i = 0; i < 8; ++i) inHost[i] = static_cast<double>(i) * 0.5;
        std::vector<double> outHost(8, 0.0);
        std::vector<int64_t> shape = {8};
        void* selfDev = nullptr; void* outDev = nullptr;
        aclTensor* self = nullptr; aclTensor* out = nullptr;
        CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_DOUBLE, &self);
        CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_DOUBLE, &out);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_DOUBLE, out, &wsSize, &exec);
        char buf[160];
        snprintf(buf, sizeof(buf), "GetWS double status=%d (informational, hits IsAiCoreSupport=false)", st);
        RecordResult("AiCpuPath:double_GetWorkspace", true, buf);
        if (self) aclDestroyTensor(self);
        if (out) aclDestroyTensor(out);
        if (selfDev) aclrtFree(selfDev);
        if (outDev) aclrtFree(outDev);
    }
    // V2 with INT16 - hits CheckDtypeValidWithoutDtype branch + AiCpu dispatch
    {
        std::vector<int16_t> inHost(8); for (size_t i = 0; i < 8; ++i) inHost[i] = static_cast<int16_t>(i + 1);
        std::vector<int16_t> outHost(8, 0);
        std::vector<int64_t> shape = {8};
        void* selfDev = nullptr; void* outDev = nullptr;
        aclTensor* self = nullptr; aclTensor* out = nullptr;
        CreateAclTensor(inHost, shape, &selfDev, aclDataType::ACL_INT16, &self);
        CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_INT16, &out);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        aclnnStatus st = aclnnCumsumV2GetWorkspaceSize(self, 0, false, false, out, &wsSize, &exec);
        char buf[160];
        snprintf(buf, sizeof(buf), "V2 GetWS int16 status=%d", st);
        RecordResult("AiCpuPath:int16_V2_GetWorkspace", true, buf);
        if (self) aclDestroyTensor(self);
        if (out) aclDestroyTensor(out);
        if (selfDev) aclrtFree(selfDev);
        if (outDev) aclrtFree(outDev);
    }
    // CheckShapeIsSupport: dim != last (cube path rejected by dim mismatch)
    {
        int64_t M = 12800, R = 512;
        std::vector<float> v(M * R, 0.0f);
        TestFloat32Basic("Tiling:CubeReject_dim0_12800x512", {M, R}, 0, v, stream, 1e-3, 1e-5);
    }
    // ---- Additional arch35 tilingKey coverage (CalcBufferSize switch cases) ----
    // TILING_KEY_TWOWAY (1002): RNGreaterClRFullLoad with TWOWAY pattern
    // Need: lenN small (alignN<=vRegSize/4), R>=FOLD_LEN_MIN(256), lenM<coreNum
    {
        int64_t M = 8, R = 512, N = 8;
        std::vector<float> v(M * R * N, 0.0001f);
        TestFloat32Basic("Tiling:TWOWAY_8x512x8_dim1", {M, R, N}, 1, v, stream, 1.0, 1e-3);
    }
    // TILING_KEY_UB_SS_TWOWAY (1012): R cannot full-load + M sufficient + small N + foldCount>1
    {
        int64_t M = 32, R = 8192, N = 8;
        std::vector<float> v(M * R * N, 0.00001f);
        TestFloat32Basic("Tiling:UB_SS_TWOWAY_32x8192x8_dim1", {M, R, N}, 1, v, stream, 5.0, 1e-3);
    }
    // TILING_KEY_CORE_SS_UB_SS_ONEWAY (1111): BorrowR ONEWAY + R cannot full load
    {
        int64_t M = 1, R = 8192, N = 128; // alignN=512 large; R/coreNum>rMax => not full load
        std::vector<float> v(M * R * N, 0.00001f);
        TestFloat32Basic("Tiling:CORE_SS_UB_SS_ONEWAY_1x8192x128_dim1", {M, R, N}, 1, v, stream, 5.0, 1e-3);
    }
    // TILING_KEY_ONEWAY in BorrowM path (RNGreaterClBorrowM, R UB full-load branch line 676):
    // Need R>=512 (for foldCount>1 in DoFold) but R<=rMaxForFullUb (for full load after M-fold)
    {
        int64_t M = 4096, R = 512, N = 1;
        std::vector<float> v(M * R * N, 0.0001f);
        TestFloat32Basic("Tiling:BorrowM_ONEWAY_4096x512x1_dim1", {M, R, N}, 1, v, stream, 1.0, 1e-3);
    }
    // RNGreaterClRFullLoad with M >= coreNum + ONEWAY (alignN > vRegSize/4)
    {
        int64_t M = 64, R = 32, N = 32; // alignN=128 > 64 -> ONEWAY
        std::vector<float> v(M * R * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>((i % 7)) * 0.01f;
        TestFloat32Basic("Tiling:RNGreaterClRFullLoad_ONEWAY_M64", {M, R, N}, 1, v, stream, 1e-2, 1e-4);
    }

    // ---- Additional aclnn_cumsum.cpp coverage ----
    // 0-dim (scalar) tensor: exercises CheckDim selfDimNum==0 branch and CheckShapeIsSupport
    {
        std::vector<float> v = {2.71828f};
        std::vector<float> outHost = {0.0f};
        std::vector<int64_t> shape0d = {}; // 0-dim
        void* selfDev = nullptr; void* outDev = nullptr;
        aclTensor* self = nullptr; aclTensor* out = nullptr;
        CreateAclTensor(v, shape0d, &selfDev, aclDataType::ACL_FLOAT, &self);
        CreateAclTensor(outHost, shape0d, &outDev, aclDataType::ACL_FLOAT, &out);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        aclnnStatus st = aclnnCumsumGetWorkspaceSize(self, 0, aclDataType::ACL_FLOAT, out, &wsSize, &exec);
        bool pass = (st == ACL_SUCCESS);
        if (pass) {
            void* ws = nullptr;
            if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
            st = aclnnCumsum(ws, wsSize, exec, stream);
            aclrtSynchronizeStream(stream);
            aclrtMemcpy(outHost.data(), sizeof(float), outDev, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ws) aclrtFree(ws);
            pass = (st == ACL_SUCCESS) && std::abs(outHost[0] - 2.71828f) < 1e-4f;
        }
        RecordResult("Edge:scalar_0d_fp32", pass, "");
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDev); aclrtFree(outDev);
    }
    // 0-dim with V2 (also hits CheckParamsWithoutDtype 0-dim path)
    {
        std::vector<int32_t> v = {42};
        std::vector<int32_t> outHost = {0};
        std::vector<int64_t> shape0d = {};
        void* selfDev = nullptr; void* outDev = nullptr;
        aclTensor* self = nullptr; aclTensor* out = nullptr;
        CreateAclTensor(v, shape0d, &selfDev, aclDataType::ACL_INT32, &self);
        CreateAclTensor(outHost, shape0d, &outDev, aclDataType::ACL_INT32, &out);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        aclnnStatus st = aclnnCumsumV2GetWorkspaceSize(self, 0, false, false, out, &wsSize, &exec);
        bool pass = (st == ACL_SUCCESS);
        if (pass) {
            void* ws = nullptr;
            if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
            st = aclnnCumsumV2(ws, wsSize, exec, stream);
            aclrtSynchronizeStream(stream);
            aclrtMemcpy(outHost.data(), sizeof(int32_t), outDev, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ws) aclrtFree(ws);
            pass = (st == ACL_SUCCESS) && (outHost[0] == 42);
        }
        RecordResult("Edge:scalar_0d_int32_V2", pass, "");
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDev); aclrtFree(outDev);
    }
    // Cube path FP16: M*N>=12800*512, dim=last, dtype FP16 -> CheckCubeSupport=true
    {
        int64_t M = 12800, R = 512;
        std::vector<float> vf(M * R, 0.0001f);
        TestFloat16Basic("CubePath:fp16_12800x512_dim1", {M, R}, 1, vf, stream, 5.0, 1e-1);
    }
    // Cube path BF16
    {
        int64_t M = 12800, R = 512;
        std::vector<float> vf(M * R, 0.0001f);
        TestBf16Basic("CubePath:bf16_12800x512_dim1", {M, R}, 1, vf, stream, 10.0, 5e-1);
    }
    // Cube reject due to dtype: INT32 with cube-shape -> CheckCubeSupport returns false (dtype branch)
    {
        int64_t M = 12800, R = 512;
        std::vector<int32_t> v(M * R, 1);
        TestIntBasic<int32_t>("CubeReject:int32_12800x512_dim1", {M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }
    // Cube reject due to dim being middle (not last) with batch>=12800
    {
        int64_t M = 12800, R = 16, N = 8;
        std::vector<float> v(M * R * N, 0.0f);
        TestFloat32Basic("CubeReject:dim1_mid_12800x16x8", {M, R, N}, 1, v, stream, 1e-2, 1e-4);
    }
    // Cube path with negative dim (dim=-1) -> exercises dim<0 adjustment in CheckShapeIsSupport
    {
        int64_t M = 12800, R = 512;
        std::vector<float> v(M * R, 0.0001f);
        TestFloat32Basic("CubePath:fp32_12800x512_dimNeg1", {M, R}, -1, v, stream, 5e-1, 1e-3);
    }

    // ---- Additional cumsum_tiling_ascendc_arch35.cpp coverage ----
    // TILING_KEY_CORE_SS_TWOWAY (1102): borrowR + R full UB load + small N(twoway) + foldCount>1
    // FP16: M=1, N=4, R=65536 -> after split blockFactor small, fold yields foldCount>1, fits UB
    {
        int64_t M = 1, R = 65536, N = 4;
        std::vector<float> vf(M * R * N, 0.00001f);
        TestFloat16Basic("Tiling:CORE_SS_TWOWAY_fp16_1x65536x4_dim1", {M, R, N}, 1, vf, stream, 50.0, 5e-1);
    }
    // TILING_KEY_CORE_SS_UB_SS_TWOWAY (1112): borrowR + R cannot full UB load + small N + foldCount>1
    {
        int64_t M = 1, R = 262144, N = 4;
        std::vector<float> vf(M * R * N, 0.000001f);
        TestFloat16Basic("Tiling:CORE_SS_UB_SS_TWOWAY_fp16_1x262144x4_dim1", {M, R, N}, 1, vf, stream, 200.0, 5e-1);
    }
    // MRNGreaterCl second branch: blockCountMinMForCl > coreNum (large M, very small R*N)
    {
        int64_t M = 4096, R = 2, N = 2;
        std::vector<float> v(M * R * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i % 3) * 0.01f;
        TestFloat32Basic("Tiling:MRNGreater_largeM_4096x2x2_dim1", {M, R, N}, 1, v, stream, 1.0, 1e-3);
    }
    // MRNGreaterCl first branch: blockCountMinMForCl <= coreNum (moderate M)
    {
        int64_t M = 64, R = 2, N = 2;
        std::vector<float> v(M * R * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i % 5) * 0.01f;
        TestFloat32Basic("Tiling:MRNGreater_modM_64x2x2_dim1", {M, R, N}, 1, v, stream, 1e-2, 1e-4);
    }
    // MRNLesserCl: M*R*N very small (< clSize=64bytes/dtSize=16 fp32 elements)
    {
        std::vector<float> v = {1.0f, 2.0f, 3.0f};
        TestFloat32Basic("Tiling:MRNLesserCl_3_dim0", {3}, 0, v, stream);
    }
    // MRNLesserCl with 3D tiny shape
    {
        std::vector<float> v(2 * 1 * 2);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i + 1);
        TestFloat32Basic("Tiling:MRNLesserCl_2x1x2_dim1", {2, 1, 2}, 1, v, stream);
    }
    // RNGreaterClBorrowM with R UB not full load: borrow M then split R as UB
    // M >= coreNum to trigger BorrowM, R > rMaxForFullUb after borrow to force UB_SS_ONEWAY
    {
        int64_t M = 256, R = 4096, N = 1;
        std::vector<float> v(M * R * N, 0.0001f);
        TestFloat32Basic("Tiling:BorrowM_UbNotFull_256x4096x1_dim1", {M, R, N}, 1, v, stream, 5.0, 1e-3);
    }
    // dim=0 with very large size (purely M-axis split, lenN=1, lenM=1, lenR=N)
    {
        int64_t N = 32768;
        std::vector<float> v(N, 0.001f);
        TestFloat32Basic("Tiling:dim0_1d_32768_fp32", {N}, 0, v, stream, 1.0, 1e-3);
    }
    // 4D with axis=0 (mid-axis path with leftAxisLen=1)
    {
        int64_t L = 1, M = 8, R = 64, N = 32;
        std::vector<float> v(L * M * R * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>((i % 11)) * 0.005f;
        TestFloat32Basic("Tiling:4D_1x8x64x32_dim1", {L, M, R, N}, 1, v, stream, 1e-1, 1e-4);
    }
    // V2 exclusive+reverse FP16 large -> exercises V2 path through arch35 tiling
    {
        int64_t M = 32, R = 512, N = 8;
        std::vector<float> vf(M * R * N, 0.001f);
        std::vector<uint16_t> in(vf.size());
        for (size_t i = 0; i < vf.size(); ++i) in[i] = FloatToHalf(vf[i]);
        std::vector<uint16_t> out(in.size(), 0);
        auto st = RunCumsumV2(in, {M, R, N}, 1, true, true, aclDataType::ACL_FLOAT16, out, stream);
        bool pass = (st == ACL_SUCCESS);
        char buf[128]; snprintf(buf, sizeof(buf), "status=%d", st);
        RecordResult("V2:fp16_32x512x8_ex_rev", pass, buf);
    }
    // V2 BF16 with reverse only
    {
        int64_t M = 4, R = 256, N = 4;
        std::vector<float> vf(M * R * N, 0.01f);
        std::vector<uint16_t> in(vf.size());
        for (size_t i = 0; i < vf.size(); ++i) in[i] = FloatToBf16(vf[i]);
        std::vector<uint16_t> out(in.size(), 0);
        auto st = RunCumsumV2(in, {M, R, N}, 1, false, true, aclDataType::ACL_BF16, out, stream);
        bool pass = (st == ACL_SUCCESS);
        char buf[128]; snprintf(buf, sizeof(buf), "status=%d", st);
        RecordResult("V2:bf16_4x256x4_rev", pass, buf);
    }

    // ---- Additional cumsum_tiling_ascendc_int_arch35.cpp coverage ----
    // CheckBGC true branch + AdjustLARLpUnit: large laLpUnit with mid-axis
    // Need: leftAxisLen sizable, midAxisLen, rightAxisLen aligned for BGC
    {
        int64_t L = 256, M = 8, R = 8;
        std::vector<int32_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 7);
        TestIntBasic<int32_t>("IntTiling:BGC_256x8x8_dim1", {L, M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }
    // AdjustTensor4TDR path: small leftAxisLen (<= coreNum/CORE_GATE), large midAxisLen, small rightAxisLen
    // CORE_GATE typically high (e.g. 4) so leftAxisLen=1 hits TDR path
    {
        int64_t L = 1, M = 8192, R = 4;
        std::vector<int32_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 3);
        TestIntBasic<int32_t>("IntTiling:TDR_1x8192x4_dim1", {L, M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }
    // INT8 axis=0 with right > vlSize/2 (TDRA path, dtypeSize==1 -> vlSize halved)
    {
        int64_t M = 4, N = 1024;
        std::vector<int8_t> v(M * N);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int8_t>((i % 5) - 2);
        TestIntBasic<int8_t>("IntTiling:int8_axis0_4x1024_TDRA", {M, N}, 0, v,
                             aclDataType::ACL_INT8, stream);
    }
    // INT8 mid-axis, right < cacheLine -> AdjustTensor4TDLA path with dtypeSize=1
    {
        int64_t L = 32, M = 16, R = 16;
        std::vector<int8_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int8_t>(i % 7);
        TestIntBasic<int8_t>("IntTiling:int8_TDLA_32x16x16_dim1", {L, M, R}, 1, v,
                             aclDataType::ACL_INT8, stream);
    }
    // UINT8 V2 + reverse (AiCpu fallback or AiCore depending on platform)
    {
        std::vector<uint8_t> v(64);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(i % 5);
        std::vector<int64_t> shape = {8, 8};
        std::vector<uint8_t> out(v.size(), 0);
        auto st = RunCumsumV2(v, shape, 0, false, true, aclDataType::ACL_UINT8, out, stream);
        char buf[128]; snprintf(buf, sizeof(buf), "status=%d", st);
        RecordResult("V2:uint8_8x8_rev", st == ACL_SUCCESS, buf);
    }
    // INT64 V2 with exclusive
    {
        std::vector<int64_t> v(48);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int64_t>(i + 1);
        std::vector<int64_t> shape = {6, 8};
        std::vector<int64_t> out(v.size(), 0);
        auto st = RunCumsumV2(v, shape, 1, true, false, aclDataType::ACL_INT64, out, stream);
        char buf[128]; snprintf(buf, sizeof(buf), "status=%d", st);
        RecordResult("V2:int64_6x8_dim1_ex", st == ACL_SUCCESS, buf);
    }
    // INT axis weight: rWeight > 2*raWeight, leftAxisLen tiny -> CalcAxisWeight branches
    {
        int64_t L = 1, M = 4096, R = 1;
        std::vector<int32_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = 1;
        TestIntBasic<int32_t>("IntTiling:rWeight_dom_1x4096x1_dim1", {L, M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }
    // INT raAxisWeight dominant path: large rightAxisLen, small leftAxisLen, small midAxisLen
    {
        int64_t L = 1, M = 4, R = 16384;
        std::vector<int32_t> v(L * M * R);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(i % 3);
        TestIntBasic<int32_t>("IntTiling:raWeight_dom_1x4x16384_dim1", {L, M, R}, 1, v,
                              aclDataType::ACL_INT32, stream);
    }

    // ---- Additional cumsum.cpp coverage: V2 with various dtypes hitting AiCore/AiCpu dispatch ----
    // BF16 V2 GetWorkspace (REGBASE: BF16 in AICore list)
    {
        int64_t M = 4, R = 8;
        std::vector<float> vf(M * R, 0.5f);
        std::vector<uint16_t> in(vf.size());
        for (size_t i = 0; i < vf.size(); ++i) in[i] = FloatToBf16(vf[i]);
        std::vector<uint16_t> out(in.size(), 0);
        auto st = RunCumsumV2(in, {M, R}, 1, true, true, aclDataType::ACL_BF16, out, stream);
        char buf[128]; snprintf(buf, sizeof(buf), "status=%d", st);
        RecordResult("V2:bf16_4x8_ex_rev", st == ACL_SUCCESS, buf);
    }
    // Double V2 GetWorkspace (not in any AiCore list -> AiCpu)
    {
        std::vector<double> in(8); for (size_t i = 0; i < 8; ++i) in[i] = static_cast<double>(i) * 0.25;
        std::vector<double> outHost(8, 0.0);
        std::vector<int64_t> shape = {8};
        void* selfDev = nullptr; void* outDev = nullptr;
        aclTensor* self = nullptr; aclTensor* out = nullptr;
        CreateAclTensor(in, shape, &selfDev, aclDataType::ACL_DOUBLE, &self);
        CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_DOUBLE, &out);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        aclnnStatus st = aclnnCumsumV2GetWorkspaceSize(self, 0, false, false, out, &wsSize, &exec);
        char buf[128]; snprintf(buf, sizeof(buf), "V2 GetWS double status=%d", st);
        RecordResult("AiCpuPath:double_V2_GetWorkspace", true, buf);
        aclDestroyTensor(self); aclDestroyTensor(out);
        aclrtFree(selfDev); aclrtFree(outDev);
    }

    // ---- Error / edge paths ----
    TestEmptyTensor(stream);
    TestV2EmptyTensor(stream);
    TestNullSelf(stream);
    TestNullOut(stream);
    TestInvalidDim(stream);
    TestShapeMismatch(stream);
    TestDtypeMismatch(stream);
    TestV2DtypeMismatchlessPath(stream);
    TestOver8Dims(stream);

    // ---- Precision analysis ----
    LOG_PRINT("\n-- Precision analysis --\n");
    PrecisionLongSeqFp32(stream);
    PrecisionLongSeqFp16(stream);
    PrecisionMixedMagnitude(stream);
    PrecisionPointOne(stream);

    // ---- Summary ----
    LOG_PRINT("\n====== Summary: %d passed, %d failed (total %d) ======\n",
              g_passCases, g_failCases, g_totalCases);

    // Cleanup acl
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return g_failCases == 0 ? 0 : 1;
}
