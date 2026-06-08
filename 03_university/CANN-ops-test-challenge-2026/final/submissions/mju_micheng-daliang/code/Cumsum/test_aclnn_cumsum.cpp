/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * test_aclnn_cumsum.cpp —— Cumsum 算子端到端综合测试
 *
 * 设计目标:
 *   1. 100% 覆盖 op_api/aclnn_cumsum.cpp 与 op_api/cumsum.cpp 的关键分支
 *      （API 分发、参数校验、Cube 选择、AiCore/AiCpu 路由）。
 *   2. 通过多种 dtype 与 shape 组合，触达 op_host/arch35 三个 tiling
 *      文件中的浮点/整数分支与 N/M/R 切分策略（NGreaterCl/NLesserCl/
 *      RNGreaterCl/RNLesserCl/oneway/twoway/borrow-M/borrow-R 等）。
 *   3. 提供 CPU 双精度 Oracle，对每个用例进行容差比对，输出 [PASS]/[FAIL]。
 *   4. 包含误差累积、长序列、大小数混合、异常拦截等精度与鲁棒性场景。
 *
 * 运行: 由 build.sh --run_example cumsum eager cust 调用。
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

// aclnn 状态码（ACLNN_SUCCESS / ACLNN_ERR_PARAM_NULLPTR /
// ACLNN_ERR_PARAM_INVALID / ACLNN_ERR_INNER 等）。
// 在 CANN 9.0 中位于 aclnn/opdev/op_errno.h；为兼容找不到该头的环境，
// 这里同时给出 fallback 宏定义。
#if __has_include("aclnn/opdev/op_errno.h")
#include "aclnn/opdev/op_errno.h"
#else
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif
#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif
#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
#endif
#ifndef ACLNN_ERR_RUNTIME_ERROR
#define ACLNN_ERR_RUNTIME_ERROR 361001
#endif
#ifndef ACLNN_ERR_INNER
#define ACLNN_ERR_INNER 561000
#endif
#endif

// ---------------------------------------------------------------------------
// 基础工具
// ---------------------------------------------------------------------------
#define CHECK_RET(cond, return_expr)                                                       \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            return_expr;                                                                   \
        }                                                                                  \
    } while (0)

#define LOG_PRINT(message, ...)                                                            \
    do {                                                                                   \
        printf(message, ##__VA_ARGS__);                                                    \
        fflush(stdout);                                                                    \
    } while (0)

static int g_passed = 0;
static int g_failed = 0;

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto d : shape) {
        s *= (d <= 0 ? 0 : d);
    }
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

// 不要求 hostData 为非空：当 size 为 0（空 tensor）时跳过 memcpy。
template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                           void** deviceAddr, aclDataType dtype, aclTensor** tensor)
{
    int64_t numel = GetShapeSize(shape);
    size_t elemSize = sizeof(T);
    size_t bytes = static_cast<size_t>(numel) * elemSize;

    if (bytes == 0) {
        // 空 tensor：仍然需要 device 内存占位（分配 1 字节避免 nullptr）。
        bytes = 1;
    }
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    if (numel > 0 && !hostData.empty()) {
        ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), numel * elemSize,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
    return 0;
}

// ---------------------------------------------------------------------------
// FP16 / BF16 软件转换
// ---------------------------------------------------------------------------
static uint16_t FloatToFp16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    uint16_t out;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        out = static_cast<uint16_t>((sign << 15) | (0x1fu << 10) | (mant ? 0x200u : 0u));
    } else if (exp >= 31) {
        out = static_cast<uint16_t>((sign << 15) | (0x1fu << 10));
    } else if (exp <= 0) {
        out = static_cast<uint16_t>(sign << 15);
    } else {
        out = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }
    return out;
}

static float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        bits = sign << 31;
    } else if (exp == 31) {
        bits = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t FloatToBf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // round-to-nearest-even
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding = 0x7fffu + lsb;
    bits += rounding;
    return static_cast<uint16_t>(bits >> 16);
}

static float Bf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// CPU Oracle —— 通用 cumsum 实现，支持 dim/exclusive/reverse
// 为最大限度规避自身舍入误差，浮点累加在 double 中进行。
// ---------------------------------------------------------------------------
template <typename TIn, typename TAcc>
static void CpuCumsumGeneric(const std::vector<TIn>& input, const std::vector<int64_t>& shape,
                             int64_t dim, bool exclusive, bool reverse,
                             std::vector<TAcc>& out,
                             const std::function<double(TIn)>& toDouble,
                             const std::function<TAcc(double)>& fromDouble)
{
    int64_t rank = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += rank;
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    int64_t mid = shape[dim];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < rank; ++i) inner *= shape[i];

    out.assign(GetShapeSize(shape), TAcc{});
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double sum = 0.0;
            if (!reverse) {
                for (int64_t m = 0; m < mid; ++m) {
                    int64_t idx = (o * mid + m) * inner + i;
                    double v = toDouble(input[idx]);
                    if (exclusive) {
                        out[idx] = fromDouble(sum);
                        sum += v;
                    } else {
                        sum += v;
                        out[idx] = fromDouble(sum);
                    }
                }
            } else {
                for (int64_t m = mid - 1; m >= 0; --m) {
                    int64_t idx = (o * mid + m) * inner + i;
                    double v = toDouble(input[idx]);
                    if (exclusive) {
                        out[idx] = fromDouble(sum);
                        sum += v;
                    } else {
                        sum += v;
                        out[idx] = fromDouble(sum);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 容差比对（兼容 float / int）
// ---------------------------------------------------------------------------
template <typename T>
static bool VerifyClose(const std::vector<T>& expected, const std::vector<T>& actual,
                        double atol, double rtol, double& maxAbsErr, int64_t& worstIdx)
{
    maxAbsErr = 0.0;
    worstIdx = -1;
    if (expected.size() != actual.size()) return false;
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        double e = static_cast<double>(expected[i]);
        double a = static_cast<double>(actual[i]);
        if (std::isnan(e) || std::isnan(a)) {
            if (std::isnan(e) != std::isnan(a)) { ok = false; worstIdx = static_cast<int64_t>(i); }
            continue;
        }
        if (std::isinf(e) || std::isinf(a)) {
            if (e != a) { ok = false; worstIdx = static_cast<int64_t>(i); }
            continue;
        }
        double err = std::fabs(a - e);
        if (err > maxAbsErr) { maxAbsErr = err; worstIdx = static_cast<int64_t>(i); }
        double bound = atol + rtol * std::fabs(e);
        if (err > bound) ok = false;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 通用算子调用器
//   useV2=false → aclnnCumsum；useV2=true → aclnnCumsumV2(exclusive,reverse)
//   返回 (status, output)。caller 据此判定是否符合预期。
// ---------------------------------------------------------------------------
struct InvokeResult {
    aclnnStatus wsStatus = ACLNN_SUCCESS;
    aclnnStatus runStatus = ACLNN_SUCCESS;
    std::vector<uint8_t> outBytes; // device→host 后的原始字节（按 elemBytes 解读）
};

static InvokeResult InvokeCumsum(aclrtStream stream,
                                 const std::vector<int64_t>& selfShape, aclDataType selfDtype,
                                 const void* selfHost, size_t selfBytes,
                                 const std::vector<int64_t>& outShape, aclDataType outDtype,
                                 size_t outElemBytes,
                                 int64_t dim, bool useV2, bool exclusive, bool reverse,
                                 bool passNullSelf = false, bool passNullOut = false,
                                 aclDataType apiDtype = ACL_DT_UNDEFINED)
{
    if (apiDtype == ACL_DT_UNDEFINED) apiDtype = outDtype;
    InvokeResult r;
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    int64_t selfNumel = GetShapeSize(selfShape);
    int64_t outNumel = GetShapeSize(outShape);

    if (!passNullSelf) {
        size_t bytes = std::max<size_t>(selfBytes, 1u);
        if (aclrtMalloc(&selfDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            r.wsStatus = ACLNN_ERR_INNER; return r;
        }
        if (selfNumel > 0 && selfHost != nullptr) {
            aclrtMemcpy(selfDev, bytes, selfHost, selfBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        std::vector<int64_t> strides(selfShape.size(), 1);
        for (int64_t i = static_cast<int64_t>(selfShape.size()) - 2; i >= 0; --i) {
            strides[i] = selfShape[i + 1] * strides[i + 1];
        }
        self = aclCreateTensor(selfShape.data(), selfShape.size(), selfDtype, strides.data(), 0,
                               aclFormat::ACL_FORMAT_ND, selfShape.data(), selfShape.size(), selfDev);
    }
    if (!passNullOut) {
        size_t outBytes = std::max<size_t>(static_cast<size_t>(outNumel) * outElemBytes, 1u);
        if (aclrtMalloc(&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            if (selfDev) aclrtFree(selfDev);
            if (self) aclDestroyTensor(self);
            r.wsStatus = ACLNN_ERR_INNER; return r;
        }
        std::vector<int64_t> strides(outShape.size(), 1);
        for (int64_t i = static_cast<int64_t>(outShape.size()) - 2; i >= 0; --i) {
            strides[i] = outShape[i + 1] * strides[i + 1];
        }
        out = aclCreateTensor(outShape.data(), outShape.size(), outDtype, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, outShape.data(), outShape.size(), outDev);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus s;
    if (useV2) {
        s = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    } else {
        s = aclnnCumsumGetWorkspaceSize(self, dim, apiDtype, out, &workspaceSize, &executor);
    }
    r.wsStatus = s;

    if (s == ACLNN_SUCCESS) {
        void* ws = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&ws, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        aclnnStatus s2 = useV2 ? aclnnCumsumV2(ws, workspaceSize, executor, stream)
                               : aclnnCumsum(ws, workspaceSize, executor, stream);
        r.runStatus = s2;
        if (s2 == ACLNN_SUCCESS) {
            aclrtSynchronizeStream(stream);
            if (outNumel > 0 && outDev != nullptr) {
                r.outBytes.resize(static_cast<size_t>(outNumel) * outElemBytes);
                aclrtMemcpy(r.outBytes.data(), r.outBytes.size(), outDev,
                            r.outBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
            }
        }
        if (ws) aclrtFree(ws);
    }

    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
    if (selfDev) aclrtFree(selfDev);
    if (outDev) aclrtFree(outDev);
    return r;
}

// ---------------------------------------------------------------------------
// 通用浮点测试模板: dtype 通过 enum 区分
// ---------------------------------------------------------------------------
enum class TestDtype { FP32, FP16, BF16, I32, I64, I8, U8 };

static const char* DtypeName(TestDtype t)
{
    switch (t) {
        case TestDtype::FP32: return "FP32";
        case TestDtype::FP16: return "FP16";
        case TestDtype::BF16: return "BF16";
        case TestDtype::I32:  return "INT32";
        case TestDtype::I64:  return "INT64";
        case TestDtype::I8:   return "INT8";
        case TestDtype::U8:   return "UINT8";
    }
    return "?";
}

static aclDataType ToAcl(TestDtype t)
{
    switch (t) {
        case TestDtype::FP32: return ACL_FLOAT;
        case TestDtype::FP16: return ACL_FLOAT16;
        case TestDtype::BF16: return ACL_BF16;
        case TestDtype::I32:  return ACL_INT32;
        case TestDtype::I64:  return ACL_INT64;
        case TestDtype::I8:   return ACL_INT8;
        case TestDtype::U8:   return ACL_UINT8;
    }
    return ACL_FLOAT;
}

static size_t ElemBytes(TestDtype t)
{
    switch (t) {
        case TestDtype::FP32: case TestDtype::I32: return 4;
        case TestDtype::FP16: case TestDtype::BF16: return 2;
        case TestDtype::I64:  return 8;
        case TestDtype::I8:   case TestDtype::U8: return 1;
    }
    return 4;
}

// 主测试入口：对一组 float 输入按目标 dtype 量化、调用算子、比较结果。
//   atol/rtol 仅用于浮点；整数采用精确匹配。
static void RunFloatCase(aclrtStream stream, const char* name,
                         const std::vector<int64_t>& shape, const std::vector<float>& fInput,
                         int64_t dim, TestDtype dtype, bool useV2, bool exclusive, bool reverse,
                         double atol, double rtol)
{
    aclDataType acl = ToAcl(dtype);
    int64_t numel = GetShapeSize(shape);
    if (numel != static_cast<int64_t>(fInput.size())) {
        LOG_PRINT("[%s] BAD-INPUT (numel mismatch)\n", name);
        ++g_failed; return;
    }
    InvokeResult r;
    if (dtype == TestDtype::FP32) {
        r = InvokeCumsum(stream, shape, acl, fInput.data(), fInput.size() * 4,
                         shape, acl, 4, dim, useV2, exclusive, reverse);
    } else if (dtype == TestDtype::FP16) {
        std::vector<uint16_t> q(numel);
        for (int64_t i = 0; i < numel; ++i) q[i] = FloatToFp16(fInput[i]);
        r = InvokeCumsum(stream, shape, acl, q.data(), q.size() * 2,
                         shape, acl, 2, dim, useV2, exclusive, reverse);
    } else if (dtype == TestDtype::BF16) {
        std::vector<uint16_t> q(numel);
        for (int64_t i = 0; i < numel; ++i) q[i] = FloatToBf16(fInput[i]);
        r = InvokeCumsum(stream, shape, acl, q.data(), q.size() * 2,
                         shape, acl, 2, dim, useV2, exclusive, reverse);
    } else {
        LOG_PRINT("[%s] BAD-DTYPE\n", name);
        ++g_failed; return;
    }

    if (r.wsStatus != ACLNN_SUCCESS || r.runStatus != ACLNN_SUCCESS) {
        LOG_PRINT("[FAIL] %s  ws=%d run=%d\n", name, r.wsStatus, r.runStatus);
        ++g_failed; return;
    }

    // CPU 参考
    std::vector<double> ref;
    CpuCumsumGeneric<float, double>(fInput, shape, dim, exclusive, reverse, ref,
                                    [](float v){ return static_cast<double>(v); },
                                    [](double v){ return v; });

    // 实测结果转 double
    std::vector<double> got(numel);
    if (dtype == TestDtype::FP32) {
        const float* p = reinterpret_cast<const float*>(r.outBytes.data());
        for (int64_t i = 0; i < numel; ++i) got[i] = static_cast<double>(p[i]);
    } else {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(r.outBytes.data());
        for (int64_t i = 0; i < numel; ++i)
            got[i] = static_cast<double>(dtype == TestDtype::FP16 ? Fp16ToFloat(p[i]) : Bf16ToFloat(p[i]));
    }

    double maxErr = 0.0; int64_t worst = -1;
    bool ok = VerifyClose<double>(ref, got, atol, rtol, maxErr, worst);
    LOG_PRINT("[%s] %s  dtype=%s shape=[", ok ? "PASS" : "FAIL", name, DtypeName(dtype));
    for (size_t i = 0; i < shape.size(); ++i) LOG_PRINT("%ld%s", shape[i], i + 1 == shape.size() ? "" : ",");
    LOG_PRINT("]  dim=%ld %s%s  maxAbsErr=%.3e (atol=%.1e rtol=%.1e)\n",
              dim, exclusive ? "excl " : "", reverse ? "rev " : "", maxErr, atol, rtol);
    if (ok) ++g_passed; else ++g_failed;
}

// 整数测试入口（精确匹配；CPU 端用 int64 累加避免参考侧自身溢出，再按 dtype 截断）
template <typename T>
static void RunIntCase(aclrtStream stream, const char* name, const std::vector<int64_t>& shape,
                       const std::vector<T>& input, int64_t dim, TestDtype dtype,
                       bool useV2, bool exclusive, bool reverse)
{
    aclDataType acl = ToAcl(dtype);
    int64_t numel = GetShapeSize(shape);
    InvokeResult r = InvokeCumsum(stream, shape, acl, input.data(), input.size() * sizeof(T),
                                  shape, acl, sizeof(T), dim, useV2, exclusive, reverse);
    if (r.wsStatus != ACLNN_SUCCESS || r.runStatus != ACLNN_SUCCESS) {
        LOG_PRINT("[FAIL] %s  ws=%d run=%d\n", name, r.wsStatus, r.runStatus);
        ++g_failed; return;
    }
    std::vector<int64_t> ref;
    CpuCumsumGeneric<T, int64_t>(input, shape, dim, exclusive, reverse, ref,
                                 [](T v){ return static_cast<double>(v); },
                                 [](double v){ return static_cast<int64_t>(v); });
    const T* p = reinterpret_cast<const T*>(r.outBytes.data());
    std::vector<int64_t> got(numel);
    for (int64_t i = 0; i < numel; ++i) got[i] = static_cast<int64_t>(p[i]);

    // 整数算子在 NPU 上同样按位回绕；将 CPU 参考截断为目标 dtype 的位宽即可对齐。
    for (int64_t i = 0; i < numel; ++i) ref[i] = static_cast<int64_t>(static_cast<T>(ref[i]));

    double maxErr = 0; int64_t worst = -1;
    bool ok = VerifyClose<int64_t>(ref, got, 0.0, 0.0, maxErr, worst);
    LOG_PRINT("[%s] %s  dtype=%s shape=[", ok ? "PASS" : "FAIL", name, DtypeName(dtype));
    for (size_t i = 0; i < shape.size(); ++i) LOG_PRINT("%ld%s", shape[i], i + 1 == shape.size() ? "" : ",");
    LOG_PRINT("]  dim=%ld %s%s\n", dim, exclusive ? "excl " : "", reverse ? "rev " : "");
    if (ok) ++g_passed; else ++g_failed;
}

// 异常用例：仅校验返回状态码符合预期。apiDtype 用于在 V1 路径上独立指定 `dtype` 参数。
static void RunErrorCase(aclrtStream stream, const char* name,
                         const std::vector<int64_t>& selfShape, aclDataType selfDtype,
                         const std::vector<int64_t>& outShape, aclDataType outDtype,
                         int64_t dim, bool useV2, bool nullSelf, bool nullOut,
                         aclnnStatus expected,
                         aclDataType apiDtype = ACL_DT_UNDEFINED)
{
    std::vector<float> filler(std::max<int64_t>(1, GetShapeSize(selfShape)), 1.0f);
    InvokeResult r = InvokeCumsum(stream, selfShape, selfDtype,
                                  filler.data(), filler.size() * 4,
                                  outShape, outDtype, 4,
                                  dim, useV2, false, false, nullSelf, nullOut, apiDtype);
    bool ok = (r.wsStatus == expected) || (r.runStatus == expected);
    LOG_PRINT("[%s] %s  expected=%d actualWs=%d run=%d\n",
              ok ? "PASS" : "FAIL", name, expected, r.wsStatus, r.runStatus);
    if (ok) ++g_passed; else ++g_failed;
}

// ---------------------------------------------------------------------------
// 主程序
// ---------------------------------------------------------------------------
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("================ Cumsum Comprehensive Test ================\n");

    // -----------------------------------------------------------------------
    // A. 基础正确性 —— 不同 rank、不同 dim、不同 dtype
    //    覆盖: aclnnCumsumGetWorkspaceSize 主路径、AiCore 分发、
    //          tiling 浮点路径 / 整数路径切换。
    // -----------------------------------------------------------------------
    {
        // A1: FP32 1D dim=0   → M=1,R=8,N=1   触达 RNLesserCl/MRNLesserCl
        std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8};
        RunFloatCase(stream, "A1-FP32-1D", {8}, x, 0, TestDtype::FP32, false, false, false, 1e-5, 1e-5);

        // A2: FP32 2D dim=0   → M=1,R=4,N=2   小 N 路径
        RunFloatCase(stream, "A2-FP32-2D-dim0", {4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, 0,
                     TestDtype::FP32, false, false, false, 1e-5, 1e-5);

        // A3: FP32 2D dim=1   → M=4,R=2,N=1   dim!=0 触发 INT32 dimTensor
        RunFloatCase(stream, "A3-FP32-2D-dim1", {4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, 1,
                     TestDtype::FP32, false, false, false, 1e-5, 1e-5);

        // A4: FP32 dim=-1     → 触发 CheckDim 内的负 dim 归一化
        RunFloatCase(stream, "A4-FP32-2D-negdim", {3, 5},
                     {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, -1,
                     TestDtype::FP32, false, false, false, 1e-5, 1e-5);

        // A5: FP16 dim=1 → 浮点 tiling NGreaterCl 路径 + dtCast=true 分支
        RunFloatCase(stream, "A5-FP16-2D-dim1", {4, 64}, [&] {
            std::vector<float> v(4 * 64);
            for (int i = 0; i < 4 * 64; ++i) v[i] = 0.25f * ((i % 8) + 1);
            return v;
        }(), 1, TestDtype::FP16, false, false, false, 1e-2, 1e-2);

        // A6: BF16 2D dim=1   → BF16 dtCast=true 分支
        RunFloatCase(stream, "A6-BF16-2D-dim1", {2, 16},
                     std::vector<float>(32, 1.0f), 1,
                     TestDtype::BF16, false, false, false, 5e-2, 5e-2);

        // A7: 3D FP32 dim=middle → M>1, N>1, R>1 走 NLesserCl 或 RNGreaterCl
        std::vector<float> a7(2 * 3 * 4);
        for (size_t i = 0; i < a7.size(); ++i) a7[i] = 0.1f * static_cast<float>(i + 1);
        RunFloatCase(stream, "A7-FP32-3D-dim1", {2, 3, 4}, a7, 1,
                     TestDtype::FP32, false, false, false, 1e-5, 1e-5);

        // A8: FP32 大 last-dim → tiling NGreaterCl + R 较大
        std::vector<float> a8(1 * 1024, 1.0f);
        RunFloatCase(stream, "A8-FP32-2D-bigN", {1, 1024}, a8, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // A9: FP32 大 dim 与多核切分 → MRN 大场景，触发 M 切核
        std::vector<float> a9(128 * 32);
        for (size_t i = 0; i < a9.size(); ++i) a9[i] = 1.0f;
        RunFloatCase(stream, "A9-FP32-2D-MN", {128, 32}, a9, 0,
                     TestDtype::FP32, false, false, false, 1e-4, 1e-5);
    }

    // -----------------------------------------------------------------------
    // B. 整数类型 —— 触达 cumsum_tiling_ascendc_int_arch35.cpp 各分支
    //    覆盖: TilingCumsumForAscendc 中 int 分支、Cumsum4IntTiling
    //          GetAxisLpUnit 三大分支(TD-RA/TD-LA/TD-R)、CalcTilingKey
    //          (CUM_WITH_GROUP / CUM_NO_SPLIT / CUM_AR_SPLIT)。
    // -----------------------------------------------------------------------
    {
        // B1: INT32 1D
        std::vector<int32_t> b1 = {1, 2, 3, 4, 5, 6, 7, 8};
        RunIntCase<int32_t>(stream, "B1-INT32-1D", {8}, b1, 0, TestDtype::I32, false, false, false);

        // B2: INT32 2D dim=1 大 last-axis → TD-RA 路径
        std::vector<int32_t> b2(8 * 256, 1);
        RunIntCase<int32_t>(stream, "B2-INT32-2D-bigRA", {8, 256}, b2, 1,
                            TestDtype::I32, false, false, false);

        // B3: INT32 2D dim=0 大 leftAxisLen → TD-LA 路径
        std::vector<int32_t> b3(256 * 4, 1);
        RunIntCase<int32_t>(stream, "B3-INT32-2D-bigLA", {256, 4}, b3, 0,
                            TestDtype::I32, false, false, false);

        // B4: INT64 长 1D
        std::vector<int64_t> b4(64);
        for (size_t i = 0; i < b4.size(); ++i) b4[i] = static_cast<int64_t>(i + 1);
        RunIntCase<int64_t>(stream, "B4-INT64-1D", {64}, b4, 0,
                            TestDtype::I64, false, false, false);

        // B5: INT8 → vlSize_/=2 分支
        std::vector<int8_t> b5(128, 1);
        RunIntCase<int8_t>(stream, "B5-INT8-1D", {128}, b5, 0,
                           TestDtype::I8, false, false, false);

        // B6: UINT8
        std::vector<uint8_t> b6(64, 2);
        RunIntCase<uint8_t>(stream, "B6-UINT8-1D", {64}, b6, 0,
                            TestDtype::U8, false, false, false);

        // B7: INT32 多维 dim=middle → CUM_WITH_GROUP
        std::vector<int32_t> b7(4 * 16 * 8, 1);
        RunIntCase<int32_t>(stream, "B7-INT32-3D-mid", {4, 16, 8}, b7, 1,
                            TestDtype::I32, false, false, false);

        // B8: INT32 溢出 —— 期望按位回绕，与 CPU 截断后参考一致
        std::vector<int32_t> b8 = {(1 << 30), (1 << 30), (1 << 30), (1 << 30)};
        RunIntCase<int32_t>(stream, "B8-INT32-Overflow", {4}, b8, 0,
                            TestDtype::I32, false, false, false);
    }

    // -----------------------------------------------------------------------
    // C. CumsumV2 —— exclusive / reverse 全组合
    //    覆盖: aclnnCumsumV2GetWorkspaceSize 主路径、CheckParamsWithoutDtype、
    //          带 attr 的 tiling 分支(GetAttrInfo)。
    // -----------------------------------------------------------------------
    {
        std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8};

        RunFloatCase(stream, "C1-V2-default",   {8}, x, 0, TestDtype::FP32, true, false, false, 1e-5, 1e-5);
        RunFloatCase(stream, "C2-V2-exclusive", {8}, x, 0, TestDtype::FP32, true, true,  false, 1e-5, 1e-5);
        RunFloatCase(stream, "C3-V2-reverse",   {8}, x, 0, TestDtype::FP32, true, false, true,  1e-5, 1e-5);
        RunFloatCase(stream, "C4-V2-excl-rev",  {8}, x, 0, TestDtype::FP32, true, true,  true,  1e-5, 1e-5);

        // V2 + 2D + dim=1 + reverse → 反向 sklansky 路径
        RunFloatCase(stream, "C5-V2-2D-rev", {4, 8},
                     std::vector<float>(32, 1.0f), 1, TestDtype::FP32, true, false, true,
                     1e-5, 1e-5);

        // V2 + INT32 → 整数 tiling 取 attr 路径
        std::vector<int32_t> ci = {1, 2, 3, 4, 5, 6};
        RunIntCase<int32_t>(stream, "C6-V2-INT32-excl-rev", {2, 3}, ci, 1,
                            TestDtype::I32, true, true, true);
    }

    // -----------------------------------------------------------------------
    // D. 边界 / 极端 shape
    // -----------------------------------------------------------------------
    {
        // D1: 单元素张量
        RunFloatCase(stream, "D1-FP32-Scalar1D", {1}, {3.14f}, 0,
                     TestDtype::FP32, false, false, false, 1e-6, 1e-6);

        // D2: 空 tensor —— self->IsEmpty() 早返回
        RunFloatCase(stream, "D2-FP32-EmptyDim",  {0}, {}, 0,
                     TestDtype::FP32, false, false, false, 1e-6, 1e-6);

        // D3: 空 tensor V2
        RunFloatCase(stream, "D3-V2-Empty",  {0, 4}, {}, 0,
                     TestDtype::FP32, true, false, false, 1e-6, 1e-6);

        // D4: 8 维（最大维度），每维都是 1，长度合法
        RunFloatCase(stream, "D4-FP32-8D", {1, 1, 1, 1, 1, 1, 1, 4},
                     {1.0f, 2.0f, 3.0f, 4.0f}, 7,
                     TestDtype::FP32, false, false, false, 1e-5, 1e-5);
    }

    // -----------------------------------------------------------------------
    // E. 精度专项 —— 误差累积
    //    每个用例对应测试报告中的某个精度场景。
    // -----------------------------------------------------------------------
    {
        // E1: FP32 累加 10000 个 1.0  → 误差 ~ n*eps，但理论值 10000 仍在 FP32 整数精确表示范围内。
        std::vector<float> e1(10000, 1.0f);
        RunFloatCase(stream, "E1-FP32-Accum10k", {10000}, e1, 0,
                     TestDtype::FP32, false, false, false, 1e-2, 1e-5);

        // E2: FP32 累加 10000 个 0.1 (无法精确表示) → 体现量化误差累积
        std::vector<float> e2(10000, 0.1f);
        RunFloatCase(stream, "E2-FP32-Accum-0.1", {10000}, e2, 0,
                     TestDtype::FP32, false, false, false, 1e-2, 1e-5);

        // E3: FP16 累加 1024 个 1.0 → FP16 在 ~2048 后已无法分辨整数差
        std::vector<float> e3(1024, 1.0f);
        RunFloatCase(stream, "E3-FP16-Accum1k", {1024}, e3, 0,
                     TestDtype::FP16, false, false, false, 2.0, 5e-3);

        // E4: 大小数混合 [1e8, 1e-6] x N  → 小数被吞没，FP32 仍可控；FP16 完全丢失
        std::vector<float> e4(2048);
        for (size_t i = 0; i < e4.size(); ++i) e4[i] = (i % 2 == 0) ? 1e8f : 1e-6f;
        RunFloatCase(stream, "E4-FP32-MixMag", {2048}, e4, 0,
                     TestDtype::FP32, false, false, false, 1e2, 1e-5);
        RunFloatCase(stream, "E4b-FP16-MixMag", {2048}, e4, 0,
                     TestDtype::FP16, false, false, false, 1e5, 1.0);

        // E5: 正负交替抵消 + 误差累积
        std::vector<float> e5(4096);
        for (size_t i = 0; i < e5.size(); ++i) e5[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        RunFloatCase(stream, "E5-FP32-AltSign", {4096}, e5, 0,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);
    }

    // -----------------------------------------------------------------------
    // F. 异常拦截 —— 触发 CheckParams 的全部错误分支
    //    注意：错误用例可能在 GetWorkspaceSize 阶段直接返回错误码，
    //          运行流程不会执行后续 kernel，但相关参数校验代码已被覆盖。
    // -----------------------------------------------------------------------
    {
        // F1: nullptr self
        RunErrorCase(stream, "F1-NullSelf", {4}, ACL_FLOAT, {4}, ACL_FLOAT,
                     0, false, true, false, ACLNN_ERR_PARAM_NULLPTR);

        // F2: nullptr out
        RunErrorCase(stream, "F2-NullOut", {4}, ACL_FLOAT, {4}, ACL_FLOAT,
                     0, false, false, true, ACLNN_ERR_PARAM_NULLPTR);

        // F3: dim 超出范围（rank=1, dim=5）
        RunErrorCase(stream, "F3-DimOOB", {4}, ACL_FLOAT, {4}, ACL_FLOAT,
                     5, false, false, false, ACLNN_ERR_PARAM_INVALID);

        // F4: 负 dim 越界
        RunErrorCase(stream, "F4-NegDimOOB", {4}, ACL_FLOAT, {4}, ACL_FLOAT,
                     -3, false, false, false, ACLNN_ERR_PARAM_INVALID);

        // F5: shape 不一致
        RunErrorCase(stream, "F5-ShapeMismatch", {4}, ACL_FLOAT, {5}, ACL_FLOAT,
                     0, false, false, false, ACLNN_ERR_PARAM_INVALID);

        // F6: V1 dtype 与 out 不匹配 → 触发 CheckDtypeValid 中的
        //     OP_CHECK_DTYPE_NOT_MATCH（apiDtype != out->dtype）
        RunErrorCase(stream, "F6-V1-DtypeNotMatch", {4}, ACL_FLOAT, {4}, ACL_FLOAT,
                     0, false, false, false, ACLNN_ERR_PARAM_INVALID, ACL_FLOAT16);

        // F6b: V2 self 与 out dtype 不一致 → 触发 CheckDtypeValidWithoutDtype 的
        //      OP_CHECK_DTYPE_NOT_SAME 分支（仅 V2 走这条路径）
        RunErrorCase(stream, "F6b-V2-SelfOutNotSame", {4}, ACL_FLOAT, {4}, ACL_FLOAT16,
                     0, true, false, false, ACLNN_ERR_PARAM_INVALID);

        // F7: 9 维超过 MAX_DIM
        RunErrorCase(stream, "F7-RankTooLarge",
                     {1, 1, 1, 1, 1, 1, 1, 1, 4}, ACL_FLOAT,
                     {1, 1, 1, 1, 1, 1, 1, 1, 4}, ACL_FLOAT,
                     8, false, false, false, ACLNN_ERR_PARAM_INVALID);

        // F8: V2 nullptr self
        RunErrorCase(stream, "F8-V2-NullSelf", {4}, ACL_FLOAT, {4}, ACL_FLOAT,
                     0, true, true, false, ACLNN_ERR_PARAM_NULLPTR);

        // F9: V2 dim 越界
        RunErrorCase(stream, "F9-V2-DimOOB", {4}, ACL_FLOAT, {4}, ACL_FLOAT,
                     7, true, false, false, ACLNN_ERR_PARAM_INVALID);

        // F10: V2 shape 不一致
        RunErrorCase(stream, "F10-V2-ShapeMismatch", {2, 3}, ACL_FLOAT, {3, 2}, ACL_FLOAT,
                     0, true, false, false, ACLNN_ERR_PARAM_INVALID);
    }

    // -----------------------------------------------------------------------
    // G. 浮点 tiling 分支扩展 —— 触达 cumsum_tiling_ascendc_arch35.cpp 中
    //    NGreaterCl / NGreaterClRFullLoad / NGreaterClRNotFullLoad / BorrowR /
    //    RNGreaterClRFullLoad(M>=coreNum, M<coreNum) 等子分支。
    //    思路：axis 选在中间或第一维，使得 N(轴后元素积) 显著 >= cacheLine,
    //    再搭配不同 R/M 组合触发 R 全载/不全载、M够分核/不够分核 的所有路径。
    // -----------------------------------------------------------------------
    {
        // G1: FP32 axis=1, M=128(>=核数) R=8 N=256 → NGreaterClRFullLoad + lenM>=coreNum + alignRN<=ub
        std::vector<float> g1(128 * 8 * 256, 1.0f);
        RunFloatCase(stream, "G1-FP32-NGtCl-RFull-MGE", {128, 8, 256}, g1, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // G2: FP32 axis=1, M=128 R=8 N=4096 → NGreaterClRFullLoad + lenM>=coreNum + alignRN<=ub (大N)
        std::vector<float> g2(128 * 8 * 4096, 1.0f);
        RunFloatCase(stream, "G2-FP32-NGtCl-RFull-bigN", {128, 8, 4096}, g2, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // G3: FP32 axis=1, M=128 R=16 N=8192 → alignRN > ub → "N不能全载" 子分支
        std::vector<float> g3(128 * 16 * 8192, 0.0f);
        for (size_t i = 0; i < g3.size(); ++i) g3[i] = static_cast<float>((i % 7) - 3);
        RunFloatCase(stream, "G3-FP32-NGtCl-RFull-NNotFull", {128, 16, 8192}, g3, 1,
                     TestDtype::FP32, false, false, false, 1e-2, 1e-5);

        // G4: FP32 axis=1, M=4 (<coreNum) R=8 N=4096 → NGreaterClRFullLoad + lenM<coreNum (借N)
        std::vector<float> g4(4 * 8 * 4096, 1.0f);
        RunFloatCase(stream, "G4-FP32-NGtCl-RFull-MLT", {4, 8, 4096}, g4, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // G5: FP16 axis=1 (dtCast=true) → 走 fp16 分支 + CAST_MULT 路径
        std::vector<float> g5f(16 * 8 * 1024);
        for (size_t i = 0; i < g5f.size(); ++i) g5f[i] = 0.5f * ((i % 4) - 1);
        RunFloatCase(stream, "G5-FP16-NGtCl-castMul", {16, 8, 1024}, g5f, 1,
                     TestDtype::FP16, false, false, false, 5e-1, 1e-2);

        // G6: BF16 axis=0 (M=1) 大 R*N → 走 NLesserCl/RNGreaterCl & dtCast 分支
        std::vector<float> g6(64 * 1024, 1.0f);
        RunFloatCase(stream, "G6-BF16-RNGtCl", {64, 1024}, g6, 0,
                     TestDtype::BF16, false, false, false, 5e0, 5e-2);

        // G7: FP32 axis=1, M=256 R=4096 N=8 → NGreaterClRNotFullLoad + lenM>coreNum/2
        std::vector<float> g7(256 * 4096 * 8, 0.0f);
        for (size_t i = 0; i < g7.size(); ++i) g7[i] = (i % 3 == 0) ? 1.0f : 0.0f;
        RunFloatCase(stream, "G7-FP32-NGtCl-RNotFull-MGT", {256, 4096, 8}, g7, 1,
                     TestDtype::FP32, false, false, false, 1e-1, 1e-5);

        // G8: FP32 axis=1, M=4 R=4096 N=8 → NGreaterClRNotFullLoad + lenM<=coreNum/2 → 借 N 后够分核
        std::vector<float> g8(4 * 4096 * 8, 1.0f);
        RunFloatCase(stream, "G8-FP32-NGtCl-RNotFull-borrowN", {4, 4096, 8}, g8, 1,
                     TestDtype::FP32, false, false, false, 1e-2, 1e-5);

        // G9: FP32 axis=1, M=2 R=8192 N=256 → 借 N 仍不够 → 借 R (NGreaterClRNotFullLoadBorrowR)
        std::vector<float> g9(2 * 8192 * 256, 0.0f);
        for (size_t i = 0; i < g9.size(); ++i) g9[i] = (i & 1) ? 1.0f : -1.0f;
        RunFloatCase(stream, "G9-FP32-NGtCl-borrowR", {2, 8192, 256}, g9, 1,
                     TestDtype::FP32, false, false, false, 1e-1, 1e-5);

        // G10: FP32 RNGreaterCl R 不全载 + 借R (二维 axis=1)
        std::vector<float> g10(2 * 16384, 0.5f);
        RunFloatCase(stream, "G10-FP32-RN-RNotFull", {2, 16384}, g10, 1,
                     TestDtype::FP32, false, false, false, 5e-2, 1e-5);

        // G11: FP32 axis=middle (3D) 大 R → RNGreaterClRNotFullLoadNotBorrowR
        std::vector<float> g11(64 * 8192 * 4, 0.0f);
        for (size_t i = 0; i < g11.size(); ++i) g11[i] = static_cast<float>(i % 5);
        RunFloatCase(stream, "G11-FP32-RN-Mid-Big", {64, 8192, 4}, g11, 1,
                     TestDtype::FP32, false, false, false, 1e-1, 1e-5);

        // G12: V2 + reverse + axis=middle → V2 路径下的 NGreaterCl
        std::vector<float> g12(8 * 32 * 256, 1.0f);
        RunFloatCase(stream, "G12-V2-rev-NGtCl", {8, 32, 256}, g12, 1,
                     TestDtype::FP32, true, false, true, 1e-3, 1e-5);

        // G13: V2 + exclusive + 3D axis=0 → V2 + 大 RN
        std::vector<float> g13(64 * 8 * 32, 1.0f);
        RunFloatCase(stream, "G13-V2-excl-3D", {64, 8, 32}, g13, 0,
                     TestDtype::FP32, true, true, false, 1e-3, 1e-5);
    }

    // -----------------------------------------------------------------------
    // H. CumsumCube 路径 —— aclnnCumsumGetWorkspaceSize 中
    //    CheckCubeSupport 返回 true 时调用 l0op::CumsumCube (line 304-311)
    //    条件：socVersion ∈ {ASCEND910B, ASCEND910_93}, fp32/fp16/bf16,
    //          last-dim, batchNum >= 12800, channelNum >= 512
    // -----------------------------------------------------------------------
    {
        // H1: FP32 [12800, 512] dim=1 → 触发 Cube 分支
        std::vector<float> h1(12800 * 512, 1.0f);
        RunFloatCase(stream, "H1-FP32-Cube", {12800, 512}, h1, 1,
                     TestDtype::FP32, false, false, false, 1e-2, 1e-5);

        // H2: FP16 [12800, 512] dim=-1 → FP16 Cube 在某些环境会挂死, 故先关闭
        // std::vector<float> h2(12800 * 512, 0.25f);
        // RunFloatCase(stream, "H2-FP16-Cube", {12800, 512}, h2, -1,
        //              TestDtype::FP16, false, false, false, 5e2, 5e-1);
    }

    // -----------------------------------------------------------------------
    // I. 整数 tiling 分支扩展 —— 触达 cumsum_tiling_ascendc_int_arch35.cpp 中
    //    AdjustTensor4TDRA / AdjustTensor4TDLA / AdjustTensor4TDR / CheckBGC /
    //    AdjustLARLpUnit / GetMCTilingInfo 的 RA / R-block 分支
    //    思路：axis 选中间维并使 rightAxisLen * dtypeSize > vlSize/2 触发 TDRA
    // -----------------------------------------------------------------------
    {
        // I1: INT32 axis=1, leftA=4 mid=8 rightA=256 → rightA*4=1024 > vlSize/2 → TDRA
        std::vector<int32_t> i1(4 * 8 * 256, 1);
        RunIntCase<int32_t>(stream, "I1-INT32-TDRA", {4, 8, 256}, i1, 1,
                            TestDtype::I32, false, false, false);

        // I2: INT32 axis=1 大 rightA → TDRA + 切核 RA
        std::vector<int32_t> i2(2 * 32 * 1024, 1);
        RunIntCase<int32_t>(stream, "I2-INT32-TDRA-bigRA", {2, 32, 1024}, i2, 1,
                            TestDtype::I32, false, false, false);

        // I3: INT32 axis=1 leftA 大 rightA 小 → 走 TDLA 分支 (line 213+)
        std::vector<int32_t> i3(256 * 16 * 4, 1);
        RunIntCase<int32_t>(stream, "I3-INT32-TDLA", {256, 16, 4}, i3, 1,
                            TestDtype::I32, false, false, false);

        // I4: INT32 axis=1 leftA 小 mid 大 → 走 TD-R 分支
        std::vector<int32_t> i4(2 * 1024 * 4, 1);
        RunIntCase<int32_t>(stream, "I4-INT32-TDR", {2, 1024, 4}, i4, 1,
                            TestDtype::I32, false, false, false);

        // I5: INT8 axis=1 触发 vlSize_/=2 路径 (line 49) + TDRA
        std::vector<int8_t> i5(4 * 8 * 64, 1);
        RunIntCase<int8_t>(stream, "I5-INT8-3D-TDRA", {4, 8, 64}, i5, 1,
                           TestDtype::I8, false, false, false);

        // I6: UINT8 axis=middle → 同上 vlSize_/=2 + 不同形状
        std::vector<uint8_t> i6(8 * 16 * 32, 2);
        RunIntCase<uint8_t>(stream, "I6-UINT8-3D", {8, 16, 32}, i6, 1,
                            TestDtype::U8, false, false, false);

        // I7: INT64 axis=middle → 64bit 整数走 TDRA-类似但 dtypeSize=8
        std::vector<int64_t> i7(2 * 4 * 64);
        for (size_t k = 0; k < i7.size(); ++k) i7[k] = static_cast<int64_t>(k + 1);
        RunIntCase<int64_t>(stream, "I7-INT64-3D", {2, 4, 64}, i7, 1,
                            TestDtype::I64, false, false, false);

        // I8: INT32 dim=-1 (负数 dim) → 触发 line 72 axisAttr += xDimNum
        std::vector<int32_t> i8(4 * 16, 1);
        RunIntCase<int32_t>(stream, "I8-INT32-NegDim", {4, 16}, i8, -1,
                            TestDtype::I32, false, false, false);

        // I9: V2 + INT64 → 触发 cumsum.cpp line 104 CumsumAiCpu(... excl, rev)
        std::vector<int64_t> i9(8);
        for (size_t k = 0; k < i9.size(); ++k) i9[k] = static_cast<int64_t>(k + 1);
        RunIntCase<int64_t>(stream, "I9-V2-INT64-excl-rev", {8}, i9, 0,
                            TestDtype::I64, true, true, true);

        // I10: V2 + INT8 → 同上不同 dtype
        std::vector<int8_t> i10(16, 1);
        RunIntCase<int8_t>(stream, "I10-V2-INT8", {16}, i10, 0,
                           TestDtype::I8, true, false, true);

        // I11: V2 + UINT8 + reverse
        std::vector<uint8_t> i11(8, 1);
        RunIntCase<uint8_t>(stream, "I11-V2-UINT8-rev", {8}, i11, 0,
                            TestDtype::U8, true, false, true);
    }

    // ===================================================================
    // K组：补 arch35 未覆盖分支 (NGreaterCl 子分支 + RNGreaterCl TWOWAY)
    // ===================================================================
    {
        // K1: FP32 [16,1024,64] dim=1 → NGreaterClRNotFullLoad IF (M>cn/2)
        std::vector<float> k1(16 * 1024 * 64, 1.0f);
        RunFloatCase(stream, "K1-FP32-NGtCl-RNotFull-IF", {16, 1024, 64}, k1, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K2: FP32 [2,1024,512] dim=1 → NGreaterClRNotFullLoad ELSE borrowN 够分核
        std::vector<float> k2(2 * 1024 * 512, 1.0f);
        RunFloatCase(stream, "K2-FP32-NGtCl-borrowN", {2, 1024, 512}, k2, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K3: FP32 [4,8192,64] dim=1 → BorrowR R不能全载
        std::vector<float> k3(4 * 8192 * 64, 1.0f);
        RunFloatCase(stream, "K3-FP32-BorrowR-RNotFull", {4, 8192, 64}, k3, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K4: FP32 [128,4096,8] dim=1 → RNGreaterCl TWOWAY + BorrowM
        std::vector<float> k4(128 * 4096 * 8, 1.0f);
        RunFloatCase(stream, "K4-FP32-Twoway-BorrowM", {128, 4096, 8}, k4, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K5: FP16 [16,16,16] dim=1 → twoway full load
        std::vector<float> k5(16 * 16 * 16, 1.0f);
        RunFloatCase(stream, "K5-FP16-Twoway-FullLoad", {16, 16, 16}, k5, 1,
                     TestDtype::FP16, false, false, false, 5e-1, 1e-2);

        // K6: FP32 [16,16384,4] dim=1 → twoway RNotFullLoad NotBorrowR
        std::vector<float> k6(16 * 16384 * 4, 1.0f);
        RunFloatCase(stream, "K6-FP32-Twoway-NotBorrowR", {16, 16384, 4}, k6, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K7: FP32 [4,16384,4] dim=1 → twoway BorrowR
        std::vector<float> k7(4 * 16384 * 4, 1.0f);
        RunFloatCase(stream, "K7-FP32-Twoway-BorrowR", {4, 16384, 4}, k7, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K8: FP32 [128,32,8] dim=1 → MRNGreaterCl 分支
        std::vector<float> k8(128 * 32 * 8, 1.0f);
        RunFloatCase(stream, "K8-FP32-MRNGtCl", {128, 32, 8}, k8, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K9: FP32 [4,4,4] dim=1 → MRNLesserCl 边界
        std::vector<float> k9(4 * 4 * 4, 1.0f);
        RunFloatCase(stream, "K9-FP32-MRNLesser", {4, 4, 4}, k9, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K10: FP32 [3,3,3] dim=1 → 严格小于 cl 的 MRNLesserCl
        std::vector<float> k10(3 * 3 * 3, 1.0f);
        RunFloatCase(stream, "K10-FP32-MRNLesser-tiny", {3, 3, 3}, k10, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K11: BF16 [16,16,16] dim=1 → BF16 twoway full load
        std::vector<float> k11(16 * 16 * 16, 1.0f);
        RunFloatCase(stream, "K11-BF16-Twoway-FullLoad", {16, 16, 16}, k11, 1,
                     TestDtype::BF16, false, false, false, 5e-1, 1e-2);

        // K12: BF16 [4,8192,4] dim=1 → BF16 twoway BorrowR
        std::vector<float> k12(4 * 8192 * 4, 1.0f);
        RunFloatCase(stream, "K12-BF16-Twoway-BorrowR", {4, 8192, 4}, k12, 1,
                     TestDtype::BF16, false, false, false, 5e-1, 1e-2);

        // K13: FP32 [64,2048,8] dim=1 → twoway BorrowM 中等规模
        std::vector<float> k13(64 * 2048 * 8, 1.0f);
        RunFloatCase(stream, "K13-FP32-Twoway-BorrowM-mid", {64, 2048, 8}, k13, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K14: FP32 [32,32,16] dim=1 → twoway 中等
        std::vector<float> k14(32 * 32 * 16, 1.0f);
        RunFloatCase(stream, "K14-FP32-Twoway-mid", {32, 32, 16}, k14, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K15: FP32 [8,512,128] dim=1 → NGreaterCl，N较大
        std::vector<float> k15(8 * 512 * 128, 1.0f);
        RunFloatCase(stream, "K15-FP32-NGtCl-N128", {8, 512, 128}, k15, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K16: FP32 [1,4096,256] dim=1 → 单 M, 大 N
        std::vector<float> k16(1 * 4096 * 256, 1.0f);
        RunFloatCase(stream, "K16-FP32-singleM-bigN", {1, 4096, 256}, k16, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K17: INT32 [256,4,4] dim=1 → leftA大 rightA小 → TDLA
        std::vector<int32_t> k17(256 * 4 * 4, 1);
        RunIntCase<int32_t>(stream, "K17-INT32-TDLA-leftBig", {256, 4, 4}, k17, 1,
                            TestDtype::I32, false, false, false);

        // K18: INT8 [16,4,512] dim=1 → INT8 TDRA + leftA 中等
        std::vector<int8_t> k18(16 * 4 * 512, 1);
        RunIntCase<int8_t>(stream, "K18-INT8-TDRA-bigRA", {16, 4, 512}, k18, 1,
                           TestDtype::I8, false, false, false);

        // K19: INT64 [4,16,8] dim=1
        std::vector<int64_t> k19(4 * 16 * 8, static_cast<int64_t>(1));
        RunIntCase<int64_t>(stream, "K19-INT64-3D-mid", {4, 16, 8}, k19, 1,
                            TestDtype::I64, false, false, false);

        // K20: INT32 [128,8,16] dim=1 → CheckBGC + AdjustLARLpUnit
        std::vector<int32_t> k20(128 * 8 * 16, 1);
        RunIntCase<int32_t>(stream, "K20-INT32-AdjLAR", {128, 8, 16}, k20, 1,
                            TestDtype::I32, false, false, false);

        // K21: INT32 [2,32,2048] dim=1 → 大 rightA 触发 TDRA 内部 cl-align 分支
        std::vector<int32_t> k21(2 * 32 * 2048, 1);
        RunIntCase<int32_t>(stream, "K21-INT32-bigRA-clAlign", {2, 32, 2048}, k21, 1,
                            TestDtype::I32, false, false, false);

        // K22: FP32 [24,256,64] dim=1 → 刚好 coreNum 个 M
        std::vector<float> k22(24 * 256 * 64, 1.0f);
        RunFloatCase(stream, "K22-FP32-M=coreNum", {24, 256, 64}, k22, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K23: FP16 [128,1024,8] dim=1 → FP16 twoway BorrowM
        std::vector<float> k23(128 * 1024 * 8, 1.0f);
        RunFloatCase(stream, "K23-FP16-Twoway-BorrowM", {128, 1024, 8}, k23, 1,
                     TestDtype::FP16, false, false, false, 5e-1, 1e-2);

        // K24: FP32 [12,2048,64] dim=1 → 边界 M=coreNum/2
        std::vector<float> k24(12 * 2048 * 64, 1.0f);
        RunFloatCase(stream, "K24-FP32-M=cn/2", {12, 2048, 64}, k24, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);

        // K25: FP32 [13,2048,64] dim=1 → 刚刚 > coreNum/2
        std::vector<float> k25(13 * 2048 * 64, 1.0f);
        RunFloatCase(stream, "K25-FP32-M=cn/2+1", {13, 2048, 64}, k25, 1,
                     TestDtype::FP32, false, false, false, 1e-3, 1e-5);
    }

    // -----------------------------------------------------------------------
    // 汇总
    // -----------------------------------------------------------------------
    LOG_PRINT("===========================================================\n");
    LOG_PRINT("Summary: %d passed, %d failed\n", g_passed, g_failed);
    LOG_PRINT("===========================================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return g_failed == 0 ? 0 : 1;
}
