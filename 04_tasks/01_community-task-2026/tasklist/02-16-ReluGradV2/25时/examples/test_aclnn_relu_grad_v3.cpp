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
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <cfloat>

#include "acl/acl.h"
#include <cstdlib>
#include "aclnnop/aclnn_relu_grad_v3.h"

#define CHECK_RET(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); goto cleanup; } } while (0)


static const char* g_filter_dtype = nullptr;
static const char* g_filter_shape = nullptr;

static bool ShouldRun(const char* dtype_name, bool is_small) {
    if (g_filter_dtype && strcmp(g_filter_dtype, dtype_name) != 0) return false;
    if (g_filter_shape && strcmp(g_filter_shape, is_small ? "small" : "large") != 0) return false;
    return true;
}
static int64_t GetShapeSize(const std::vector<int64_t>& s) {
    int64_t sz = 1; for (auto x : s) sz *= x; return sz;
}

static int Init(int32_t devId, aclrtStream* st) {
    aclError ret;
    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] aclInit ret=%d, err=%s\n", ret, aclGetRecentErrMsg());
        return 1;
    }
    ret = aclrtSetDevice(devId);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] setDevice ret=%d, err=%s\n", ret, aclGetRecentErrMsg());
        return 1;
    }
    ret = aclrtCreateStream(st);
    if (ret != ACL_SUCCESS) {
        printf("  [FAIL] createStream ret=%d, err=%s\n", ret, aclGetRecentErrMsg());
        return 1;
    }
    //printf("  [INFO] aclInit+setDevice+createStream OK\n");
    return 0;
}

static void Cleanup(aclTensor* t[3], void* d[3], void* ws, aclOpExecutor* ex, aclrtStream st, int devId) {
    for (int i = 0; i < 3; ++i) { if (t[i]) aclDestroyTensor(t[i]); if (d[i]) aclrtFree(d[i]); }
    if (ws) aclrtFree(ws);
    if (ex) aclDestroyAclOpExecutor(ex);
    if (st) aclrtDestroyStream(st);
    aclrtResetDevice(devId);
    aclFinalize();
    //printf("  [INFO] Cleanup done\n");
}

static uint16_t F2H(float f) { uint32_t b; memcpy(&b,&f,4); return ((b>>16)&0x8000)|((((b>>23)&0xFF)-127+15)<<10)|((b>>13)&0x3FF); }
static uint16_t F2B(float f) { uint32_t b; memcpy(&b,&f,4); return b>>16; }

template <typename T, typename U = T>
int RunOneDtype(const std::vector<int64_t>& shape, const char* name,
                 const std::vector<T>& gradVals, const std::vector<uint8_t>& maskVals,
                 aclDataType dt, double tol) {
    int64_t N = GetShapeSize(shape), devId = 0;
    aclrtStream st = nullptr;
    void* devs[3] = {};
    aclTensor* tensors[3] = {};
    void* wsAddr = nullptr;
    aclOpExecutor* exe = nullptr;
    int failed = 0;
    uint64_t wsSize = 0;       // 函数作用域
    std::vector<U> expected;  // 函数作用域，供后续校验使用

    //printf("\n  === [%s] shape=[", name);
    //for (size_t i = 0; i < shape.size(); ++i) printf("%lld%s", (long long)shape[i], i+1<shape.size()?",":"");
    //printf("] N=%lld dtype=%d ===\n", (long long)N, (int)dt);

    // ---- Init ACL ----
    if (Init(devId, &st) != 0) { failed = 1; goto cleanup; }

    // ---- 准备数据 ----
    {
        std::vector<U> grad(N);
        expected.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            grad[i] = static_cast<U>(gradVals[i % gradVals.size()]);
            expected[i] = (maskVals[i % maskVals.size()] != 0) ? grad[i] : static_cast<U>(0);
        }

        // grad tensor
        auto sz = N * sizeof(U);
        CHECK_RET(aclrtMalloc(&devs[0], sz, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "grad malloc");
        CHECK_RET(aclrtMemcpy(devs[0], sz, grad.data(), sz, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "grad H2D");
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = (int)shape.size()-2; i >= 0; --i) strides[i] = shape[i+1]*strides[i+1];
        tensors[0] = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), devs[0]);
        CHECK_RET(tensors[0] != nullptr, "create grad tensor");

        // mask tensor
        auto maskBytes = N;
        std::vector<uint8_t> mask(N);
        for (int64_t i = 0; i < N; ++i) mask[i] = maskVals[i % maskVals.size()];
        CHECK_RET(aclrtMalloc(&devs[1], maskBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "mask malloc");
        CHECK_RET(aclrtMemcpy(devs[1], maskBytes, mask.data(), maskBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "mask H2D");
        tensors[1] = aclCreateTensor(shape.data(), shape.size(), ACL_UINT8, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), devs[1]);
        CHECK_RET(tensors[1] != nullptr, "create mask tensor");

        // out tensor
        auto outBytes = N * sizeof(U);
        CHECK_RET(aclrtMalloc(&devs[2], outBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "out malloc");
        tensors[2] = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), devs[2]);
        CHECK_RET(tensors[2] != nullptr, "create out tensor");

        //printf("  [INFO] 3 tensors created, dev memory allocated\n");
    }

    // ---- GetWorkspaceSize ----
    {
        wsSize = 0;
        aclnnStatus wsRet = aclnnReluGradV3GetWorkspaceSize(tensors[0], tensors[1], tensors[2], &wsSize, &exe);
        if (wsRet != ACL_SUCCESS) {
            printf("  [FAIL] GetWorkspaceSize ret=%d, wsSize=%llu\n", (int)wsRet, (unsigned long long)wsSize);
            printf("  [FAIL] aclGetRecentErrMsg: %s\n", aclGetRecentErrMsg());
            failed = 1;
            goto cleanup;
        }
        //printf("  [INFO] GetWorkspaceSize OK, wsSize=%llu bytes (%.2f MB)\n",
        //       (unsigned long long)wsSize, wsSize / (1024.0 * 1024.0));

        if (wsSize) {
            CHECK_RET(aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "ws alloc");
            //printf("  [INFO] workspace allocated\n");
        }
    }

    // ---- Kernel 执行 ----
    {
        CHECK_RET(aclnnReluGradV3(wsAddr, wsSize, exe, st) == ACL_SUCCESS, "kernel launch");
        aclError syncRet = aclrtSynchronizeStream(st);
        if (syncRet != ACL_SUCCESS) {
            printf("  [FAIL] stream sync ret=%d, err=%s\n",
                   syncRet, aclGetRecentErrMsg());
            failed = 1;
            goto cleanup;
        }
    }

    // ---- 结果校验 ----
    {
        // 半精度浮点类型: 0x8000(-0.0) 与 0x0000(+0.0) 应等价
        auto canonZero = [dt](uint16_t v) -> uint16_t {
            if ((dt == ACL_FLOAT16 || dt == ACL_BF16) && v == 0x8000u)
                return 0x0000u;
            return v;
        };
        auto toCompareVal = [dt, canonZero](uint16_t raw) -> double {
            if (dt == ACL_BF16) { uint32_t bits = (uint32_t)canonZero(raw) << 16; float f; memcpy(&f, &bits, 4); return (double)f; }
            // float16 / 整数: 直接值比较（已消除 -0.0）
            return (double)canonZero(raw);
        };

        std::vector<U> out(N);
        CHECK_RET(aclrtMemcpy(out.data(), N * sizeof(U), devs[2], N * sizeof(U), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS, "out D2H");

        int pass = 0;
        for (int64_t i = 0; i < N; ++i) {
            double diff = fabs(toCompareVal(out[i]) - toCompareVal(expected[i]));
            if (diff <= tol) pass++;
        }

        bool isProf = (g_filter_dtype != nullptr && g_filter_shape != nullptr);
        if (!isProf) {
            // 打印比对结果
            printf("\n  [%s] shape=[", name);
            for (size_t i = 0; i < shape.size(); ++i) printf("%lld%s", (long long)shape[i], i+1<shape.size()?",":"");
            printf("] tol=%.4f\n", tol);
            // grad
            printf("    grad    :{");
            for (int64_t i = 0; i < N; ++i) {
                if (i>0) printf(",");
                if (dt == ACL_FLOAT16 || dt == ACL_BF16)
                    printf("0x%04X", (unsigned)gradVals[i % gradVals.size()]);
                else if (dt == ACL_FLOAT)
                    printf("%g", (double)gradVals[i % gradVals.size()]);
                else
                    printf("%d", (int)gradVals[i % gradVals.size()]);
            }
            printf("}\n");
            // mask
            printf("    mask    :{");
            for (int64_t i = 0; i < N; ++i) printf("%s%d", i>0?",":"", (int)maskVals[i % maskVals.size()]);
            printf("}\n");
            // expected
            printf("    expected:{");
            for (int64_t i = 0; i < N; ++i) {
                if (i>0) printf(",");
                if (dt == ACL_FLOAT16 || dt == ACL_BF16)
                    printf("0x%04X", (unsigned)expected[i]);
                else if (dt == ACL_FLOAT)
                    printf("%g", (double)expected[i]);
                else
                    printf("%d", (int)expected[i]);
            }
            printf("}\n");
            // got
            printf("    got     :{");
            for (int64_t i = 0; i < N; ++i) {
                if (i>0) printf(",");
                if (dt == ACL_FLOAT16 || dt == ACL_BF16)
                    printf("0x%04X", (unsigned)out[i]);
                else if (dt == ACL_FLOAT)
                    printf("%g", (double)out[i]);
                else
                    printf("%d", (int)out[i]);
            }
            printf("}");
            if (pass == N) printf("  -- PASS\n");
            else printf("  -- FAIL (%lld/%lld)\n", (long long)(N-pass), (long long)N);
        }

        // 如果失败，打印前几个错误值
        if (pass != N) {
            int printed = 0;
            for (int64_t i = 0; i < N && printed < 5; ++i) {
                double diff = fabs(toCompareVal(out[i]) - toCompareVal(expected[i]));
                if (diff > tol) {
                    printf("    idx=%lld: expected=%g, got=%g, diff=%g\n",
                           (long long)i, toCompareVal(expected[i]), toCompareVal(out[i]), diff);
                    printed++;
                }
            }
            failed = 1;
        }
    }

cleanup:
    Cleanup(tensors, devs, wsAddr, exe, st, devId);
    return failed;
}

int main() {
    g_filter_dtype = getenv("RELU_DTYPE");
    g_filter_shape = getenv("RELU_SHAPE");
    std::vector<int64_t> small_shape = {3, 3};
    bool isProfMode = (g_filter_dtype != nullptr && g_filter_shape != nullptr);

    if (!isProfMode) {
    printf("============================================================\n");
    printf("ReluGradV3 test: 6 dtypes, shape={3,3}\n");
    printf("============================================================\n");
    }

    std::vector<uint8_t> mask_small = {1,0,1,0,1,0,1,0,1};

    int failed = 0;

    bool runSmall = ShouldRun("float", true) || ShouldRun("int32", true) ||
                    ShouldRun("int8", true) || ShouldRun("uint8", true) ||
                    ShouldRun("float16", true) || ShouldRun("bf16", true);

    // ============ small shape ============
    if (runSmall) {
    //printf("\n--- small shape {");
    //for (size_t i = 0; i < small_shape.size(); ++i)
    //    printf("%lld%s", (long long)small_shape[i], i+1 < small_shape.size() ? "," : "");
    //printf("} ---\n");
    if (ShouldRun("float", true))
        failed += RunOneDtype<float,float>(small_shape, "float", {3.f,-2.f,0.f,1.f,-5.f,0.f,FLT_MAX,FLT_MIN,0.f}, mask_small, ACL_FLOAT, 0.001);
    if (ShouldRun("int32", true))
        failed += RunOneDtype<int32_t,int32_t>(small_shape, "int32", {3,-2,0,1,-5,0,2147483647,-2147483648,0}, mask_small, ACL_INT32, 0);
    if (ShouldRun("int8", true))
        failed += RunOneDtype<int8_t,int8_t>(small_shape, "int8", {3,-2,0,1,-5,0,127,-128,0}, mask_small, ACL_INT8, 0);
    if (ShouldRun("uint8", true))
        failed += RunOneDtype<uint8_t,uint8_t>(small_shape, "uint8", {3,2,0,1,5,0,255,0,0}, mask_small, ACL_UINT8, 0);
    if (ShouldRun("float16", true))
        failed += RunOneDtype<uint16_t,uint16_t>(small_shape, "fp16", {F2H(3),F2H(-2),F2H(0),F2H(1),F2H(-5),F2H(0),0x0400U,F2H(0),F2H(0)}, {1,0,1,0,1,0,0,1,0}, ACL_FLOAT16, 0.01);
    if (ShouldRun("bf16", true))
        failed += RunOneDtype<uint16_t,uint16_t>(small_shape, "bf16", {F2B(3),F2B(-2),F2B(0),F2B(1),F2B(-5),F2B(0),0x0080U,F2B(0),F2B(0)}, {1,0,1,0,1,0,0,1,0}, ACL_BF16, 0.01);
    }

    return failed;
}