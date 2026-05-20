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
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { return_expr; } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t s = 1;
    for (auto d : shape) s *= d;
    return s;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed\n"); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("set device failed\n"); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create stream failed\n"); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                     void** devAddr, aclDataType dtype, aclTensor** tensor) {
    int64_t bytes = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc failed\n"); return ret);
    ret = aclrtMemcpy(*devAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memcpy failed\n"); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size()-2; i >= 0; --i)
        strides[i] = shape[i+1] * strides[i+1];

    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *devAddr);
    return 0;
}

bool AlmostEqual(double e, double a, double atol=1e-5, double rtol=1e-5) {
    if (std::isnan(e) && std::isnan(a)) return true;
    if (std::isinf(e) && std::isinf(a)) return (e>0)==(a>0);
    return std::fabs(a-e) <= atol + rtol * std::fabs(e);
}

// ==========================
// 用例1：标准加法 tensor + tensor
// ==========================
int TestAdd(aclrtStream stream) {
    std::vector<int64_t> shape = {2,2};
    std::vector<float> x1 = {1,2,3,4};
    std::vector<float> x2 = {10,20,30,40};
    std::vector<float> out(4, 0);
    float a_val = 1.0f;

    void *d1 = nullptr, *d2 = nullptr, *dout = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    void *waddr = nullptr;
    aclOpExecutor *exec = nullptr;
    int fail = 0;

    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(out, shape, &dout, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    auto ret = aclnnAddGetWorkspaceSize(t1, t2, alpha, tout, &ws, &exec);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (ws) aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnAdd(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out.data(), 4*sizeof(float), dout, 4*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int i=0; i<4; i++) {
        double expect = x1[i] + a_val * x2[i];
        if (!AlmostEqual(expect, out[i])) fail++;
    }

cleanup:
    if (exec) aclDestroyAclOpExecutor(exec);
    if (t1) aclDestroyTensor(t1);
    if (t2) aclDestroyTensor(t2);
    if (tout) aclDestroyTensor(tout);
    if (alpha) aclDestroyScalar(alpha);
    if (d1) aclrtFree(d1);
    if (d2) aclrtFree(d2);
    if (dout) aclrtFree(dout);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] Add\n" : "[PASS] Add\n");
    return fail;
}

// ==========================
// 用例2：广播 shape 测试
// ==========================
int TestAddBroadcast(aclrtStream stream) {
    std::vector<int64_t> sA = {2,3};
    std::vector<int64_t> sB = {1,3};
    std::vector<float> A = {1,2,3, 4,5,6};
    std::vector<float> B = {0.1f, 0.2f, 0.3f};
    std::vector<float> out(6, 0);
    float a_val = 1.0f;
    std::vector<float> expect = {1.1f, 2.2f, 3.3f, 4.1f, 5.2f, 6.3f};

    void *dA = nullptr, *dB = nullptr, *dOut = nullptr;
    aclTensor *tA = nullptr, *tB = nullptr, *tOut = nullptr;
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    void *waddr = nullptr;
    aclOpExecutor *exec = nullptr;
    int fail = 0;

    CreateAclTensor(A, sA, &dA, ACL_FLOAT, &tA);
    CreateAclTensor(B, sB, &dB, ACL_FLOAT, &tB);
    CreateAclTensor(out, sA, &dOut, ACL_FLOAT, &tOut);

    uint64_t ws = 0;
    auto ret = aclnnAddGetWorkspaceSize(tA, tB, alpha, tOut, &ws, &exec);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (ws) aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnAdd(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out.data(), 6*sizeof(float), dOut, 6*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int i=0; i<6; i++) {
        if (!AlmostEqual(expect[i], out[i])) fail++;
    }

cleanup:
    if (exec) aclDestroyAclOpExecutor(exec);
    if (tA) aclDestroyTensor(tA);
    if (tB) aclDestroyTensor(tB);
    if (tOut) aclDestroyTensor(tOut);
    if (alpha) aclDestroyScalar(alpha);
    if (dA) aclrtFree(dA);
    if (dB) aclrtFree(dB);
    if (dOut) aclrtFree(dOut);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] AddBroadcast\n" : "[PASS] AddBroadcast\n");
    return fail;
}

// ==========================
// 用例3：tensor + 标量 Adds
// ==========================
int TestAdds(aclrtStream stream) {
    std::vector<int64_t> shape = {2,2};
    std::vector<float> x1 = {1,2,3,4};
    float scalar = 5.0f;
    std::vector<float> out(4, 0);
    float a_val = 2.0f;

    void *d1 = nullptr, *dout = nullptr;
    aclTensor *t1 = nullptr, *tout = nullptr;
    aclScalar *s = aclCreateScalar(&scalar, ACL_FLOAT);
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    void *waddr = nullptr;
    aclOpExecutor *exec = nullptr;
    int fail = 0;

    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(out, shape, &dout, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    auto ret = aclnnAddsGetWorkspaceSize(t1, s, alpha, tout, &ws, &exec);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (ws) aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnAdds(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out.data(), 4*sizeof(float), dout, 4*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int i=0; i<4; i++) {
        double e = x1[i] + a_val * scalar;
        if (!AlmostEqual(e, out[i])) fail++;
    }

cleanup:
    if (exec) aclDestroyAclOpExecutor(exec);
    if (t1) aclDestroyTensor(t1);
    if (tout) aclDestroyTensor(tout);
    if (s) aclDestroyScalar(s);
    if (alpha) aclDestroyScalar(alpha);
    if (d1) aclrtFree(d1);
    if (dout) aclrtFree(dout);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] Adds\n" : "[PASS] Adds\n");
    return fail;
}

// ==========================
// 用例4：InplaceAdd（关键覆盖率）
// ==========================
int TestInplaceAdd(aclrtStream stream) {
    std::vector<int64_t> shape = {2,2};
    std::vector<float> self = {1,2,3,4};
    std::vector<float> other = {10,20,30,40};
    float a_val = 1.0f;

    void *dself = nullptr, *dother = nullptr;
    aclTensor *tself = nullptr, *tother = nullptr;
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    void *waddr = nullptr;
    aclOpExecutor *exec = nullptr;
    int fail = 0;

    CreateAclTensor(self, shape, &dself, ACL_FLOAT, &tself);
    CreateAclTensor(other, shape, &dother, ACL_FLOAT, &tother);

    uint64_t ws = 0;
    auto ret = aclnnInplaceAddGetWorkspaceSize(tself, tother, alpha, &ws, &exec);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (ws) aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceAdd(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(self.data(), 4*sizeof(float), dself, 4*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int i=0; i<4; i++) {
        double e = (i+1) + a_val * (i+1)*10;
        if (!AlmostEqual(e, self[i])) fail++;
    }

cleanup:
    if (exec) aclDestroyAclOpExecutor(exec);
    if (tself) aclDestroyTensor(tself);
    if (tother) aclDestroyTensor(tother);
    if (alpha) aclDestroyScalar(alpha);
    if (dself) aclrtFree(dself);
    if (dother) aclrtFree(dother);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] InplaceAdd\n" : "[PASS] InplaceAdd\n");
    return fail;
}

// ==========================
// 用例5：InplaceAdds（关键覆盖率）
// ==========================
int TestInplaceAdds(aclrtStream stream) {
    std::vector<int64_t> shape = {2,2};
    std::vector<float> self = {1,2,3,4};
    float scalar = 5.0f;
    float a_val = 1.0f;

    void *dself = nullptr;
    aclTensor *tself = nullptr;
    aclScalar *s = aclCreateScalar(&scalar, ACL_FLOAT);
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    void *waddr = nullptr;
    aclOpExecutor *exec = nullptr;
    int fail = 0;

    CreateAclTensor(self, shape, &dself, ACL_FLOAT, &tself);

    uint64_t ws = 0;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(tself, s, alpha, &ws, &exec);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (ws) aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceAdds(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(self.data(), 4*sizeof(float), dself, 4*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int i=0; i<4; i++) {
        double e = (i+1) + a_val * scalar;
        if (!AlmostEqual(e, self[i])) fail++;
    }

cleanup:
    if (exec) aclDestroyAclOpExecutor(exec);
    if (tself) aclDestroyTensor(tself);
    if (s) aclDestroyScalar(s);
    if (alpha) aclDestroyScalar(alpha);
    if (dself) aclrtFree(dself);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] InplaceAdds\n" : "[PASS] InplaceAdds\n");
    return fail;
}

// ==========================
// 用例6：广播极端情况 (4,1)+(1,4)
// ==========================
int TestAddBroadcastExtreme(aclrtStream stream) {
    void* dA = nullptr;
    void* dB = nullptr;
    void* dOut = nullptr;
    aclTensor* tA = nullptr;
    aclTensor* tB = nullptr;
    aclTensor* tOut = nullptr;
    aclScalar* alpha = nullptr;
    void* waddr = nullptr;
    aclOpExecutor* exec = nullptr;
    int fail = 0;

    std::vector<int64_t> shapeA = {4, 1};
    std::vector<int64_t> shapeB = {1, 4};
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {10, 20, 30, 40};
    std::vector<float> out(16, 0);
    float a_val = 1.0f;

    alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    CreateAclTensor(A, shapeA, &dA, ACL_FLOAT, &tA);
    CreateAclTensor(B, shapeB, &dB, ACL_FLOAT, &tB);
    CreateAclTensor(out, {4, 4}, &dOut, ACL_FLOAT, &tOut);

    uint64_t ws = 0;
    aclnnAddGetWorkspaceSize(tA, tB, alpha, tOut, &ws, &exec);
    if (ws) {
        aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnAdd(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out.data(), 16 * sizeof(float), dOut, 16 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> expect = {
        1+10,1+20,1+30,1+40,
        2+10,2+20,2+30,2+40,
        3+10,3+20,3+30,3+40,
        4+10,4+20,4+30,4+40
    };
    for (int i = 0; i < 16; i++) {
        if (!AlmostEqual(expect[i], out[i])) {
            fail++;
        }
    }

    if (exec) aclDestroyAclOpExecutor(exec);
    if (tA) aclDestroyTensor(tA);
    if (tB) aclDestroyTensor(tB);
    if (tOut) aclDestroyTensor(tOut);
    if (alpha) aclDestroyScalar(alpha);
    if (dA) aclrtFree(dA);
    if (dB) aclrtFree(dB);
    if (dOut) aclrtFree(dOut);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] AddBroadcastExtreme\n" : "[PASS] AddBroadcastExtreme\n");
    return fail;
}

// ==========================
// 用例7：INT32 类型
// ==========================
int TestAddInt32(aclrtStream stream) {
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    aclScalar* alpha = nullptr;
    void* waddr = nullptr;
    aclOpExecutor* exec = nullptr;
    int fail = 0;

    std::vector<int64_t> shape = {2,2};
    std::vector<int32_t> x1 = {1, 2, 3, 4};
    std::vector<int32_t> x2 = {10, 20, 30, 40};
    std::vector<int32_t> out(4, 0);
    float a_val = 1.0f;

    alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    CreateAclTensor(x1, shape, &d1, ACL_INT32, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_INT32, &t2);
    CreateAclTensor(out, shape, &dout, ACL_INT32, &tout);

    uint64_t ws = 0;
    aclnnAddGetWorkspaceSize(t1, t2, alpha, tout, &ws, &exec);
    if (ws) {
        aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnAdd(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out.data(), 4 * sizeof(int32_t), dout, 4 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int i = 0; i < 4; i++) {
        int32_t expect = x1[i] + x2[i];
        if (out[i] != expect) {
            fail++;
        }
    }

    if (exec) aclDestroyAclOpExecutor(exec);
    if (t1) aclDestroyTensor(t1);
    if (t2) aclDestroyTensor(t2);
    if (tout) aclDestroyTensor(tout);
    if (alpha) aclDestroyScalar(alpha);
    if (d1) aclrtFree(d1);
    if (d2) aclrtFree(d2);
    if (dout) aclrtFree(dout);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] AddInt32\n" : "[PASS] AddInt32\n");
    return fail;
}

// ==========================
// 用例8：边界值测试（零、负数）
// ==========================
int TestAddEdgeCases(aclrtStream stream) {
    std::vector<int64_t> shape = {2,2};
    std::vector<float> x1 = {0.0f, -1.0f, -5.5f, 1e-6f};
    std::vector<float> x2 = {0.0f, 1.0f, -3.2f, -1e-6f};
    std::vector<float> out(4, 0);
    std::vector<float> expect = {0.0f, 0.0f, -8.7f, 0.0f};  // ← 移到前面
    float a_val = 1.0f;

    void *d1 = nullptr, *d2 = nullptr, *dout = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    void *waddr = nullptr;
    aclOpExecutor *exec = nullptr;
    int fail = 0;

    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(out, shape, &dout, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    auto ret = aclnnAddGetWorkspaceSize(t1, t2, alpha, tout, &ws, &exec);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (ws) aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnAdd(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out.data(), 4*sizeof(float), dout, 4*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 验证
    for (int i=0; i<4; i++) {
        if (!AlmostEqual(expect[i], out[i])) fail++;
    }

cleanup:
    if (exec) aclDestroyAclOpExecutor(exec);
    if (t1) aclDestroyTensor(t1);
    if (t2) aclDestroyTensor(t2);
    if (tout) aclDestroyTensor(tout);
    if (alpha) aclDestroyScalar(alpha);
    if (d1) aclrtFree(d1);
    if (d2) aclrtFree(d2);
    if (dout) aclrtFree(dout);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] AddEdgeCases\n" : "[PASS] AddEdgeCases\n");
    return fail;
}

// ==========================
// 用例9：极大值测试
// ==========================
int TestAddLargeValues(aclrtStream stream) {
    std::vector<int64_t> shape = {2,2};
    std::vector<float> x1 = {1e30f, 1e20f, -1e30f, 1e10f};
    std::vector<float> x2 = {1e30f, 1e20f, 1e30f, 1e10f};
    std::vector<float> out(4, 0);
    float a_val = 1.0f;

    void *d1 = nullptr, *d2 = nullptr, *dout = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    void *waddr = nullptr;
    aclOpExecutor *exec = nullptr;
    int fail = 0;

    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(out, shape, &dout, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    auto ret = aclnnAddGetWorkspaceSize(t1, t2, alpha, tout, &ws, &exec);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (ws) aclrtMalloc(&waddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnAdd(waddr, ws, exec, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out.data(), 4*sizeof(float), dout, 4*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 验证：inf, 2e20, 0, 2e10
    if (!std::isinf(out[0]) || out[0] < 0) fail++;
    if (!AlmostEqual(2e20f, out[1], 1e15f, 1e-3)) fail++;
    if (!AlmostEqual(0.0f, out[2], 1e-3, 1e-3)) fail++;
    if (!AlmostEqual(2e10f, out[3], 1e5, 1e-3)) fail++;

cleanup:
    if (exec) aclDestroyAclOpExecutor(exec);
    if (t1) aclDestroyTensor(t1);
    if (t2) aclDestroyTensor(t2);
    if (tout) aclDestroyTensor(tout);
    if (alpha) aclDestroyScalar(alpha);
    if (d1) aclrtFree(d1);
    if (d2) aclrtFree(d2);
    if (dout) aclrtFree(dout);
    if (waddr) aclrtFree(waddr);

    LOG_PRINT(fail ? "[FAIL] AddLargeValues\n" : "[PASS] AddLargeValues\n");
    return fail;
}

int main() {
    int32_t dev = 0;
    aclrtStream stream;
    int ret = Init(dev, &stream);
    CHECK_RET(ret == 0, return 1);

    int total = 0;
    total += TestAdd(stream);                    // 基础 Add
    total += TestAddBroadcast(stream);           // 广播 (2,3)+(1,3)
    total += TestAdds(stream);                   // tensor + scalar
    total += TestInplaceAdd(stream);             // InplaceAdd
    total += TestInplaceAdds(stream);            // InplaceAdds
    total += TestAddBroadcastExtreme(stream);    // 极端广播 (4,1)+(1,4)
    total += TestAddInt32(stream);               // INT32 类型
    total += TestAddEdgeCases(stream);           // 边界值
    total += TestAddLargeValues(stream);         // 极大值

    LOG_PRINT("=== total failed: %d ===\n", total);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(dev);
    aclFinalize();
    return total;
}