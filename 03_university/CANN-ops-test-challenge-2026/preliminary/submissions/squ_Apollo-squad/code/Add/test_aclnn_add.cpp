// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// #include <iostream>
// #include <vector>
// #include "acl/acl.h"
// #include "aclnnop/aclnn_add.h"

// #define CHECK_RET(cond, return_expr) \
//     do {                             \
//         if (!(cond)) {               \
//             return_expr;             \
//         }                            \
//     } while (0)

// #define LOG_PRINT(message, ...)         \
//     do {                                \
//         printf(message, ##__VA_ARGS__); \
//     } while (0)

// int64_t GetShapeSize(const std::vector<int64_t>& shape)
// {
//     int64_t shapeSize = 1;
//     for (auto i : shape) {
//         shapeSize *= i;
//     }
//     return shapeSize;
// }

// int Init(int32_t deviceId, aclrtStream* stream)
// {
//     // 固定写法，资源初始化
//     auto ret = aclInit(nullptr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtSetDevice(deviceId);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtCreateStream(stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
//     return 0;
// }

// template <typename T>
// int CreateAclTensor(
//     const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
//     aclTensor** tensor)
// {
//     auto size = GetShapeSize(shape) * sizeof(T);
//     // 调用aclrtMalloc申请device侧内存
//     auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
//     // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
//     ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

//     // 计算连续tensor的strides
//     std::vector<int64_t> strides(shape.size(), 1);
//     for (int64_t i = shape.size() - 2; i >= 0; i--) {
//         strides[i] = shape[i + 1] * strides[i + 1];
//     }

//     // 调用aclCreateTensor接口创建aclTensor
//     *tensor = aclCreateTensor(
//         shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
//         *deviceAddr);
//     return 0;
// }

// int main()
// {
//     // 1. （固定写法）device/stream初始化，参考acl API手册
//     // 根据自己的实际device填写deviceId
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {4, 2};
//     std::vector<int64_t> otherShape = {4, 2};
//     std::vector<int64_t> outShape = {4, 2};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
//     std::vector<double> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
//     std::vector<double> outHostData(8, 0);
//     double alphaValue = 1.2f;
//     // 创建self aclTensor
//     ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     // 创建other aclTensor
//     ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_DOUBLE, &other);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     // 创建alpha aclScalar
//     alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_DOUBLE);
//     CHECK_RET(alpha != nullptr, return ret);
//     // 创建out aclTensor
//     ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);

//     uint64_t workspaceSize = 0;
//     aclOpExecutor* executor;

//     // aclnnAdd接口调用示例
//     // 3. 调用CANN算子库API
//     // 调用aclnnAdd第一段接口
//     ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
//     // 根据第一段接口计算出的workspaceSize申请device内存
//     void* workspaceAddr = nullptr;
//     if (workspaceSize > 0) {
//         ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
//         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
//     }
//     // 调用aclnnAdd第二段接口
//     ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);

//     // 4. （固定写法）同步等待任务执行结束
//     ret = aclrtSynchronizeStream(stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

//     // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
//     auto size = GetShapeSize(outShape);
//     std::vector<double> resultData(size, 0);
//     ret = aclrtMemcpy(
//         resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
//         ACL_MEMCPY_DEVICE_TO_HOST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//     for (int64_t i = 0; i < size; i++) {
//         LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
//     }

//     // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
//     aclDestroyTensor(self);
//     aclDestroyTensor(other);
//     aclDestroyScalar(alpha);
//     aclDestroyTensor(out);

//     // 7. 释放Device资源，需要根据具体API的接口定义修改
//     aclrtFree(selfDeviceAddr);
//     aclrtFree(otherDeviceAddr);
//     aclrtFree(outDeviceAddr);
//     if (workspaceSize > 0) {
//         aclrtFree(workspaceAddr);
//     }
//     aclrtDestroyStream(stream);
//     aclrtResetDevice(deviceId);
//     aclFinalize();

//     return 0;
// }





/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comprehensive test file for Add operator - maximizing code coverage
 *
 * Coverage targets:
 *   aclnn_add.cpp     : CheckParams, aclnnAddGetWorkspaceSize (all branches), aclnnAddsGetWorkspaceSize,
 *                       aclnnInplaceAddGetWorkspaceSize, aclnnInplaceAddsGetWorkspaceSize
 *   aclnn_add_v3.cpp  : CheckParams(V3), aclnnAddV3GetWorkspaceSize (3 branches), aclnnInplaceAddV3GetWorkspaceSize
 *   add.cpp           : Add(), AddAiCore(), AddAiCpu(), IsAddSupportNonContiguous(), GetAiCoreDtypeSupportList()
 *   add_tiling_arch35.cpp : DoOpTiling() - all dtype branches (fp16,bf16,fp32,bool,int64,uint8,int8,int32,mixedDtype)
 *
 * Branch matrix for aclnnAdd:
 *   isMixDataType && alpha==1  → direct Add (mixed dtype, no cast)
 *   alpha==1, same dtype       → Add directly (possibly noncontiguous)
 *   alpha==1, diff dtype       → Cast+Cast+Add
 *   alpha!=1, IsSupportAxpy    → Cast+Cast+Axpy  (FLOAT/FLOAT16/INT32 on regbase)
 *   alpha!=1, IsSupportAxpyV2  → Cast+Cast+AxpyV2 (FLOAT/BF16/FLOAT16/INT32/INT64/INT8/UINT8/BOOL)
 *   alpha!=1, else             → Cast+Cast+Mul+Add
 *
 * Branch matrix for aclnnAddV3 (scalar + alpha*tensor):
 *   alpha==1 → Add(selfTensor, otherCasted)
 *   alpha!=1, IsSupportAxpy(promoteType) → Axpy
 *   alpha!=1, else → Mul+Add
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <functional>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

/* ─────────────────────── macros ─────────────────────── */
#define CHECK_RET(cond, return_expr)  \
    do {                              \
        if (!(cond)) { return_expr; } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { printf(message, ##__VA_ARGS__); } while (0)

/* ─────────────────────── globals ─────────────────────── */
static int g_passCount = 0;
static int g_failCount = 0;

/* ─────────────────────── helpers ─────────────────────── */
static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto d : shape) s *= d;
    return s;
}

static int InitACL(int32_t deviceId, aclrtStream* stream)
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
static int CreateAclTensor(const std::vector<T>& hostData,
                            const std::vector<int64_t>& shape,
                            void** deviceAddr,
                            aclDataType dataType,
                            aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--)
        strides[i] = shape[i + 1] * strides[i + 1];

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                               strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                               shape.data(), shape.size(), *deviceAddr);
    return 0;
}

static void RunAndSync(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
                       aclnnStatus (*execFn)(void*, uint64_t, aclOpExecutor*, aclrtStream))
{
    if (workspaceSize > 0) {
        void* ws = nullptr;
        aclrtMalloc(&ws, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        execFn(ws, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        aclrtFree(ws);
    } else {
        execFn(nullptr, 0, executor, stream);
        aclrtSynchronizeStream(stream);
    }
}

static void PrintResult(const std::string& testName, bool passed)
{
    if (passed) {
        LOG_PRINT("[PASS] %s\n", testName.c_str());
        g_passCount++;
    } else {
        LOG_PRINT("[FAIL] %s\n", testName.c_str());
        g_failCount++;
    }
}

/* tolerance check */
static bool CheckClose(double actual, double expected, double atol = 1e-3, double rtol = 1e-3)
{
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual) && (expected > 0) == (actual > 0)) return true;
    return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 1: aclnnAdd  FLOAT32, alpha=1  (same dtype, alpha==1 → direct Add path)
 *  Covers: aclnn_add.cpp IsEqualToOne→true, same dtype branch
 *          add_tiling_arch35.cpp DT_FLOAT branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 alpha=1 (same shape)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData  = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    std::vector<float> otherData = {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f};
    std::vector<float> outData(8, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, aclDataType::ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    // read back & verify
    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 1.0 * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 2: aclnnAdd  FLOAT32, alpha=2.0  (IsSupportAxpy path on regbase FLOAT)
 *  Covers: aclnn_add.cpp IsSupportAxpy branch (FLOAT in ARCH_REGBASE_AXPY_DTYPE_SUPPORT_LIST)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 alpha=2.0 (Axpy path)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData  = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    std::vector<float> otherData = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    std::vector<float> outData(8, 0.f);
    float alphaVal = 2.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 2.0 * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 3: aclnnAdd  FLOAT16, alpha=1  
 *  Covers: add_tiling_arch35.cpp DT_FLOAT16 branch, ARCH_REGBASE_AXPY_DTYPE_SUPPORT_LIST
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float16_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT16 alpha=1 (fp16 tiling branch)";
    std::vector<int64_t> shape = {4, 2};
    // store as uint16 (fp16 bit pattern for small integers is same as truncated float)
    // use simple values: 1.0h=0x3C00, 2.0h=0x4000, 3.0h=0x4200, 4.0h=0x4400
    std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
    std::vector<uint16_t> outData(8, 0);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    // Just verify no crash and return success
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 4: aclnnAdd  FLOAT16, alpha=2.0  (Axpy with FLOAT16)
 *  Covers: ARCH_REGBASE_AXPY_DTYPE_SUPPORT_LIST FLOAT16 entry
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float16_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT16 alpha=2.0 (Axpy fp16)";
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400}; // 1,2,3,4
    std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00}; // 1,1,1,1
    std::vector<uint16_t> outData(4, 0);
    float alphaVal = 2.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 5: aclnnAdd  INT32, alpha=1
 *  Covers: add_tiling_arch35.cpp DT_INT32 branch (AddWithoutCastCompute<int32_t>)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int32_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT32 alpha=1 (int32 tiling branch)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int32_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<int32_t> otherData = {10,10,10,10,10,10,10,10};
    std::vector<int32_t> outData(8, 0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT32, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<int32_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int32_t expected = selfData[i] + 1 * otherData[i];
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 6: aclnnAdd  INT32, alpha=3  (IsSupportAxpy INT32 path)
 *  Covers: ARCH_REGBASE_AXPY_DTYPE_SUPPORT_LIST INT32, alpha!=1 Axpy branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int32_Alpha3(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT32 alpha=3 (Axpy int32)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int32_t> selfData  = {1,2,3,4,5,6,7,8};
    std::vector<int32_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<int32_t> outData(8, 0);
    float alphaVal = 3.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT32, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<int32_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int32_t expected = selfData[i] + 3 * otherData[i];
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 7: aclnnAdd  INT8, alpha=1
 *  Covers: add_tiling_arch35.cpp DT_INT8 branch (AddWithoutCastCompute<int8_t>)
 *          add.cpp ASCEND910B_AICORE_DTYPE_SUPPORT_LIST INT8
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int8_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT8 alpha=1 (int8 tiling branch)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int8_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<int8_t> otherData = {10,10,10,10,10,10,10,10};
    std::vector<int8_t> outData(8, 0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT8, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<int8_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int8_t), outDev, 8*sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int8_t expected = selfData[i] + otherData[i];
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 8: aclnnAdd  INT8, alpha=2  (AxpyV2 path for INT8 on regbase)
 *  Covers: IsSupportAxpyV2 branch (INT8 in ARCH_REGBASE_AXPY_V2_DTYPE_SUPPORT_LIST)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int8_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT8 alpha=2 (AxpyV2 int8)";
    std::vector<int64_t> shape = {2, 4};
    std::vector<int8_t> selfData  = {1,2,3,4,5,6,7,8};
    std::vector<int8_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<int8_t> outData(8, 0);
    int32_t alphaVal = 2;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT8, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 9: aclnnAdd  UINT8, alpha=1
 *  Covers: add_tiling_arch35.cpp DT_UINT8 branch (AddWithoutCastCompute<uint8_t>)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Uint8_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd UINT8 alpha=1 (uint8 tiling branch)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint8_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<uint8_t> otherData = {10,10,10,10,10,10,10,10};
    std::vector<uint8_t> outData(8, 0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_UINT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_UINT8, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_UINT8, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<uint8_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(uint8_t), outDev, 8*sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        uint8_t expected = selfData[i] + otherData[i];
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 10: aclnnAdd  INT64, alpha=1
 *  Covers: add_tiling_arch35.cpp DT_INT64 branch (AddWithoutCastCompute<int64_t>)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int64_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT64 alpha=1 (int64 tiling branch)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int64_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<int64_t> otherData = {100,100,100,100,100,100,100,100};
    std::vector<int64_t> outData(8, 0);
    int64_t alphaVal = 1;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT64, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT64, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT64, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<int64_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int64_t), outDev, 8*sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        ok &= (result[i] == selfData[i] + otherData[i]);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 11: aclnnAdd  INT64, alpha=2  (AxpyV2 path for INT64 on regbase)
 *  Covers: ARCH_REGBASE_AXPY_V2_DTYPE_SUPPORT_LIST INT64, AxpyV2 branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int64_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT64 alpha=2 (AxpyV2 int64)";
    std::vector<int64_t> shape = {2, 4};
    std::vector<int64_t> selfData  = {10,20,30,40,50,60,70,80};
    std::vector<int64_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<int64_t> outData(8, 0);
    int64_t alphaVal = 2;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT64, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT64, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT64, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 12: aclnnAdd  BOOL, alpha=1
 *  Covers: add_tiling_arch35.cpp DT_BOOL branch (AddBoolCompute<int8_t>)
 *          aclnn_add.cpp AxpyV2 BOOL entry
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Bool_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd BOOL alpha=1 (bool tiling branch)";
    std::vector<int64_t> shape = {2, 4};
    std::vector<uint8_t> selfData  = {0,1,1,0,1,0,0,1};
    std::vector<uint8_t> otherData = {1,0,1,0,0,1,0,1};
    std::vector<uint8_t> outData(8, 0);
    bool alphaVal = true;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BOOL);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_BOOL, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_BOOL, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_BOOL, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 13: aclnnAdd  FLOAT32 broadcast (shape {4,1} + {1,4} → {4,4})
 *  Covers: CheckShape broadcast path, tiling with broadcast
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_Broadcast(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 broadcast {4,1}+{1,4}→{4,4}";
    std::vector<int64_t> selfShape  = {4, 1};
    std::vector<int64_t> otherShape = {1, 4};
    std::vector<int64_t> outShape   = {4, 4};
    std::vector<float> selfData  = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> otherData = {10.f, 20.f, 30.f, 40.f};
    std::vector<float> outData(16, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  selfShape,  &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   outShape,   &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(16, 0.f);
    aclrtMemcpy(result.data(), 16*sizeof(float), outDev, 16*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int r = 0; r < 4 && ok; r++)
        for (int c = 0; c < 4 && ok; c++) {
            double expected = (double)selfData[r] + (double)otherData[c];
            ok &= CheckClose((double)result[r*4+c], expected);
        }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 14: aclnnAdd  FLOAT32 alpha=0
 *  Covers: IsEqualToOne returns false (0 ≠ 1) → Axpy/AxpyV2 branch with alpha=0
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_Alpha0(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 alpha=0 (out == self)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> otherData = {100.f,100.f,100.f,100.f,100.f,100.f,100.f,100.f};
    std::vector<float> outData(8, 0.f);
    float alphaVal = 0.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 0.0 * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected, 1e-2, 1e-2);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 15: aclnnAdd  FLOAT32 alpha=-1.5  (negative alpha)
 *  Covers: Axpy with negative alpha value
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_AlphaNeg(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 alpha=-1.5 (negative alpha)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData  = {10.f,10.f,10.f,10.f,10.f,10.f,10.f,10.f};
    std::vector<float> otherData = {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
    std::vector<float> outData(8, 0.f);
    float alphaVal = -1.5f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + (-1.5) * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 16: aclnnAdd  mixed dtype FLOAT16+FLOAT32 (MixDtype path, alpha=1)
 *  Covers: isAddMixDtypeSupport == true, alpha==1 → mixed kernel path
 *          add_tiling_arch35.cpp isMixedDtype && input1Dtype==DT_FLOAT branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_MixedDtype_F16F32_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT16+FLOAT32 mixed alpha=1 (mixed dtype kernel)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800}; // fp16
    std::vector<float>    otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f}; // fp32
    std::vector<float>    outData(8, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT,   &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT,   &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 17: aclnnAdd  mixed dtype FLOAT32+FLOAT16 (input0=fp32, input1=fp16)
 *  Covers: add_tiling_arch35.cpp isMixedDtype && input0Dtype==DT_FLOAT branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_MixedDtype_F32F16_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32+FLOAT16 mixed alpha=1 (mixed tiling fp32+fp16)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float>    selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00}; // fp16 1.0
    std::vector<float>    outData(8, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT,   &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT,   &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 18: aclnnAdd  BF16, alpha=1
 *  Covers: add_tiling_arch35.cpp DT_BF16 branch (AddWithCastCompute<half>)
 *          ARCH_REGBASE_AXPY_DTYPE_SUPPORT_LIST BF16
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_BF16_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd BF16 alpha=1 (bf16 tiling branch)";
    std::vector<int64_t> shape = {4, 2};
    // BF16: 1.0 = 0x3F80, 2.0 = 0x4000, 3.0 = 0x4040
    std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
    std::vector<uint16_t> outData(8, 0);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_BF16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_BF16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 19: aclnnAdd  BF16, alpha=2
 *  Covers: ARCH_REGBASE_AXPY_DTYPE_SUPPORT_LIST BF16 + alpha!=1 Axpy
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_BF16_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd BF16 alpha=2 (bf16 Axpy branch)";
    std::vector<int64_t> shape = {2, 4};
    std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
    std::vector<uint16_t> outData(8, 0);
    float alphaVal = 2.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_BF16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_BF16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 20: aclnnAdds  FLOAT32 + scalar, alpha=1  (tensor + scalar)
 *  Covers: aclnnAddsGetWorkspaceSize, IsEqualToOne→true branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdds FLOAT32 + scalar alpha=1";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8, 0.f);
    float otherVal = 10.0f;
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 1.0 * 10.0;
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 21: aclnnAdds  FLOAT32 + scalar, alpha=2.5  (Axpy path)
 *  Covers: aclnnAdds IsSupportAxpy branch, non-unit alpha
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Float32_Alpha2p5(aclrtStream stream)
{
    const std::string name = "aclnnAdds FLOAT32 + scalar alpha=2.5 (Axpy)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8, 0.f);
    float otherVal = 3.0f;
    float alphaVal = 2.5f;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 2.5 * 3.0;
        ok &= CheckClose((double)result[i], expected, 1e-2, 1e-2);
    }

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 22: aclnnAdds  INT32 + scalar, alpha=1  (INT32 scalar add)
 *  Covers: PromoteTypeScalar INT32 path (non-floating self, non-floating other)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Int32_Scalar_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdds INT32 + scalar alpha=1";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int32_t> selfData = {1,2,3,4,5,6,7,8};
    std::vector<int32_t> outData(8, 0);
    int32_t otherVal = 100;
    int32_t alphaVal = 1;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_INT32, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    std::vector<int32_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int32_t expected = selfData[i] + 100;
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 23: aclnnAdds  BOOL + BOOL scalar, alpha=BOOL(true)
 *  Covers: bool clamping logic in aclnnAddsGetWorkspaceSize
 *          (self==BOOL, other==BOOL, alpha==BOOL, out!=BOOL → extra Cast)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Bool_BoolScalar_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdds BOOL + bool scalar alpha=true (bool clamp branch)";
    std::vector<int64_t> shape = {2, 4};
    std::vector<uint8_t> selfData = {0,1,1,0,0,1,0,1};
    std::vector<int32_t> outData(8, 0); // out is INT32, not BOOL
    bool otherVal = true;
    bool alphaVal = true;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_BOOL);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BOOL);

    CreateAclTensor(selfData, shape, &selfDev, ACL_BOOL,  &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_INT32, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 24: aclnnInplaceAdd  FLOAT32, alpha=1  (inplace tensor+tensor)
 *  Covers: aclnnInplaceAddGetWorkspaceSize → CheckInplace → aclnnAddGetWorkspaceSize
 *          aclnnInplaceAdd execution path
 * ═══════════════════════════════════════════════════════════ */
static void Test_InplaceAdd_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnInplaceAdd FLOAT32 alpha=1 (in-place)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdd);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), selfDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 25: aclnnInplaceAdd  FLOAT32, alpha=2 (in-place with Axpy)
 * ═══════════════════════════════════════════════════════════ */
static void Test_InplaceAdd_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnInplaceAdd FLOAT32 alpha=2 (in-place Axpy)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData  = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    std::vector<float> otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    float alphaVal = 2.0f;

    void *selfDev=nullptr, *otherDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdd);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), selfDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 2.0 * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 26: aclnnInplaceAdds  FLOAT32 += scalar * alpha
 *  Covers: aclnnInplaceAddsGetWorkspaceSize → aclnnAddsGetWorkspaceSize
 *          aclnnInplaceAdds execution
 * ═══════════════════════════════════════════════════════════ */
static void Test_InplaceAdds_Float32(aclrtStream stream)
{
    const std::string name = "aclnnInplaceAdds FLOAT32 += scalar";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float otherVal = 5.0f;
    float alphaVal = 1.0f;

    void *selfDev=nullptr;
    aclTensor *self=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdds);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), selfDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 5.0;
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 27: aclnnInplaceAdds  INT32 += scalar * 3
 *  Covers: InplaceAdds with integer and non-unit alpha
 * ═══════════════════════════════════════════════════════════ */
static void Test_InplaceAdds_Int32_Alpha3(aclrtStream stream)
{
    const std::string name = "aclnnInplaceAdds INT32 alpha=3";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int32_t> selfData = {0,1,2,3,4,5,6,7};
    int32_t otherVal = 2;
    float alphaVal = 3.0f;

    void *selfDev=nullptr;
    aclTensor *self=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdds);

    aclDestroyTensor(self);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 28: aclnnAddV3  scalar + alpha*tensor, alpha=1  (V3 alpha==1 branch)
 *  Covers: aclnn_add_v3.cpp alpha==1 → direct Add(selfTensor, otherCasted)
 * ═══════════════════════════════════════════════════════════ */
static void Test_AddV3_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAddV3 FLOAT32 alpha=1 (scalar + tensor)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8, 0.f);
    float selfVal  = 10.0f;
    float alphaVal = 1.0f;

    void *otherDev=nullptr, *outDev=nullptr;
    aclTensor *other=nullptr, *out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = 10.0 + 1.0 * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 29: aclnnAddV3  scalar + alpha*tensor, alpha=2  (V3 IsSupportAxpy path)
 *  Covers: aclnn_add_v3.cpp IsSupportAxpy branch (FLOAT in AXPY_DTYPE_SUPPORT_LIST)
 * ═══════════════════════════════════════════════════════════ */
static void Test_AddV3_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAddV3 FLOAT32 alpha=2 (Axpy branch)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8, 0.f);
    float selfVal  = 5.0f;
    float alphaVal = 2.0f;

    void *otherDev=nullptr, *outDev=nullptr;
    aclTensor *other=nullptr, *out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = 5.0 + 2.0 * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 30: aclnnAddV3  INT32 scalar + alpha=3*tensor  (V3 else: Mul+Add)
 *  Covers: aclnn_add_v3.cpp else branch (not alpha==1, not IsSupportAxpy for INT32 in V3's list)
 *  Note: V3's AXPY_DTYPE_SUPPORT_LIST = {FLOAT, INT32, FLOAT16}
 *        INT32 IS in the list, so it will use Axpy. Test with INT8 for Mul+Add else branch.
 * ═══════════════════════════════════════════════════════════ */
static void Test_AddV3_Int32_Alpha3(aclrtStream stream)
{
    const std::string name = "aclnnAddV3 INT32 alpha=3 (Axpy int32 V3)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int32_t> otherData = {1,2,3,4,5,6,7,8};
    std::vector<int32_t> outData(8, 0);
    int32_t selfVal  = 100;
    float alphaVal = 3.0f;

    void *otherDev=nullptr, *outDev=nullptr;
    aclTensor *other=nullptr, *out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

    std::vector<int32_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int32_t expected = 100 + 3 * otherData[i];
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 31: aclnnAddV3  INT8 scalar + alpha=2*tensor  (V3 else: Mul+Add)
 *  Covers: V3 else branch since INT8 not in AXPY_DTYPE_SUPPORT_LIST={FLOAT,INT32,FLOAT16}
 * ═══════════════════════════════════════════════════════════ */
static void Test_AddV3_Int8_AlphaMulAdd(aclrtStream stream)
{
    const std::string name = "aclnnAddV3 INT8 alpha=2 (Mul+Add else branch)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int8_t> otherData = {1,2,3,4,5,6,7,8};
    std::vector<int8_t> outData(8, 0);
    int8_t  selfVal  = 10;
    float   alphaVal = 2.0f;

    void *otherDev=nullptr, *outDev=nullptr;
    aclTensor *other=nullptr, *out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_INT8);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT8, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

    std::vector<int8_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int8_t), outDev, 8*sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int expected = (int)selfVal + 2 * (int)otherData[i];
        ok &= (result[i] == (int8_t)expected);
    }

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 32: aclnnAddV3  FLOAT16 scalar + alpha*tensor
 *  Covers: V3 with fp16 type (in V3 AXPY_DTYPE_SUPPORT_LIST)
 * ═══════════════════════════════════════════════════════════ */
static void Test_AddV3_Float16_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAddV3 FLOAT16 alpha=2 (Axpy fp16 V3)";
    std::vector<int64_t> shape = {2, 4};
    std::vector<uint16_t> otherData = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800}; // 1,2,3,4...
    std::vector<uint16_t> outData(8, 0);
    uint16_t selfVal16 = 0x4200; // 3.0 in fp16
    float alphaVal = 2.0f;

    void *otherDev=nullptr, *outDev=nullptr;
    aclTensor *other=nullptr, *out=nullptr;
    // self is aclScalar with fp16 type - use float value that fits in fp16
    float selfValF = 3.0f;
    aclScalar* self  = aclCreateScalar(&selfValF, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 33: aclnnInplaceAddV3  scalar inplace add to tensor
 *  Covers: aclnnInplaceAddV3GetWorkspaceSize → aclnnAddV3GetWorkspaceSize
 *          aclnnInplaceAddV3 execution
 * ═══════════════════════════════════════════════════════════ */
static void Test_InplaceAddV3_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnInplaceAddV3 FLOAT32 alpha=1 (in-place V3)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float selfVal  = 10.0f;
    float alphaVal = 1.0f;

    void *otherDev=nullptr;
    aclTensor *other=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAddV3);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), otherDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    // InplaceAddV3: out = scalar + alpha*tensor (result stored back to other)
    for (int i = 0; i < 8 && ok; i++) {
        double expected = 10.0 + 1.0 * (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(other);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 34: aclnnInplaceAddV3  FLOAT32 alpha=2
 *  Covers: InplaceAddV3 with Axpy branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_InplaceAddV3_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnInplaceAddV3 FLOAT32 alpha=2 (in-place V3 Axpy)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float selfVal  = 100.0f;
    float alphaVal = 2.0f;

    void *otherDev=nullptr;
    aclTensor *other=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAddV3);

    aclDestroyTensor(other);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 35: aclnnAdd  larger tensor (stress coverage of tiling logic)
 *  Covers: large N triggering multi-block tiling
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_LargeTensor(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 large tensor (1024x1024)";
    std::vector<int64_t> shape = {1024, 1024};
    int64_t N = 1024*1024;
    std::vector<float> selfData(N, 1.0f);
    std::vector<float> otherData(N, 2.0f);
    std::vector<float> outData(N, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    // spot check a few values
    std::vector<float> result(N, 0.f);
    aclrtMemcpy(result.data(), N*sizeof(float), outDev, N*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    ok &= CheckClose((double)result[0], 3.0) && CheckClose((double)result[N-1], 3.0);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 36: aclnnAdd  FLOAT32 1D broadcast {8} + {1}
 *  Covers: broadcast shape with scalar-like shape
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_Broadcast1D(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 broadcast {8}+{1}";
    std::vector<int64_t> selfShape  = {8};
    std::vector<int64_t> otherShape = {1};
    std::vector<int64_t> outShape   = {8};
    std::vector<float> selfData  = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    std::vector<float> otherData = {100.f};
    std::vector<float> outData(8, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  selfShape,  &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   outShape,   &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 100.0;
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 37: aclnnAdds  FLOAT16 + float scalar (PromoteTypeScalar fp16 path)
 *  Covers: PromoteTypeScalar IsFloatingType(self) → return self dtype branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Float16_Scalar(aclrtStream stream)
{
    const std::string name = "aclnnAdds FLOAT16 + float scalar";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> selfData = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<uint16_t> outData(8, 0);
    float otherVal = 2.0f;
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 38: aclnnAdd  UINT8 alpha=2 (AxpyV2 path for UINT8)
 *  Covers: IsSupportAxpyV2 UINT8 on regbase
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Uint8_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd UINT8 alpha=2 (AxpyV2 uint8)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint8_t> selfData  = {10,20,30,40,50,60,70,80};
    std::vector<uint8_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<uint8_t> outData(8, 0);
    int32_t alphaVal = 2;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_UINT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_UINT8, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_UINT8, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 39: aclnnAdd  BOOL alpha=2 (AxpyV2 path for BOOL on regbase)
 *  Covers: IsSupportAxpyV2 BOOL entry
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Bool_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd BOOL alpha=2 (AxpyV2 bool)";
    std::vector<int64_t> shape = {2, 4};
    std::vector<uint8_t> selfData  = {0,1,1,0,0,1,0,1};
    std::vector<uint8_t> otherData = {1,0,1,1,1,0,1,0};
    std::vector<uint8_t> outData(8, 0);
    int32_t alphaVal = 2;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_BOOL, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_BOOL, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_BOOL, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 40: aclnnAdd  mixed BF16+FLOAT (alpha=1)
 *  Covers: add_tiling_arch35.cpp IsMixedDtype BF16+FLOAT
 *          isAddMixDtypeSupport BF16+FLOAT branch
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_MixedDtype_BF16Float_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd BF16+FLOAT32 mixed alpha=1 (bf16+fp32 mixed kernel)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100}; // bf16
    std::vector<float>    otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    std::vector<float>    outData(8, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_BF16,  &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 41: aclnnAdd  FLOAT+BF16 mixed (self=fp32, other=bf16)
 *  Covers: isAddMixDtypeSupport FLOAT+BF16 branch, isMixedDtype tiling BF16 side
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_MixedDtype_FloatBF16_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32+BF16 mixed alpha=1";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float>    selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
    std::vector<float>    outData(8, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_BF16,  &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 42: aclnnAdd  FLOAT32 3D tensor
 *  Covers: 3D tensor shape handling in CheckShape and tiling
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_3D(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 3D tensor {2,3,4}";
    std::vector<int64_t> shape = {2, 3, 4};
    int64_t N = 24;
    std::vector<float> selfData(N), otherData(N), outData(N, 0.f);
    for (int i = 0; i < N; i++) { selfData[i] = (float)i; otherData[i] = (float)(N-i); }
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(N, 0.f);
    aclrtMemcpy(result.data(), N*sizeof(float), outDev, N*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < N && ok; i++) {
        double expected = (double)selfData[i] + (double)otherData[i];
        ok &= CheckClose((double)result[i], expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 43: aclnnAdd  FLOAT32 zero values
 *  Covers: zero tensor computation (boundary value)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_ZeroValues(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 zero values boundary";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData(8, 0.f);
    std::vector<float> otherData(8, 0.f);
    std::vector<float> outData(8, 1.f); // pre-fill with non-zero
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(8, 1.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        ok &= CheckClose((double)result[i], 0.0);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 44: aclnnAdds  BF16 + scalar (PromoteTypeScalar bf16 path)
 *  Covers: isKeepB16 logic in PromoteTypeScalar on regbase
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_BF16_Scalar(aclrtStream stream)
{
    const std::string name = "aclnnAdds BF16 + scalar (bf16 keep path)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> selfData = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> outData(8, 0);
    float otherVal = 1.0f;
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_BF16, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_BF16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 45: aclnnAddV3  BF16 scalar + alpha*tensor (V3 with BF16)
 *  Covers: V3 PromoteTypeScalar → IsFloatingType(other) → DT_FLOAT path
 * ═══════════════════════════════════════════════════════════ */
static void Test_AddV3_BF16_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAddV3 BF16 alpha=1 (bf16 V3)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> otherData = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> outData(8, 0);
    float selfVal  = 1.0f;
    float alphaVal = 1.0f;

    void *otherDev=nullptr, *outDev=nullptr;
    aclTensor *other=nullptr, *out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_BF16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 46: aclnnAdd  INT32 large tensor (exercise tiling with many elements)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int32_LargeTensor(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT32 large tensor alpha=1";
    std::vector<int64_t> shape = {256, 256};
    int64_t N = 256*256;
    std::vector<int32_t> selfData(N, 1);
    std::vector<int32_t> otherData(N, 2);
    std::vector<int32_t> outData(N, 0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT32, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<int32_t> result(N, 0);
    aclrtMemcpy(result.data(), N*sizeof(int32_t), outDev, N*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ok &= (result[0] == 3) && (result[N-1] == 3);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 47: aclnnAdds  INT64 + scalar (covers INT64 in AxpyV2 for Adds)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Int64_Scalar_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdds INT64 + scalar alpha=2";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int64_t> selfData = {1,2,3,4,5,6,7,8};
    std::vector<int64_t> outData(8, 0);
    int64_t otherVal = 10;
    int64_t alphaVal = 2;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_INT64);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);

    CreateAclTensor(selfData, shape, &selfDev, ACL_INT64, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_INT64, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    std::vector<int64_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int64_t), outDev, 8*sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int64_t expected = selfData[i] + 2LL * 10LL;
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 48: aclnnInplaceAdd  INT32 broadcast (other is smaller shape)
 *  Covers: CheckInplace broadcast check path
 * ═══════════════════════════════════════════════════════════ */
static void Test_InplaceAdd_Broadcast(aclrtStream stream)
{
    const std::string name = "aclnnInplaceAdd INT32 broadcast {4,4}+={1,4}";
    std::vector<int64_t> selfShape  = {4, 4};
    std::vector<int64_t> otherShape = {1, 4};
    std::vector<int32_t> selfData(16, 1);
    std::vector<int32_t> otherData = {10,20,30,40};
    int32_t alphaVal = 1;

    void *selfDev=nullptr, *otherDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  selfShape,  &selfDev,  ACL_INT32, &self);
    CreateAclTensor(otherData, otherShape, &otherDev, ACL_INT32, &other);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdd);

    std::vector<int32_t> result(16, 0);
    aclrtMemcpy(result.data(), 16*sizeof(int32_t), selfDev, 16*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    // row 0: [11,21,31,41], row 1-3 same pattern
    for (int r = 0; r < 4 && ok; r++)
        for (int c = 0; c < 4 && ok; c++) {
            int32_t expected = 1 + otherData[c];
            ok &= (result[r*4+c] == expected);
        }

    aclDestroyTensor(self); aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 49: aclnnAdd  alpha=1.0 exact float (IsFloatEqual coverage)
 *  Use alpha=1.0f stored as double to exercise GetCastedFloat
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Float32_AlphaExactOne(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 alpha=1.0 (exact float comparison coverage)";
    std::vector<int64_t> shape = {8, 8};
    int64_t N = 64;
    std::vector<float> selfData(N, 3.14f);
    std::vector<float> otherData(N, 2.71f);
    std::vector<float> outData(N, 0.f);
    double alphaVal = 1.0; // double typed alpha=1

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<float> result(N, 0.f);
    aclrtMemcpy(result.data(), N*sizeof(float), outDev, N*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    ok &= CheckClose((double)result[0], (double)(3.14f + 2.71f), 1e-2, 1e-2);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 50: aclnnAdds  INT8 + scalar (AxpyV2 INT8 for Adds)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Int8_Scalar_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdds INT8 + scalar alpha=2 (AxpyV2 int8 adds)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int8_t> selfData = {1,2,3,4,5,6,7,8};
    std::vector<int8_t> outData(8, 0);
    int8_t  otherVal = 5;
    int32_t alphaVal = 2;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_INT8);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData, shape, &selfDev, ACL_INT8, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_INT8, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}


/* ═══════════════════════════════════════════════════════════
 *  TEST 51: aclnnAdd  FLOAT32 empty tensor {4,0}
 *  Covers: aclnn_add.cpp  if(self->IsEmpty()||other->IsEmpty()) early return
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_EmptyTensor(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT32 empty {4,0} (IsEmpty early return)";
    std::vector<int64_t> shape = {4, 0};
    float alphaVal = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    // strides[0]=0, strides[1]=1 for empty dimension
    std::vector<int64_t> strides = {0, 1};
    auto mkEmpty = [&]() {
        return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                               strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                               shape.data(), shape.size(), nullptr);
    };
    aclTensor* self  = mkEmpty();
    aclTensor* other = mkEmpty();
    aclTensor* out   = mkEmpty();

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    // Must succeed with wsSize==0 and skip computation
    bool ok = (ret == ACL_SUCCESS && wsSize == 0);
    if (ok && exec) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 52: aclnnAdds  FLOAT32 empty tensor {0,4}
 *  Covers: aclnn_add.cpp  if(self->IsEmpty()) early return in aclnnAddsGetWorkspaceSize
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_EmptyTensor(aclrtStream stream)
{
    const std::string name = "aclnnAdds FLOAT32 empty {0,4} (IsEmpty early return)";
    std::vector<int64_t> shape = {0, 4};
    float otherVal = 1.0f, alphaVal = 1.0f;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    std::vector<int64_t> strides = {4, 1};
    auto mkEmpty = [&]() {
        return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                               strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                               shape.data(), shape.size(), nullptr);
    };
    aclTensor* self = mkEmpty();
    aclTensor* out  = mkEmpty();

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS && wsSize == 0);
    if (ok && exec) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 53: aclnnAddV3  empty tensor {2,0}
 *  Covers: aclnn_add_v3.cpp  if(other->IsEmpty()) early return
 * ═══════════════════════════════════════════════════════════ */
static void Test_AddV3_EmptyTensor(aclrtStream stream)
{
    const std::string name = "aclnnAddV3 FLOAT32 empty {2,0} (IsEmpty early return)";
    std::vector<int64_t> shape = {2, 0};
    float selfVal = 1.0f, alphaVal = 1.0f;
    aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    std::vector<int64_t> strides = {0, 1};
    auto mkEmpty = [&]() {
        return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                               strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                               shape.data(), shape.size(), nullptr);
    };
    aclTensor* other = mkEmpty();
    aclTensor* out   = mkEmpty();

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS && wsSize == 0);
    if (ok && exec) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 54: aclnnAdd  DOUBLE, alpha=1  ← 最关键！
 *  Covers: add.cpp AddAiCpu (DOUBLE不在AICore支持列表)
 *          aclnn_add.cpp IsEqualToOne → DT_DOUBLE分支 (alpha->ToDouble())
 *          add.cpp GetAiCoreDtypeSupportListBySocVersion → IsAiCoreSupport→false
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Double_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd DOUBLE alpha=1 (AiCpu path + ToDouble branch)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<double> selfData  = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    std::vector<double> otherData = {10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0};
    std::vector<double> outData(8, 0.0);
    double alphaVal = 1.0;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_DOUBLE, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_DOUBLE, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_DOUBLE, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<double> result(8, 0.0);
    aclrtMemcpy(result.data(), 8*sizeof(double), outDev, 8*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++)
        ok &= (std::abs(result[i] - (selfData[i] + otherData[i])) < 1e-9);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 55: aclnnAdd  DOUBLE, alpha=2  ← 覆盖 else(Mul+Add) 分支！
 *  Covers: aclnn_add.cpp  else branch (DOUBLE不在任何Axpy列表)
 *          add.cpp AddAiCpu (Mul结果也走AiCpu)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Double_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd DOUBLE alpha=2 (else Mul+Add branch, AiCpu)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<double> selfData  = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    std::vector<double> otherData = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    std::vector<double> outData(8, 0.0);
    double alphaVal = 2.0;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_DOUBLE, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_DOUBLE, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_DOUBLE, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<double> result(8, 0.0);
    aclrtMemcpy(result.data(), 8*sizeof(double), outDev, 8*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = selfData[i] + 2.0 * otherData[i];
        ok &= (std::abs(result[i] - expected) < 1e-9);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 56: aclnnAdd  INT16, alpha=1
 *  Covers: add.cpp AddAiCpu (INT16不在AICore支持列表)
 *          aclnn_add.cpp IsAiCoreSupport→false分支
 *          add.cpp GetAiCoreDtypeSupportListBySocVersion default分支
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int16_Alpha1(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT16 alpha=1 (AiCpu path)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int16_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<int16_t> otherData = {10,10,10,10,10,10,10,10};
    std::vector<int16_t> outData(8, 0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<int16_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int16_t), outDev, 8*sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int16_t expected = selfData[i] + otherData[i];
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 57: aclnnAdd  INT16, alpha=3 (else Mul+Add for INT16)
 *  Covers: aclnn_add.cpp else branch (INT16不在Axpy/AxpyV2列表)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Int16_Alpha3(aclrtStream stream)
{
    const std::string name = "aclnnAdd INT16 alpha=3 (else Mul+Add, AiCpu)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int16_t> selfData  = {1,2,3,4,5,6,7,8};
    std::vector<int16_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<int16_t> outData(8, 0);
    float alphaVal = 3.0f;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT16, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_INT16, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<int16_t> result(8, 0);
    aclrtMemcpy(result.data(), 8*sizeof(int16_t), outDev, 8*sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        int16_t expected = selfData[i] + 3 * otherData[i];
        ok &= (result[i] == expected);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 58: aclnnAdds  FLOAT16 + 大标量（FP16溢出）→ isKeepB16=false
 *  Covers: aclnn_add.cpp PromoteTypeScalar RegBase分支
 *          isKeepB16 = false → promoteType提升为DT_FLOAT
 *          GetCastedFloat DT_FLOAT16分支 (scalar->ToFp16())
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Float16_Overflow_Scalar(aclrtStream stream)
{
    const std::string name = "aclnnAdds FLOAT16 + large scalar (isKeepB16=false → promote FLOAT)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> selfData = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<float>    outData(8, 0.f); // out is FLOAT since promoted
    // 70000.0 超过 FP16 最大值 65504 → ToFp16() = inf ≠ 70000 → isKeepB16=false
    float otherVal = 70000.0f;
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT,   &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 59: aclnnAdds  BF16 + 大标量（BF16溢出）→ isKeepB16=false
 *  Covers: GetCastedFloat DT_BF16分支 (scalar->ToBf16())
 *          isKeepB16=false for BF16 path → DT_FLOAT promote
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_BF16_Overflow_Scalar(aclrtStream stream)
{
    const std::string name = "aclnnAdds BF16 + large scalar (isKeepB16=false BF16 path)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> selfData = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<float>    outData(8, 0.f);
    // BF16最大有限值约3.39e38，使用超精度而非超范围：选一个在BF16精度下会舍入的值
    // BF16有8位尾数，1/512精度。选 0.001f → ToBf16()后 = 0.0009765625f ≠ 0.001f
    float otherVal = 0.001f;
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_BF16,  &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 60: aclnnAdd  混合 FP16+FP32, alpha=2（isMixDataType但alpha!=1）
 *  Covers: aclnn_add.cpp  isMixDataType=true 但 alpha!=1 → 走 else 分支
 *          promoteType=FLOAT, IsSupportAxpy(FLOAT)=true → Axpy路径
 *          (区别于 T16/T17 只测了 alpha=1 的混合路径)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_MixedDtype_F16F32_Alpha2(aclrtStream stream)
{
    const std::string name = "aclnnAdd FLOAT16+FLOAT32 mixed alpha=2 (fallthrough to Axpy)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<float>    otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    std::vector<float>    outData(8, 0.f);
    float alphaVal = 2.0f; // ← alpha!=1, isMixDataType 条件不满足，走 else 分支

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT,   &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT,   &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 61: aclnnAdds  INT32 self + FLOAT 标量
 *  Covers: aclnn_add.cpp PromoteTypeScalar RegBase分支
 *          CombineCategoriesWithComplex: IsFloatingType(lower=FLOAT) → PromoteType
 *          → promoteType=FLOAT (INT32自动提升为FLOAT)
 * ═══════════════════════════════════════════════════════════ */
static void Test_Adds_Int32_FloatScalar(aclrtStream stream)
{
    const std::string name = "aclnnAdds INT32 self + FLOAT scalar (CombineCategories FloatLower)";
    std::vector<int64_t> shape = {4, 2};
    std::vector<int32_t> selfData = {1,2,3,4,5,6,7,8};
    std::vector<float>   outData(8, 0.f); // out is FLOAT since promoteType promoted
    float otherVal = 0.5f;
    float alphaVal = 1.0f;

    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);
    CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

    std::vector<float> result(8, 0.f);
    aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 8 && ok; i++) {
        double expected = (double)selfData[i] + 0.5;
        ok &= CheckClose((double)result[i], expected, 1e-3, 1e-3);
    }

    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ═══════════════════════════════════════════════════════════
 *  TEST 62: aclnnAdd  DOUBLE, 1D 大张量
 *  Covers: add.cpp AddAiCpu with larger shape (多块tiling via AiCpu)
 *          aclnn_add.cpp IsEqualToOne DT_DOUBLE alpha=1.0 again on larger input
 * ═══════════════════════════════════════════════════════════ */
static void Test_Add_Double_LargeTensor(aclrtStream stream)
{
    const std::string name = "aclnnAdd DOUBLE large {512,512} (AiCpu large)";
    std::vector<int64_t> shape = {512, 512};
    int64_t N = 512*512;
    std::vector<double> selfData(N, 1.0), otherData(N, 2.0), outData(N, 0.0);
    double alphaVal = 1.0;

    void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
    aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);

    CreateAclTensor(selfData,  shape, &selfDev,  ACL_DOUBLE, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_DOUBLE, &other);
    CreateAclTensor(outData,   shape, &outDev,   ACL_DOUBLE, &out);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    bool ok = (ret == ACL_SUCCESS);
    if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

    std::vector<double> result(N, 0.0);
    aclrtMemcpy(result.data(), N*sizeof(double), outDev, N*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
    ok &= (std::abs(result[0] - 3.0) < 1e-9) && (std::abs(result[N-1] - 3.0) < 1e-9);

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    PrintResult(name, ok);
}

/* ══════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════ */
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = InitACL(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("\n========== Add Operator Comprehensive Tests ==========\n\n");

    /* ── aclnnAdd: dtype × alpha branches ── */
    Test_Add_Float32_Alpha1(stream);         // T01: FLOAT32, alpha=1, same dtype direct Add
    Test_Add_Float32_Alpha2(stream);         // T02: FLOAT32, alpha=2, Axpy path
    Test_Add_Float16_Alpha1(stream);         // T03: FLOAT16, alpha=1, fp16 tiling
    Test_Add_Float16_Alpha2(stream);         // T04: FLOAT16, alpha=2, Axpy fp16
    Test_Add_Int32_Alpha1(stream);           // T05: INT32,   alpha=1, int32 tiling
    Test_Add_Int32_Alpha3(stream);           // T06: INT32,   alpha=3, Axpy int32
    Test_Add_Int8_Alpha1(stream);            // T07: INT8,    alpha=1, int8 tiling
    Test_Add_Int8_Alpha2(stream);            // T08: INT8,    alpha=2, AxpyV2 int8
    Test_Add_Uint8_Alpha1(stream);           // T09: UINT8,   alpha=1, uint8 tiling
    Test_Add_Int64_Alpha1(stream);           // T10: INT64,   alpha=1, int64 tiling
    Test_Add_Int64_Alpha2(stream);           // T11: INT64,   alpha=2, AxpyV2 int64
    Test_Add_Bool_Alpha1(stream);            // T12: BOOL,    alpha=1, bool tiling
    Test_Add_Float32_Broadcast(stream);      // T13: FLOAT32, broadcast {4,1}+{1,4}
    Test_Add_Float32_Alpha0(stream);         // T14: FLOAT32, alpha=0
    Test_Add_Float32_AlphaNeg(stream);       // T15: FLOAT32, alpha=-1.5 negative

    /* ── Mixed dtype ── */
    Test_Add_MixedDtype_F16F32_Alpha1(stream);    // T16: fp16+fp32 mixed, alpha=1
    Test_Add_MixedDtype_F32F16_Alpha1(stream);    // T17: fp32+fp16 mixed, alpha=1
    Test_Add_BF16_Alpha1(stream);                 // T18: BF16, alpha=1
    Test_Add_BF16_Alpha2(stream);                 // T19: BF16, alpha=2 Axpy
    Test_Add_MixedDtype_BF16Float_Alpha1(stream); // T40: bf16+fp32 mixed
    Test_Add_MixedDtype_FloatBF16_Alpha1(stream); // T41: fp32+bf16 mixed

    /* ── Bool/UINT8 extra alpha ── */
    Test_Add_Uint8_Alpha2(stream);           // T38: UINT8 alpha=2
    Test_Add_Bool_Alpha2(stream);            // T39: BOOL  alpha=2

    /* ── aclnnAdds ── */
    Test_Adds_Float32_Alpha1(stream);        // T20: FLOAT32 + scalar, alpha=1
    Test_Adds_Float32_Alpha2p5(stream);      // T21: FLOAT32 + scalar, alpha=2.5
    Test_Adds_Int32_Scalar_Alpha1(stream);   // T22: INT32 + scalar, alpha=1
    Test_Adds_Bool_BoolScalar_Alpha1(stream);// T23: BOOL + bool scalar (clamp)
    Test_Adds_Float16_Scalar(stream);        // T37: FLOAT16 + scalar
    Test_Adds_BF16_Scalar(stream);           // T44: BF16 + scalar
    Test_Adds_Int64_Scalar_Alpha2(stream);   // T47: INT64 + scalar, alpha=2
    Test_Adds_Int8_Scalar_Alpha2(stream);    // T50: INT8 + scalar, alpha=2

    /* ── aclnnInplaceAdd / InplaceAdds ── */
    Test_InplaceAdd_Float32_Alpha1(stream);  // T24: InplaceAdd FLOAT32, alpha=1
    Test_InplaceAdd_Float32_Alpha2(stream);  // T25: InplaceAdd FLOAT32, alpha=2
    Test_InplaceAdds_Float32(stream);        // T26: InplaceAdds FLOAT32
    Test_InplaceAdds_Int32_Alpha3(stream);   // T27: InplaceAdds INT32, alpha=3
    Test_InplaceAdd_Broadcast(stream);       // T48: InplaceAdd broadcast

    /* ── aclnnAddV3 / InplaceAddV3 ── */
    Test_AddV3_Float32_Alpha1(stream);       // T28: AddV3 FLOAT32, alpha=1
    Test_AddV3_Float32_Alpha2(stream);       // T29: AddV3 FLOAT32, alpha=2 Axpy
    Test_AddV3_Int32_Alpha3(stream);         // T30: AddV3 INT32, alpha=3 Axpy
    Test_AddV3_Int8_AlphaMulAdd(stream);     // T31: AddV3 INT8, alpha=2 Mul+Add
    Test_AddV3_Float16_Alpha2(stream);       // T32: AddV3 FLOAT16, alpha=2
    Test_InplaceAddV3_Float32_Alpha1(stream);// T33: InplaceAddV3 FLOAT32, alpha=1
    Test_InplaceAddV3_Float32_Alpha2(stream);// T34: InplaceAddV3 FLOAT32, alpha=2
    Test_AddV3_BF16_Alpha1(stream);          // T45: AddV3 BF16, alpha=1

    /* ── Shape / dimension stress ── */
    Test_Add_Float32_LargeTensor(stream);    // T35: large {1024x1024}
    Test_Add_Float32_Broadcast1D(stream);    // T36: 1D broadcast {8}+{1}
    Test_Add_Float32_3D(stream);             // T42: 3D {2,3,4}
    Test_Add_Float32_ZeroValues(stream);     // T43: all zeros
    Test_Add_Int32_LargeTensor(stream);      // T46: INT32 large {256x256}
    Test_Add_Float32_AlphaExactOne(stream);  // T49: alpha=1.0 as double

    /* ── Empty tensor paths ── */
    Test_Add_EmptyTensor(stream);            // T51: IsEmpty early return (aclnnAdd)
    Test_Adds_EmptyTensor(stream);           // T52: IsEmpty early return (aclnnAdds)
    Test_AddV3_EmptyTensor(stream);          // T53: IsEmpty early return (aclnnAddV3)

    /* ── DOUBLE → AiCpu + ToDouble分支 + else(Mul+Add) ── */
    Test_Add_Double_Alpha1(stream);          // T54: DOUBLE alpha=1, AddAiCpu
    Test_Add_Double_Alpha2(stream);          // T55: DOUBLE alpha=2, else Mul+Add + AiCpu ★
    Test_Add_Double_LargeTensor(stream);     // T62: DOUBLE large tensor

    /* ── INT16 → AiCpu + else(Mul+Add) ── */
    Test_Add_Int16_Alpha1(stream);           // T56: INT16 alpha=1, AddAiCpu
    Test_Add_Int16_Alpha3(stream);           // T57: INT16 alpha=3, else Mul+Add + AiCpu ★

    /* ── isKeepB16=false (scalar精度溢出) ── */
    Test_Adds_Float16_Overflow_Scalar(stream); // T58: FP16 + 70000 → promote to FLOAT ★
    Test_Adds_BF16_Overflow_Scalar(stream);    // T59: BF16 + 0.001 → promote to FLOAT ★

    /* ── Mixed dtype alpha!=1 fallthrough ── */
    Test_Add_MixedDtype_F16F32_Alpha2(stream); // T60: F16+F32 alpha=2, else分支 ★

    /* ── CombineCategoriesWithComplex INT32+FLOAT scalar ── */
    Test_Adds_Int32_FloatScalar(stream);       // T61: INT32+FLOAT scalar PromoteType
    /* ── Summary ── */
    LOG_PRINT("\n====================================================\n");
    LOG_PRINT("Results: %d PASSED, %d FAILED, %d TOTAL\n",
              g_passCount, g_failCount, g_passCount + g_failCount);
    LOG_PRINT("====================================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (g_failCount > 0) ? 1 : 0;
}









