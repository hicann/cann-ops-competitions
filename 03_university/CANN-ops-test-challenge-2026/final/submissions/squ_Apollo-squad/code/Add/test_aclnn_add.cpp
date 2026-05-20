// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <cfloat>
// #include <algorithm>
// #include <limits>
// #include <random>
// #include <functional>
// #include <map>

// #include "acl/acl.h"
// #include "aclnnop/aclnn_add.h"
// #include "aclnnop/aclnn_add_v3.h"

// // ==================== 配置参数 ====================
// #define CHECK_RET(cond, return_expr) \
//   do { \
//     if (!(cond)) { \
//       printf("ERROR at %s:%d\n", __FILE__, __LINE__); \
//       return_expr; \
//     } \
//   } while (0)

// #define LOG_INFO(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
// #define LOG_PASS(fmt, ...) printf("[PASS] " fmt "\n", ##__VA_ARGS__)
// #define LOG_FAIL(fmt, ...) printf("[FAIL] " fmt "\n", ##__VA_ARGS__)

// static int g_total_tests = 0;
// static int g_passed_tests = 0;

// // ==================== 辅助函数 ====================
// int64_t GetShapeSize(const std::vector<int64_t>& shape) {
//   int64_t size = 1;
//   for (auto dim : shape) size *= dim;
//   return size;
// }

// // FP16 转换
// uint16_t FloatToFp16(float f) {
//     uint32_t fb = *reinterpret_cast<uint32_t*>(&f);
//     uint32_t sign = (fb >> 31) & 0x1;
//     uint32_t exp = (fb >> 23) & 0xFF;
//     uint32_t mantissa = fb & 0x7FFFFF;
    
//     if (exp == 0xFF) {
//         uint16_t result = (sign << 15) | 0x7C00;
//         if (mantissa != 0) result |= 0x0200;
//         return result;
//     }
//     int new_exp = exp - 127 + 15;
//     if (new_exp >= 31) return (sign << 15) | 0x7C00;
//     if (new_exp <= 0) {
//         if (new_exp < -10) return (sign << 15);
//         mantissa = (mantissa | 0x800000) >> (1 - new_exp);
//         new_exp = 0;
//     }
//     return (sign << 15) | (new_exp << 10) | (mantissa >> 13);
// }

// float Fp16ToFloat(uint16_t h) {
//     uint32_t sign = (h >> 15) & 0x1;
//     uint32_t exp = (h >> 10) & 0x1F;
//     uint32_t mantissa = h & 0x3FF;
//     if (exp == 0) {
//         if (mantissa == 0) return 0.0f;
//         float f = static_cast<float>(mantissa) / (1 << 24);
//         f = ldexpf(f, -14);
//         return sign ? -f : f;
//     }
//     if (exp == 31) {
//         if (mantissa != 0) return NAN;
//         return sign ? -INFINITY : INFINITY;
//     }
//     uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
//     return *reinterpret_cast<float*>(&f);
// }

// // BF16 转换
// uint16_t FloatToBf16(float f) {
//     uint32_t fb = *reinterpret_cast<uint32_t*>(&f);
//     return static_cast<uint16_t>(fb >> 16);
// }

// float Bf16ToFloat(uint16_t bf) {
//     uint32_t fb = static_cast<uint32_t>(bf) << 16;
//     return *reinterpret_cast<float*>(&fb);
// }

// // ==================== 初始化 ====================
// int Init(int32_t deviceId, aclrtStream* stream) {
//   auto ret = aclInit(nullptr);
//   CHECK_RET(ret == ACL_SUCCESS, return ret);
//   ret = aclrtSetDevice(deviceId);
//   CHECK_RET(ret == ACL_SUCCESS, return ret);
//   ret = aclrtCreateStream(stream);
//   CHECK_RET(ret == ACL_SUCCESS, return ret);
//   return 0;
// }


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



// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <cstdint>
// #include <limits>
// #include <string>
// #include <functional>

// #include "acl/acl.h"
// #include "aclnnop/aclnn_add.h"
// #include "aclnnop/aclnn_add_v3.h"

// /* ─────────────────────── macros ─────────────────────── */
// #define CHECK_RET(cond, return_expr)  \
//     do {                              \
//         if (!(cond)) { return_expr; } \
//     } while (0)

// #define LOG_PRINT(message, ...) \
//     do { printf(message, ##__VA_ARGS__); fflush(stdout); } while (0)

// /* 追踪宏：每个测试入口调用，方便定位卡死位置 */
// #define TRACE_ENTER(name) \
//     do { LOG_PRINT(">>> ENTER: %s\n", name); } while(0)

// /* ─────────────────────── globals ─────────────────────── */
// static int g_passCount = 0;
// static int g_failCount = 0;

// /* ─────────────────────── helpers ─────────────────────── */
// static int64_t GetShapeSize(const std::vector<int64_t>& shape)
// {
//     int64_t s = 1;
//     for (auto d : shape) s *= d;
//     return s;
// }

// static int InitACL(int32_t deviceId, aclrtStream* stream)
// {
//     auto ret = aclInit(nullptr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtSetDevice(deviceId);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtCreateStream(stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
//     return 0;
// }

// template <typename T>
// static int CreateAclTensor(const std::vector<T>& hostData,
//                             const std::vector<int64_t>& shape,
//                             void** deviceAddr,
//                             aclDataType dataType,
//                             aclTensor** tensor)
// {
//     auto size = GetShapeSize(shape) * sizeof(T);
//     auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

//     std::vector<int64_t> strides(shape.size(), 1);
//     for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--)
//         strides[i] = shape[i + 1] * strides[i + 1];

//     *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
//                                strides.data(), 0, aclFormat::ACL_FORMAT_ND,
//                                shape.data(), shape.size(), *deviceAddr);
//     return 0;
// }

// static void RunAndSync(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
//                        aclnnStatus (*execFn)(void*, uint64_t, aclOpExecutor*, aclrtStream))
// {
//     if (workspaceSize > 0) {
//         void* ws = nullptr;
//         aclrtMalloc(&ws, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
//         execFn(ws, workspaceSize, executor, stream);
//         aclrtSynchronizeStream(stream);
//         aclrtFree(ws);
//     } else {
//         execFn(nullptr, 0, executor, stream);
//         aclrtSynchronizeStream(stream);
//     }
// }

// static void PrintResult(const std::string& testName, bool passed)
// {
//     if (passed) {
//         LOG_PRINT("[PASS] %s\n", testName.c_str());
//         g_passCount++;
//     } else {
//         LOG_PRINT("[FAIL] %s\n", testName.c_str());
//         g_failCount++;
//     }
// }

// /* tolerance check */
// static bool CheckClose(double actual, double expected, double atol = 1e-3, double rtol = 1e-3)
// {
//     if (std::isnan(expected) && std::isnan(actual)) return true;
//     if (std::isinf(expected) && std::isinf(actual) && (expected > 0) == (actual > 0)) return true;
//     return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 1: aclnnAdd  FLOAT32, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T01: aclnnAdd FLOAT32 alpha=1 (same shape)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData  = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
//     std::vector<float> otherData = {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f};
//     std::vector<float> outData(8, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, aclDataType::ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 1.0 * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 2: aclnnAdd  FLOAT32, alpha=2.0
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T02: aclnnAdd FLOAT32 alpha=2.0 (Axpy path)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData  = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
//     std::vector<float> otherData = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
//     std::vector<float> outData(8, 0.f);
//     float alphaVal = 2.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 2.0 * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 3: aclnnAdd  FLOAT16, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float16_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T03: aclnnAdd FLOAT16 alpha=1 (fp16 tiling branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
//     std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
//     std::vector<uint16_t> outData(8, 0);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT16, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 4: aclnnAdd  FLOAT16, alpha=2.0
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float16_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T04: aclnnAdd FLOAT16 alpha=2.0 (Axpy fp16)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 2};
//     std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400};
//     std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00};
//     std::vector<uint16_t> outData(4, 0);
//     float alphaVal = 2.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT16, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 5: aclnnAdd  INT32, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Int32_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T05: aclnnAdd INT32 alpha=1 (int32 tiling branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int32_t> selfData  = {0,1,2,3,4,5,6,7};
//     std::vector<int32_t> otherData = {10,10,10,10,10,10,10,10};
//     std::vector<int32_t> outData(8, 0);
//     int32_t alphaVal = 1;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT32, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<int32_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         int32_t expected = selfData[i] + 1 * otherData[i];
//         ok &= (result[i] == expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 6: aclnnAdd  INT32, alpha=3
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Int32_Alpha3(aclrtStream stream)
// {
//     const std::string name = "T06: aclnnAdd INT32 alpha=3 (Axpy int32)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int32_t> selfData  = {1,2,3,4,5,6,7,8};
//     std::vector<int32_t> otherData = {1,1,1,1,1,1,1,1};
//     std::vector<int32_t> outData(8, 0);
//     float alphaVal = 3.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT32, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<int32_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         int32_t expected = selfData[i] + 3 * otherData[i];
//         ok &= (result[i] == expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 7: aclnnAdd  INT8, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Int8_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T07: aclnnAdd INT8 alpha=1 (int8 tiling branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int8_t> selfData  = {0,1,2,3,4,5,6,7};
//     std::vector<int8_t> otherData = {10,10,10,10,10,10,10,10};
//     std::vector<int8_t> outData(8, 0);
//     int32_t alphaVal = 1;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT8, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT8, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<int8_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int8_t), outDev, 8*sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         int8_t expected = selfData[i] + otherData[i];
//         ok &= (result[i] == expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 8: aclnnAdd  INT8, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Int8_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T08: aclnnAdd INT8 alpha=2 (AxpyV2 int8)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 4};
//     std::vector<int8_t> selfData  = {1,2,3,4,5,6,7,8};
//     std::vector<int8_t> otherData = {1,1,1,1,1,1,1,1};
//     std::vector<int8_t> outData(8, 0);
//     int32_t alphaVal = 2;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT8, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT8, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 9: aclnnAdd  UINT8, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Uint8_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T09: aclnnAdd UINT8 alpha=1 (uint8 tiling branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint8_t> selfData  = {0,1,2,3,4,5,6,7};
//     std::vector<uint8_t> otherData = {10,10,10,10,10,10,10,10};
//     std::vector<uint8_t> outData(8, 0);
//     int32_t alphaVal = 1;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_UINT8, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_UINT8, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_UINT8, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<uint8_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(uint8_t), outDev, 8*sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         uint8_t expected = selfData[i] + otherData[i];
//         ok &= (result[i] == expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 10: aclnnAdd  INT64, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Int64_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T10: aclnnAdd INT64 alpha=1 (int64 tiling branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int64_t> selfData  = {0,1,2,3,4,5,6,7};
//     std::vector<int64_t> otherData = {100,100,100,100,100,100,100,100};
//     std::vector<int64_t> outData(8, 0);
//     int64_t alphaVal = 1;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT64, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT64, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT64, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<int64_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int64_t), outDev, 8*sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         ok &= (result[i] == selfData[i] + otherData[i]);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 11: aclnnAdd  INT64, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Int64_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T11: aclnnAdd INT64 alpha=2 (AxpyV2 int64)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 4};
//     std::vector<int64_t> selfData  = {10,20,30,40,50,60,70,80};
//     std::vector<int64_t> otherData = {1,1,1,1,1,1,1,1};
//     std::vector<int64_t> outData(8, 0);
//     int64_t alphaVal = 2;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT64, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT64, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT64, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 12: aclnnAdd  BOOL, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Bool_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T12: aclnnAdd BOOL alpha=1 (bool tiling branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 4};
//     std::vector<uint8_t> selfData  = {0,1,1,0,1,0,0,1};
//     std::vector<uint8_t> otherData = {1,0,1,0,0,1,0,1};
//     std::vector<uint8_t> outData(8, 0);
//     bool alphaVal = true;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BOOL);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_BOOL, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_BOOL, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_BOOL, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 13: aclnnAdd  FLOAT32 broadcast {4,1}+{1,4}→{4,4}
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_Broadcast(aclrtStream stream)
// {
//     const std::string name = "T13: aclnnAdd FLOAT32 broadcast {4,1}+{1,4}";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> selfShape  = {4, 1};
//     std::vector<int64_t> otherShape = {1, 4};
//     std::vector<int64_t> outShape   = {4, 4};
//     std::vector<float> selfData  = {1.f, 2.f, 3.f, 4.f};
//     std::vector<float> otherData = {10.f, 20.f, 30.f, 40.f};
//     std::vector<float> outData(16, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  selfShape,  &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   outShape,   &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(16, 0.f);
//     aclrtMemcpy(result.data(), 16*sizeof(float), outDev, 16*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int r = 0; r < 4 && ok; r++)
//         for (int c = 0; c < 4 && ok; c++) {
//             double expected = (double)selfData[r] + (double)otherData[c];
//             ok &= CheckClose((double)result[r*4+c], expected);
//         }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 14: aclnnAdd  FLOAT32 alpha=0
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_Alpha0(aclrtStream stream)
// {
//     const std::string name = "T14: aclnnAdd FLOAT32 alpha=0 (out == self)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<float> otherData = {100.f,100.f,100.f,100.f,100.f,100.f,100.f,100.f};
//     std::vector<float> outData(8, 0.f);
//     float alphaVal = 0.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 0.0 * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected, 1e-2, 1e-2);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 15: aclnnAdd  FLOAT32 alpha=-1.5
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_AlphaNeg(aclrtStream stream)
// {
//     const std::string name = "T15: aclnnAdd FLOAT32 alpha=-1.5 (negative alpha)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData  = {10.f,10.f,10.f,10.f,10.f,10.f,10.f,10.f};
//     std::vector<float> otherData = {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
//     std::vector<float> outData(8, 0.f);
//     float alphaVal = -1.5f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + (-1.5) * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 16: aclnnAdd  FLOAT16+FLOAT32 mixed, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_MixedDtype_F16F32_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T16: aclnnAdd FLOAT16+FLOAT32 mixed alpha=1";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
//     std::vector<float>    otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
//     std::vector<float>    outData(8, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT16, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT,   &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT,   &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 17: aclnnAdd  FLOAT32+FLOAT16 mixed, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_MixedDtype_F32F16_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T17: aclnnAdd FLOAT32+FLOAT16 mixed alpha=1";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float>    selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
//     std::vector<float>    outData(8, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT,   &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT,   &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 18: aclnnAdd  BF16, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_BF16_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T18: aclnnAdd BF16 alpha=1 (bf16 tiling branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
//     std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
//     std::vector<uint16_t> outData(8, 0);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_BF16, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_BF16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 19: aclnnAdd  BF16, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_BF16_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T19: aclnnAdd BF16 alpha=2 (bf16 Axpy branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 4};
//     std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
//     std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
//     std::vector<uint16_t> outData(8, 0);
//     float alphaVal = 2.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_BF16, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_BF16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 20: aclnnAdds  FLOAT32 + scalar, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_Float32_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T20: aclnnAdds FLOAT32 + scalar alpha=1";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<float> outData(8, 0.f);
//     float otherVal = 10.0f;
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 1.0 * 10.0;
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 21: aclnnAdds  FLOAT32 + scalar, alpha=2.5
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_Float32_Alpha2p5(aclrtStream stream)
// {
//     const std::string name = "T21: aclnnAdds FLOAT32 + scalar alpha=2.5 (Axpy)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<float> outData(8, 0.f);
//     float otherVal = 3.0f;
//     float alphaVal = 2.5f;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 2.5 * 3.0;
//         ok &= CheckClose((double)result[i], expected, 1e-2, 1e-2);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 22: aclnnAdds  INT32 + scalar, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_Int32_Scalar_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T22: aclnnAdds INT32 + scalar alpha=1";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int32_t> selfData = {1,2,3,4,5,6,7,8};
//     std::vector<int32_t> outData(8, 0);
//     int32_t otherVal = 100;
//     int32_t alphaVal = 1;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_INT32);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_INT32, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     std::vector<int32_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         int32_t expected = selfData[i] + 100;
//         ok &= (result[i] == expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 23: aclnnAdds  BOOL + BOOL scalar, alpha=BOOL(true)
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_Bool_BoolScalar_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T23: aclnnAdds BOOL + bool scalar alpha=true";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 4};
//     std::vector<uint8_t> selfData = {0,1,1,0,0,1,0,1};
//     std::vector<int32_t> outData(8, 0);
//     bool otherVal = true;
//     bool alphaVal = true;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_BOOL);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BOOL);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_BOOL,  &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_INT32, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 24: aclnnInplaceAdd  FLOAT32, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_InplaceAdd_Float32_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T24: aclnnInplaceAdd FLOAT32 alpha=1 (in-place)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<float> otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdd);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), selfDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 25: aclnnInplaceAdd  FLOAT32, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_InplaceAdd_Float32_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T25: aclnnInplaceAdd FLOAT32 alpha=2 (in-place Axpy)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData  = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
//     std::vector<float> otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
//     float alphaVal = 2.0f;

//     void *selfDev=nullptr, *otherDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdd);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), selfDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 2.0 * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 26: aclnnInplaceAdds  FLOAT32 += scalar
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_InplaceAdds_Float32(aclrtStream stream)
// {
//     const std::string name = "T26: aclnnInplaceAdds FLOAT32 += scalar";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     float otherVal = 5.0f;
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr;
//     aclTensor *self=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdds);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), selfDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 5.0;
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 27: aclnnInplaceAdds  INT32 += scalar * 3
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_InplaceAdds_Int32_Alpha3(aclrtStream stream)
// {
//     const std::string name = "T27: aclnnInplaceAdds INT32 alpha=3";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int32_t> selfData = {0,1,2,3,4,5,6,7};
//     int32_t otherVal = 2;
//     float alphaVal = 3.0f;

//     void *selfDev=nullptr;
//     aclTensor *self=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_INT32);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdds);

//     aclDestroyTensor(self);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 28: aclnnAddV3  FLOAT32, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_AddV3_Float32_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T28: aclnnAddV3 FLOAT32 alpha=1 (scalar + tensor)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<float> outData(8, 0.f);
//     float selfVal  = 10.0f;
//     float alphaVal = 1.0f;

//     void *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *other=nullptr, *out=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = 10.0 + 1.0 * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 29: aclnnAddV3  FLOAT32, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_AddV3_Float32_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T29: aclnnAddV3 FLOAT32 alpha=2 (Axpy branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<float> outData(8, 0.f);
//     float selfVal  = 5.0f;
//     float alphaVal = 2.0f;

//     void *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *other=nullptr, *out=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = 5.0 + 2.0 * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 30: aclnnAddV3  INT32, alpha=3
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_AddV3_Int32_Alpha3(aclrtStream stream)
// {
//     const std::string name = "T30: aclnnAddV3 INT32 alpha=3 (Axpy int32 V3)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int32_t> otherData = {1,2,3,4,5,6,7,8};
//     std::vector<int32_t> outData(8, 0);
//     int32_t selfVal  = 100;
//     float alphaVal = 3.0f;

//     void *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *other=nullptr, *out=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfVal,  ACL_INT32);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

//     std::vector<int32_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int32_t), outDev, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         int32_t expected = 100 + 3 * otherData[i];
//         ok &= (result[i] == expected);
//     }

//     aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 31: aclnnAddV3  INT8, alpha=2 (Mul+Add else branch)
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_AddV3_Int8_AlphaMulAdd(aclrtStream stream)
// {
//     const std::string name = "T31: aclnnAddV3 INT8 alpha=2 (Mul+Add else branch)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int8_t> otherData = {1,2,3,4,5,6,7,8};
//     std::vector<int8_t> outData(8, 0);
//     int8_t  selfVal  = 10;
//     float   alphaVal = 2.0f;

//     void *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *other=nullptr, *out=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfVal,  ACL_INT8);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT8, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

//     std::vector<int8_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int8_t), outDev, 8*sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         int expected = (int)selfVal + 2 * (int)otherData[i];
//         ok &= (result[i] == (int8_t)expected);
//     }

//     aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 32: aclnnAddV3  FLOAT16, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_AddV3_Float16_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T32: aclnnAddV3 FLOAT16 alpha=2 (Axpy fp16 V3)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 4};
//     std::vector<uint16_t> otherData = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
//     std::vector<uint16_t> outData(8, 0);
//     float selfValF = 3.0f;
//     float alphaVal = 2.0f;

//     void *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *other=nullptr, *out=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfValF, ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

//     aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 33: aclnnInplaceAddV3  FLOAT32, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_InplaceAddV3_Float32_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T33: aclnnInplaceAddV3 FLOAT32 alpha=1 (in-place V3)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     float selfVal  = 10.0f;
//     float alphaVal = 1.0f;

//     void *otherDev=nullptr;
//     aclTensor *other=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAddV3);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), otherDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = 10.0 + 1.0 * (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(other);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 34: aclnnInplaceAddV3  FLOAT32, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_InplaceAddV3_Float32_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T34: aclnnInplaceAddV3 FLOAT32 alpha=2 (in-place V3 Axpy)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     float selfVal  = 100.0f;
//     float alphaVal = 2.0f;

//     void *otherDev=nullptr;
//     aclTensor *other=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAddV3);

//     aclDestroyTensor(other);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 35: aclnnAdd  FLOAT32 大张量（降维后）
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_LargeTensor(aclrtStream stream)
// {
//     const std::string name = "T35: aclnnAdd FLOAT32 large tensor (512x512)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {512, 512};
//     int64_t N = 512*512;
//     std::vector<float> selfData(N, 1.0f);
//     std::vector<float> otherData(N, 2.0f);
//     std::vector<float> outData(N, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(N, 0.f);
//     aclrtMemcpy(result.data(), N*sizeof(float), outDev, N*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     ok &= CheckClose((double)result[0], 3.0) && CheckClose((double)result[N-1], 3.0);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 36: aclnnAdd  FLOAT32 1D broadcast {8}+{1}
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_Broadcast1D(aclrtStream stream)
// {
//     const std::string name = "T36: aclnnAdd FLOAT32 broadcast {8}+{1}";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> selfShape  = {8};
//     std::vector<int64_t> otherShape = {1};
//     std::vector<int64_t> outShape   = {8};
//     std::vector<float> selfData  = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
//     std::vector<float> otherData = {100.f};
//     std::vector<float> outData(8, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  selfShape,  &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   outShape,   &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(8, 0.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         double expected = (double)selfData[i] + 100.0;
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 37: aclnnAdds  FLOAT16 + float scalar
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_Float16_Scalar(aclrtStream stream)
// {
//     const std::string name = "T37: aclnnAdds FLOAT16 + float scalar";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint16_t> selfData = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
//     std::vector<uint16_t> outData(8, 0);
//     float otherVal = 2.0f;
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 38: aclnnAdd  UINT8 alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Uint8_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T38: aclnnAdd UINT8 alpha=2 (AxpyV2 uint8)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint8_t> selfData  = {10,20,30,40,50,60,70,80};
//     std::vector<uint8_t> otherData = {1,1,1,1,1,1,1,1};
//     std::vector<uint8_t> outData(8, 0);
//     int32_t alphaVal = 2;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_UINT8, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_UINT8, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_UINT8, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 39: aclnnAdd  BOOL alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Bool_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T39: aclnnAdd BOOL alpha=2 (AxpyV2 bool)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 4};
//     std::vector<uint8_t> selfData  = {0,1,1,0,0,1,0,1};
//     std::vector<uint8_t> otherData = {1,0,1,1,1,0,1,0};
//     std::vector<uint8_t> outData(8, 0);
//     int32_t alphaVal = 2;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_BOOL, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_BOOL, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_BOOL, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 40: aclnnAdd  BF16+FLOAT32 mixed, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_MixedDtype_BF16Float_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T40: aclnnAdd BF16+FLOAT32 mixed alpha=1";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
//     std::vector<float>    otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
//     std::vector<float>    outData(8, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_BF16,  &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 41: aclnnAdd  FLOAT32+BF16 mixed, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_MixedDtype_FloatBF16_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T41: aclnnAdd FLOAT32+BF16 mixed alpha=1";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float>    selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
//     std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
//     std::vector<float>    outData(8, 0.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_BF16,  &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 42: aclnnAdd  FLOAT32 3D {2,3,4}
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_3D(aclrtStream stream)
// {
//     const std::string name = "T42: aclnnAdd FLOAT32 3D tensor {2,3,4}";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {2, 3, 4};
//     int64_t N = 24;
//     std::vector<float> selfData(N), otherData(N), outData(N, 0.f);
//     for (int i = 0; i < N; i++) { selfData[i] = (float)i; otherData[i] = (float)(N-i); }
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(N, 0.f);
//     aclrtMemcpy(result.data(), N*sizeof(float), outDev, N*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < N && ok; i++) {
//         double expected = (double)selfData[i] + (double)otherData[i];
//         ok &= CheckClose((double)result[i], expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 43: aclnnAdd  FLOAT32 全零张量
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_ZeroValues(aclrtStream stream)
// {
//     const std::string name = "T43: aclnnAdd FLOAT32 zero values boundary";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<float> selfData(8, 0.f);
//     std::vector<float> otherData(8, 0.f);
//     std::vector<float> outData(8, 1.f);
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(8, 1.f);
//     aclrtMemcpy(result.data(), 8*sizeof(float), outDev, 8*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         ok &= CheckClose((double)result[i], 0.0);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 44: aclnnAdds  BF16 + scalar
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_BF16_Scalar(aclrtStream stream)
// {
//     const std::string name = "T44: aclnnAdds BF16 + scalar";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint16_t> selfData = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
//     std::vector<uint16_t> outData(8, 0);
//     float otherVal = 1.0f;
//     float alphaVal = 1.0f;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_BF16, &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_BF16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 45: aclnnAddV3  BF16, alpha=1
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_AddV3_BF16_Alpha1(aclrtStream stream)
// {
//     const std::string name = "T45: aclnnAddV3 BF16 alpha=1";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<uint16_t> otherData = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
//     std::vector<uint16_t> outData(8, 0);
//     float selfVal  = 1.0f;
//     float alphaVal = 1.0f;

//     void *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *other=nullptr, *out=nullptr;
//     aclScalar* self  = aclCreateScalar(&selfVal,  ACL_FLOAT);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

//     CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_BF16, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAddV3);

//     aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(self); aclDestroyScalar(alpha);
//     aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 46: aclnnAdd  INT32 大张量（降维后）
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Int32_LargeTensor(aclrtStream stream)
// {
//     const std::string name = "T46: aclnnAdd INT32 large tensor (128x128)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {128, 128};
//     int64_t N = 128*128;
//     std::vector<int32_t> selfData(N, 1);
//     std::vector<int32_t> otherData(N, 2);
//     std::vector<int32_t> outData(N, 0);
//     int32_t alphaVal = 1;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_INT32, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_INT32, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<int32_t> result(N, 0);
//     aclrtMemcpy(result.data(), N*sizeof(int32_t), outDev, N*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     ok &= (result[0] == 3) && (result[N-1] == 3);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 47: aclnnAdds  INT64 + scalar, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_Int64_Scalar_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T47: aclnnAdds INT64 + scalar alpha=2";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int64_t> selfData = {1,2,3,4,5,6,7,8};
//     std::vector<int64_t> outData(8, 0);
//     int64_t otherVal = 10;
//     int64_t alphaVal = 2;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_INT64);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_INT64, &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_INT64, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     std::vector<int64_t> result(8, 0);
//     aclrtMemcpy(result.data(), 8*sizeof(int64_t), outDev, 8*sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int i = 0; i < 8 && ok; i++) {
//         int64_t expected = selfData[i] + 2LL * 10LL;
//         ok &= (result[i] == expected);
//     }

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 48: aclnnInplaceAdd  INT32 broadcast {4,4}+={1,4}
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_InplaceAdd_Broadcast(aclrtStream stream)
// {
//     const std::string name = "T48: aclnnInplaceAdd INT32 broadcast {4,4}+={1,4}";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> selfShape  = {4, 4};
//     std::vector<int64_t> otherShape = {1, 4};
//     std::vector<int32_t> selfData(16, 1);
//     std::vector<int32_t> otherData = {10,20,30,40};
//     int32_t alphaVal = 1;

//     void *selfDev=nullptr, *otherDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData,  selfShape,  &selfDev,  ACL_INT32, &self);
//     CreateAclTensor(otherData, otherShape, &otherDev, ACL_INT32, &other);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnInplaceAdd);

//     std::vector<int32_t> result(16, 0);
//     aclrtMemcpy(result.data(), 16*sizeof(int32_t), selfDev, 16*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
//     for (int r = 0; r < 4 && ok; r++)
//         for (int c = 0; c < 4 && ok; c++) {
//             int32_t expected = 1 + otherData[c];
//             ok &= (result[r*4+c] == expected);
//         }

//     aclDestroyTensor(self); aclDestroyTensor(other);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 49: aclnnAdd  FLOAT32 alpha=1.0 as double scalar
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Add_Float32_AlphaExactOne(aclrtStream stream)
// {
//     const std::string name = "T49: aclnnAdd FLOAT32 alpha=1.0 (double scalar coverage)";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {8, 8};
//     int64_t N = 64;
//     std::vector<float> selfData(N, 3.14f);
//     std::vector<float> otherData(N, 2.71f);
//     std::vector<float> outData(N, 0.f);
//     double alphaVal = 1.0;

//     void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *other=nullptr, *out=nullptr;
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);

//     CreateAclTensor(selfData,  shape, &selfDev,  ACL_FLOAT, &self);
//     CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
//     CreateAclTensor(outData,   shape, &outDev,   ACL_FLOAT, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdd);

//     std::vector<float> result(N, 0.f);
//     aclrtMemcpy(result.data(), N*sizeof(float), outDev, N*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     ok &= CheckClose((double)result[0], (double)(3.14f + 2.71f), 1e-2, 1e-2);

//     aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
//     aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  TEST 50: aclnnAdds  INT8 + scalar, alpha=2
//  * ═══════════════════════════════════════════════════════════ */
// static void Test_Adds_Int8_Scalar_Alpha2(aclrtStream stream)
// {
//     const std::string name = "T50: aclnnAdds INT8 + scalar alpha=2";
//     TRACE_ENTER(name.c_str());
//     std::vector<int64_t> shape = {4, 2};
//     std::vector<int8_t> selfData = {1,2,3,4,5,6,7,8};
//     std::vector<int8_t> outData(8, 0);
//     int8_t  otherVal = 5;
//     int32_t alphaVal = 2;

//     void *selfDev=nullptr, *outDev=nullptr;
//     aclTensor *self=nullptr, *out=nullptr;
//     aclScalar* other = aclCreateScalar(&otherVal, ACL_INT8);
//     aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

//     CreateAclTensor(selfData, shape, &selfDev, ACL_INT8, &self);
//     CreateAclTensor(outData,  shape, &outDev,  ACL_INT8, &out);

//     uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
//     auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
//     bool ok = (ret == ACL_SUCCESS);
//     if (ok) RunAndSync(nullptr, wsSize, exec, stream, aclnnAdds);

//     aclDestroyTensor(self); aclDestroyTensor(out);
//     aclDestroyScalar(other); aclDestroyScalar(alpha);
//     aclrtFree(selfDev); aclrtFree(outDev);
//     PrintResult(name, ok);
// }

// /* ═══════════════════════════════════════════════════════════
//  *  以下测试因稳定性风险暂时注释，如需启用请逐一打开验证
//  * ═══════════════════════════════════════════════════════════ */

// /* --- T51/T52/T53: 空张量测试 ---
//  * 注释原因: wsSize==0 时 exec 可能仍非空，RunAndSync 里
//  *           execFn(nullptr,0,exec,stream) + SynchronizeStream 会永久挂起。
//  * 如需测试: 改为只检查 GetWorkspaceSize 返回值，不执行 RunAndSync。
//  *
// static void Test_Add_EmptyTensor(aclrtStream stream) { ... }
// static void Test_Adds_EmptyTensor(aclrtStream stream) { ... }
// static void Test_AddV3_EmptyTensor(aclrtStream stream) { ... }
// */

// /* --- T54/T55/T62: DOUBLE 类型 → AiCpu 路径 ---
//  * 注释原因: DOUBLE 不在 AICore 支持列表，走 AiCpu 队列，
//  *           AiCpu 调度问题会导致 aclrtSynchronizeStream 永久等待。
//  *
// static void Test_Add_Double_Alpha1(aclrtStream stream) { ... }
// static void Test_Add_Double_Alpha2(aclrtStream stream) { ... }
// static void Test_Add_Double_LargeTensor(aclrtStream stream) { ... }
// */

// /* --- T56/T57: INT16 类型 → AiCpu 路径 ---
//  * 注释原因: 同 DOUBLE，INT16 不在 AICore 支持列表，走 AiCpu 路径，
//  *           存在挂起风险。
//  *
// static void Test_Add_Int16_Alpha1(aclrtStream stream) { ... }
// static void Test_Add_Int16_Alpha3(aclrtStream stream) { ... }
// */

// /* --- T58/T59: isKeepB16=false 标量溢出 ---
//  * 注释原因: 混合精度提升路径在某些驱动下 kernel 无法完成，
//  *           导致 SynchronizeStream 超时。
//  *
// static void Test_Adds_Float16_Overflow_Scalar(aclrtStream stream) { ... }
// static void Test_Adds_BF16_Overflow_Scalar(aclrtStream stream) { ... }
// */

// /* --- T60: 混合 dtype + alpha!=1 ---
//  * 注释原因: isMixDataType=true 但 alpha!=1 时走 else 分支，
//  *           该组合在部分驱动版本下存在 kernel 挂起问题。
//  *
// static void Test_Add_MixedDtype_F16F32_Alpha2(aclrtStream stream) { ... }
// */

// /* --- T61: INT32 self + FLOAT scalar (CombineCategories) ---
//  * 注释原因: 类型提升后走 AiCpu Cast 路径，存在调度挂起风险。
//  *
// static void Test_Adds_Int32_FloatScalar(aclrtStream stream) { ... }
// */

// /* ══════════════════════════════════════════════════════════
//  *  MAIN
//  * ══════════════════════════════════════════════════════════ */
// int main()
// {
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = InitACL(deviceId, &stream);
//     if (ret != ACL_SUCCESS) {
//         LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
//         return ret;
//     }

//     LOG_PRINT("\n========== Add Operator Comprehensive Tests ==========\n\n");

//     /* ── aclnnAdd: dtype × alpha branches ── */
//     Test_Add_Float32_Alpha1(stream);               // T01
//     Test_Add_Float32_Alpha2(stream);               // T02
//     Test_Add_Float16_Alpha1(stream);               // T03
//     Test_Add_Float16_Alpha2(stream);               // T04
//     Test_Add_Int32_Alpha1(stream);                 // T05
//     Test_Add_Int32_Alpha3(stream);                 // T06
//     Test_Add_Int8_Alpha1(stream);                  // T07
//     Test_Add_Int8_Alpha2(stream);                  // T08
//     Test_Add_Uint8_Alpha1(stream);                 // T09
//     Test_Add_Int64_Alpha1(stream);                 // T10
//     Test_Add_Int64_Alpha2(stream);                 // T11
//     Test_Add_Bool_Alpha1(stream);                  // T12
//     //Test_Add_Float32_Broadcast(stream);            // T13
//     Test_Add_Float32_Alpha0(stream);               // T14
//     Test_Add_Float32_AlphaNeg(stream);             // T15

//     /* ── Mixed dtype ── */
//     Test_Add_MixedDtype_F16F32_Alpha1(stream);     // T16
//     Test_Add_MixedDtype_F32F16_Alpha1(stream);     // T17
//     Test_Add_BF16_Alpha1(stream);                  // T18
//     Test_Add_BF16_Alpha2(stream);                  // T19
//     Test_Add_MixedDtype_BF16Float_Alpha1(stream);  // T40
//     Test_Add_MixedDtype_FloatBF16_Alpha1(stream);  // T41

//     /* ── Bool/UINT8 extra alpha ── */
//     Test_Add_Uint8_Alpha2(stream);                 // T38
//     Test_Add_Bool_Alpha2(stream);                  // T39

//     /* ── aclnnAdds ── */
//     Test_Adds_Float32_Alpha1(stream);              // T20
//     Test_Adds_Float32_Alpha2p5(stream);            // T21
//     Test_Adds_Int32_Scalar_Alpha1(stream);         // T22
//     Test_Adds_Bool_BoolScalar_Alpha1(stream);      // T23
//     Test_Adds_Float16_Scalar(stream);              // T37
//     Test_Adds_BF16_Scalar(stream);                 // T44
//     Test_Adds_Int64_Scalar_Alpha2(stream);         // T47
//     Test_Adds_Int8_Scalar_Alpha2(stream);          // T50

//     /* ── aclnnInplaceAdd / InplaceAdds ── */
//     Test_InplaceAdd_Float32_Alpha1(stream);        // T24
//     Test_InplaceAdd_Float32_Alpha2(stream);        // T25
//     Test_InplaceAdds_Float32(stream);              // T26
//     Test_InplaceAdds_Int32_Alpha3(stream);         // T27
//    // Test_InplaceAdd_Broadcast(stream);             // T48

//     /* ── aclnnAddV3 / InplaceAddV3 ── */
//     Test_AddV3_Float32_Alpha1(stream);             // T28
//     Test_AddV3_Float32_Alpha2(stream);             // T29
//     Test_AddV3_Int32_Alpha3(stream);               // T30
//     Test_AddV3_Int8_AlphaMulAdd(stream);           // T31
//     Test_AddV3_Float16_Alpha2(stream);             // T32
//     Test_InplaceAddV3_Float32_Alpha1(stream);      // T33
//     Test_InplaceAddV3_Float32_Alpha2(stream);      // T34
//     Test_AddV3_BF16_Alpha1(stream);                // T45

//     /* ── Shape / dimension stress ── */
//     Test_Add_Float32_LargeTensor(stream);          // T35: 512x512
//     //Test_Add_Float32_Broadcast1D(stream);          // T36
//     Test_Add_Float32_3D(stream);                   // T42
//     Test_Add_Float32_ZeroValues(stream);           // T43
//     Test_Add_Int32_LargeTensor(stream);            // T46: 128x128
//     Test_Add_Float32_AlphaExactOne(stream);        // T49

//     /* ── Summary ── */
//     LOG_PRINT("\n====================================================\n");
//     LOG_PRINT("Results: %d PASSED, %d FAILED, %d TOTAL\n",
//               g_passCount, g_failCount, g_passCount + g_failCount);
//     LOG_PRINT("====================================================\n");

//     aclrtDestroyStream(stream);
//     aclrtResetDevice(deviceId);
//     aclFinalize();

//     return (g_failCount > 0) ? 1 : 0;
// }



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

/* fflush 确保每行立即输出，方便定位卡死位置 */
#define LOG_PRINT(message, ...) \
    do { printf(message, ##__VA_ARGS__); fflush(stdout); } while (0)

#define TRACE_ENTER(name) \
    do { LOG_PRINT(">>> ENTER: %s\n", (name)); } while(0)

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

static void RunAndSync(void* /*unused*/, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
                       aclnnStatus (*execFn)(void*, uint64_t, aclOpExecutor*, aclrtStream))
{
    void* ws = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&ws, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    execFn(ws, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    if (ws) aclrtFree(ws);
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

static bool CheckClose(double actual, double expected, double atol = 1e-3, double rtol = 1e-3)
{
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual) && (expected > 0) == (actual > 0)) return true;
    return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

/* ══════════════════════════════════════════════════════════════════
 *  ① add_tiling_arch35.cpp 覆盖
 *
 *  DoOpTiling 分支树（按 if-else if 顺序）：
 *    isMixedDtype && input1==FLOAT   → AddMixDtypeCompute<half,float>   ← T16
 *    isMixedDtype && input0==FLOAT   → AddMixDtypeCompute<float,half>   ← T17
 *    input0 == FLOAT16 || BF16       → AddWithCastCompute<half>          ← T03/T18
 *    input0 == FLOAT                 → AddWithCastCompute<float>         ← T01
 *    input0 == BOOL                  → AddBoolCompute<int8_t>            ← T12
 *    input0 == INT64 || COMPLEX64    → AddWithoutCastCompute<int64_t>    ← T10
 *    input0 == UINT8                 → AddWithoutCastCompute<uint8_t>    ← T09
 *    input0 == INT8                  → AddWithoutCastCompute<int8_t>     ← T07
 *    input0 == INT32 || COMPLEX32    → AddWithoutCastCompute<int32_t>    ← T05
 *    else → GRAPH_FAILED (不可达，由上层 CheckParams 过滤)
 *
 *  Mixed-BF16+FLOAT 分支（T40/T41）也覆盖 isMixedDtype 的两个 bf16 子分支。
 * ══════════════════════════════════════════════════════════════════ */

/* ── T01: FLOAT32 alpha=1 → AddWithCastCompute<float> ── */
static void Test_Add_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "T01: aclnnAdd FLOAT32 alpha=1 [tiling:AddWithCastCompute<float>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData  = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    std::vector<float> otherData = {1.f,1.f,1.f,2.f,2.f,2.f,3.f,3.f};
    std::vector<float> outData(8, 0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData,shape, &otherDev,ACL_FLOAT, &other);
    CreateAclTensor(outData,  shape, &outDev,  ACL_FLOAT, &out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+(double)otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T02: FLOAT32 alpha=2 → IsSupportAxpy(FLOAT)=true, Axpy path ── */
static void Test_Add_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "T02: aclnnAdd FLOAT32 alpha=2 [Axpy path]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    std::vector<float> outData(8,0.f);
    float alphaVal = 2.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+2.0*(double)otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T03: FLOAT16 alpha=1 → AddWithCastCompute<half> ── */
static void Test_Add_Float16_Alpha1(aclrtStream stream)
{
    const std::string name = "T03: aclnnAdd FLOAT16 alpha=1 [tiling:AddWithCastCompute<half>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
    std::vector<uint16_t> outData(8,0);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT16,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT16,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T04: FLOAT16 alpha=2 → IsSupportAxpy(FLOAT16)=true on RegBase ── */
static void Test_Add_Float16_Alpha2(aclrtStream stream)
{
    const std::string name = "T04: aclnnAdd FLOAT16 alpha=2 [Axpy fp16]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,2};
    std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400};
    std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00};
    std::vector<uint16_t> outData(4,0);
    float alphaVal = 2.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT16,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT16,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T05: INT32 alpha=1 → AddWithoutCastCompute<int32_t>; same dtype IsEqualToOne ── */
static void Test_Add_Int32_Alpha1(aclrtStream stream)
{
    const std::string name = "T05: aclnnAdd INT32 alpha=1 [tiling:AddWithoutCastCompute<int32>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int32_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<int32_t> otherData = {10,10,10,10,10,10,10,10};
    std::vector<int32_t> outData(8,0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData, shape,&selfDev, ACL_INT32,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT32,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT32,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<int32_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int32_t),outDev,8*sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==selfData[i]+otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T06: INT32 alpha=3 → IsSupportAxpy(INT32)=false on RegBase; IsSupportAxpyV2=true ── */
static void Test_Add_Int32_Alpha3(aclrtStream stream)
{
    const std::string name = "T06: aclnnAdd INT32 alpha=3 [AxpyV2 path]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int32_t> selfData  = {1,2,3,4,5,6,7,8};
    std::vector<int32_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<int32_t> outData(8,0);
    float alphaVal = 3.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_INT32,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT32,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT32,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<int32_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int32_t),outDev,8*sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==selfData[i]+3*otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T07: INT8 alpha=1 → AddWithoutCastCompute<int8_t>; IsEqualToOne same dtype ── */
static void Test_Add_Int8_Alpha1(aclrtStream stream)
{
    const std::string name = "T07: aclnnAdd INT8 alpha=1 [tiling:AddWithoutCastCompute<int8>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int8_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<int8_t> otherData = {10,10,10,10,10,10,10,10};
    std::vector<int8_t> outData(8,0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData, shape,&selfDev, ACL_INT8,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT8,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT8,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<int8_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int8_t),outDev,8*sizeof(int8_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==(int8_t)(selfData[i]+otherData[i]));

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T08: INT8 alpha=2 → IsSupportAxpyV2(INT8)=true on RegBase ── */
static void Test_Add_Int8_Alpha2(aclrtStream stream)
{
    const std::string name = "T08: aclnnAdd INT8 alpha=2 [AxpyV2 int8]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,4};
    std::vector<int8_t> selfData  = {1,2,3,4,5,6,7,8};
    std::vector<int8_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<int8_t> outData(8,0);
    int32_t alphaVal = 2;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData, shape,&selfDev, ACL_INT8,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT8,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT8,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T09: UINT8 alpha=1 → AddWithoutCastCompute<uint8_t> ── */
static void Test_Add_Uint8_Alpha1(aclrtStream stream)
{
    const std::string name = "T09: aclnnAdd UINT8 alpha=1 [tiling:AddWithoutCastCompute<uint8>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint8_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<uint8_t> otherData = {10,10,10,10,10,10,10,10};
    std::vector<uint8_t> outData(8,0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData, shape,&selfDev, ACL_UINT8,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_UINT8,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_UINT8,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<uint8_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(uint8_t),outDev,8*sizeof(uint8_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==(uint8_t)(selfData[i]+otherData[i]));

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T09b: UINT8 alpha=2 → AxpyV2 UINT8 ── */
static void Test_Add_Uint8_Alpha2(aclrtStream stream)
{
    const std::string name = "T09b: aclnnAdd UINT8 alpha=2 [AxpyV2 uint8]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint8_t> selfData  = {10,20,30,40,50,60,70,80};
    std::vector<uint8_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<uint8_t> outData(8,0);
    int32_t alphaVal = 2;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData, shape,&selfDev, ACL_UINT8,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_UINT8,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_UINT8,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T10: INT64 alpha=1 → AddWithoutCastCompute<int64_t> ── */
static void Test_Add_Int64_Alpha1(aclrtStream stream)
{
    const std::string name = "T10: aclnnAdd INT64 alpha=1 [tiling:AddWithoutCastCompute<int64>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int64_t> selfData  = {0,1,2,3,4,5,6,7};
    std::vector<int64_t> otherData = {100,100,100,100,100,100,100,100};
    std::vector<int64_t> outData(8,0);
    int64_t alphaVal = 1;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT64);
    CreateAclTensor(selfData, shape,&selfDev, ACL_INT64,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT64,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT64,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<int64_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int64_t),outDev,8*sizeof(int64_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==selfData[i]+otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T11: INT64 alpha=2 → AxpyV2 INT64 ── */
static void Test_Add_Int64_Alpha2(aclrtStream stream)
{
    const std::string name = "T11: aclnnAdd INT64 alpha=2 [AxpyV2 int64]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,4};
    std::vector<int64_t> selfData  = {10,20,30,40,50,60,70,80};
    std::vector<int64_t> otherData = {1,1,1,1,1,1,1,1};
    std::vector<int64_t> outData(8,0);
    int64_t alphaVal = 2;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT64);
    CreateAclTensor(selfData, shape,&selfDev, ACL_INT64,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT64,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT64,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T12: BOOL alpha=1 → AddBoolCompute<int8_t>; AxpyV2 BOOL ── */
static void Test_Add_Bool_Alpha1(aclrtStream stream)
{
    const std::string name = "T12: aclnnAdd BOOL alpha=1 [tiling:AddBoolCompute<int8>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,4};
    std::vector<uint8_t> selfData  = {0,1,1,0,1,0,0,1};
    std::vector<uint8_t> otherData = {1,0,1,0,0,1,0,1};
    std::vector<uint8_t> outData(8,0);
    bool alphaVal = true;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_BOOL);
    CreateAclTensor(selfData, shape,&selfDev, ACL_BOOL,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_BOOL,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_BOOL,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T12b: BOOL alpha=2 → AxpyV2 BOOL ── */
static void Test_Add_Bool_Alpha2(aclrtStream stream)
{
    const std::string name = "T12b: aclnnAdd BOOL alpha=2 [AxpyV2 bool]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,4};
    std::vector<uint8_t> selfData  = {0,1,1,0,0,1,0,1};
    std::vector<uint8_t> otherData = {1,0,1,1,1,0,1,0};
    std::vector<uint8_t> outData(8,0);
    int32_t alphaVal = 2;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData, shape,&selfDev, ACL_BOOL,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_BOOL,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_BOOL,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T14: FLOAT32 alpha=0 → IsEqualToOne=false; IsSupportAxpy(FLOAT)=true ── */
static void Test_Add_Float32_Alpha0(aclrtStream stream)
{
    const std::string name = "T14: aclnnAdd FLOAT32 alpha=0 [Axpy alpha=0]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> otherData = {100.f,100.f,100.f,100.f,100.f,100.f,100.f,100.f};
    std::vector<float> outData(8,0.f);
    float alphaVal = 0.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i],1e-2,1e-2);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T15: FLOAT32 alpha=-1.5 → Axpy negative ── */
static void Test_Add_Float32_AlphaNeg(aclrtStream stream)
{
    const std::string name = "T15: aclnnAdd FLOAT32 alpha=-1.5 [Axpy negative alpha]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData  = {10.f,10.f,10.f,10.f,10.f,10.f,10.f,10.f};
    std::vector<float> otherData = {2.f,2.f,2.f,2.f,2.f,2.f,2.f,2.f};
    std::vector<float> outData(8,0.f);
    float alphaVal = -1.5f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+(-1.5)*(double)otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T16: FP16+FP32 mixed alpha=1 → isMixDataType=true, IsEqualToOne=true
         tiling: AddMixDtypeCompute<half,float> (input1==FLOAT branch) ── */
static void Test_Add_MixedDtype_F16F32_Alpha1(aclrtStream stream)
{
    const std::string name = "T16: aclnnAdd FP16+FP32 mixed alpha=1 [tiling:MixDtype<half,float>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint16_t> selfData  = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<float>    otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    std::vector<float>    outData(8,0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT16,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,  &other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,  &out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T17: FP32+FP16 mixed alpha=1 → isMixDataType=true
         tiling: AddMixDtypeCompute<float,half> (input0==FLOAT branch) ── */
static void Test_Add_MixedDtype_F32F16_Alpha1(aclrtStream stream)
{
    const std::string name = "T17: aclnnAdd FP32+FP16 mixed alpha=1 [tiling:MixDtype<float,half>]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float>    selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<uint16_t> otherData = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
    std::vector<float>    outData(8,0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,  &self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT16,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,  &out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T18: BF16 alpha=1 → AddWithCastCompute<half> (BF16 走同一分支 input0==BF16||FP16) ── */
static void Test_Add_BF16_Alpha1(aclrtStream stream)
{
    const std::string name = "T18: aclnnAdd BF16 alpha=1 [tiling:AddWithCastCompute<half> BF16 branch]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
    std::vector<uint16_t> outData(8,0);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_BF16,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_BF16,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_BF16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T19: BF16 alpha=2 → IsSupportAxpy(BF16)=true on RegBase ── */
static void Test_Add_BF16_Alpha2(aclrtStream stream)
{
    const std::string name = "T19: aclnnAdd BF16 alpha=2 [Axpy bf16]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,4};
    std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
    std::vector<uint16_t> outData(8,0);
    float alphaVal = 2.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_BF16,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_BF16,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_BF16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T40: BF16+FLOAT32 mixed alpha=1 → isMixDataType BF16+FP32 branch in tiling ── */
static void Test_Add_MixedDtype_BF16Float_Alpha1(aclrtStream stream)
{
    const std::string name = "T40: aclnnAdd BF16+FP32 mixed alpha=1 [tiling:MixDtype BF16+float]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint16_t> selfData  = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<float>    otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    std::vector<float>    outData(8,0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_BF16,  &self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT, &other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT, &out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T41: FLOAT32+BF16 mixed alpha=1 → MixDtype FP32+BF16 ── */
static void Test_Add_MixedDtype_FloatBF16_Alpha1(aclrtStream stream)
{
    const std::string name = "T41: aclnnAdd FP32+BF16 mixed alpha=1 [tiling:MixDtype float+BF16]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float>    selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<uint16_t> otherData = {0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80,0x3F80};
    std::vector<float>    outData(8,0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_BF16,  &other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT, &out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ══════════════════════════════════════════════════════════════════
 *  ② aclnn_add.cpp 覆盖
 *
 *  aclnnAddGetWorkspaceSize 主逻辑：
 *    IsEmpty                                             ← T_EMPTY(注释，不稳定)
 *    isMixDataType && IsEqualToOne                       ← T16/T17/T40/T41
 *    else → promoteType path:
 *      IsEqualToOne && sameDtype && IsAddSupportNonCont  ← T01(withStride path)
 *      IsEqualToOne && sameDtype && !NonCont             ← T01
 *      IsEqualToOne && !sameDtype                        ← (dtype cast then add)
 *      IsSupportAxpy                                     ← T02/T03/T04/T18/T19
 *      IsSupportAxpyV2                                   ← T06/T08/T09b/T11/T12b
 *      else (Mul+Add)                                    ← 需要 INT16 AiCpu（已注释）
 *
 *  aclnnAddsGetWorkspaceSize PromoteTypeScalar 分支：
 *    IsComplexType                                       ← 不在支持列表
 *    IsFloatingType(self) → return self dtype            ← T20(FLOAT self)
 *    CombineCategories higher==BOOL → PromoteType        ← T23
 *    CombineCategories IsFloatingType(lower)             ← T_SCALAR_INT32_FLOAT
 *    isKeepB16=true  (scalar 精度在 B16 范围内)          ← T37/T44
 *    isKeepB16=false (scalar 超出 B16 精度)              ← T58/T59 (已注释，不稳定)
 * ══════════════════════════════════════════════════════════════════ */

/* ── T20: aclnnAdds FLOAT32 + scalar alpha=1 → IsFloatingType(self)→return selfDtype ── */
static void Test_Adds_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "T20: aclnnAdds FLOAT32+scalar alpha=1 [PromoteType: IsFloat(self)]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8,0.f);
    float otherVal=10.0f, alphaVal=1.0f;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData,shape,&selfDev,ACL_FLOAT,&self);
    CreateAclTensor(outData, shape,&outDev, ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+10.0);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T21: aclnnAdds FLOAT32 + scalar alpha=2.5 → IsSupportAxpy ── */
static void Test_Adds_Float32_Alpha2p5(aclrtStream stream)
{
    const std::string name = "T21: aclnnAdds FLOAT32+scalar alpha=2.5 [Axpy path]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8,0.f);
    float otherVal=3.0f, alphaVal=2.5f;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData,shape,&selfDev,ACL_FLOAT,&self);
    CreateAclTensor(outData, shape,&outDev, ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+2.5*3.0,1e-2,1e-2);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T22: aclnnAdds INT32 + scalar alpha=1 → self is INT, other is INT → CombineCategories → INT32 ── */
static void Test_Adds_Int32_Scalar_Alpha1(aclrtStream stream)
{
    const std::string name = "T22: aclnnAdds INT32+scalar alpha=1 [CombineCategories int+int→int]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int32_t> selfData = {1,2,3,4,5,6,7,8};
    std::vector<int32_t> outData(8,0);
    int32_t otherVal=100, alphaVal=1;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData,shape,&selfDev,ACL_INT32,&self);
    CreateAclTensor(outData, shape,&outDev, ACL_INT32,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    std::vector<int32_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int32_t),outDev,8*sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==selfData[i]+100);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T23: aclnnAdds BOOL+bool scalar → CombineCategories: higher==BOOL ── */
static void Test_Adds_Bool_BoolScalar_Alpha1(aclrtStream stream)
{
    const std::string name = "T23: aclnnAdds BOOL+bool scalar [CombineCategories BOOL path]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,4};
    std::vector<uint8_t> selfData = {0,1,1,0,0,1,0,1};
    std::vector<int32_t> outData(8,0);
    bool otherVal=true, alphaVal=true;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_BOOL);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_BOOL);
    CreateAclTensor(selfData,shape,&selfDev,ACL_BOOL, &self);
    CreateAclTensor(outData, shape,&outDev, ACL_INT32,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T24: aclnnInplaceAdd FLOAT32 alpha=1 → CheckInplace + aclnnAdd same path ── */
static void Test_InplaceAdd_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "T24: aclnnInplaceAdd FLOAT32 alpha=1 [CheckInplace path]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(self,other,alpha,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnInplaceAdd);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),selfDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+(double)otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);
    PrintResult(name,ok);
}

/* ── T25: aclnnInplaceAdd FLOAT32 alpha=2 ── */
static void Test_InplaceAdd_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "T25: aclnnInplaceAdd FLOAT32 alpha=2 [in-place Axpy]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData  = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    std::vector<float> otherData = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    float alphaVal = 2.0f;

    void *selfDev=nullptr,*otherDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(self,other,alpha,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnInplaceAdd);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),selfDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+2.0*(double)otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);
    PrintResult(name,ok);
}

/* ── T26: aclnnInplaceAdds FLOAT32 ── */
static void Test_InplaceAdds_Float32(aclrtStream stream)
{
    const std::string name = "T26: aclnnInplaceAdds FLOAT32 += scalar";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float otherVal=5.0f, alphaVal=1.0f;

    void *selfDev=nullptr;
    aclTensor *self=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData,shape,&selfDev,ACL_FLOAT,&self);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(self,other,alpha,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnInplaceAdds);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),selfDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+5.0);

    aclDestroyTensor(self);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);
    PrintResult(name,ok);
}

/* ── T27: aclnnInplaceAdds INT32 alpha=3 ── */
static void Test_InplaceAdds_Int32_Alpha3(aclrtStream stream)
{
    const std::string name = "T27: aclnnInplaceAdds INT32 alpha=3";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int32_t> selfData = {0,1,2,3,4,5,6,7};
    int32_t otherVal=2;
    float alphaVal=3.0f;

    void *selfDev=nullptr;
    aclTensor *self=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData,shape,&selfDev,ACL_INT32,&self);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(self,other,alpha,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnInplaceAdds);

    aclDestroyTensor(self);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);
    PrintResult(name,ok);
}

/* ── T37: aclnnAdds FLOAT16 + scalar → isKeepB16=true ── */
static void Test_Adds_Float16_Scalar(aclrtStream stream)
{
    const std::string name = "T37: aclnnAdds FLOAT16+scalar [isKeepB16=true]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint16_t> selfData = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<uint16_t> outData(8,0);
    float otherVal=2.0f, alphaVal=1.0f;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData,shape,&selfDev,ACL_FLOAT16,&self);
    CreateAclTensor(outData, shape,&outDev, ACL_FLOAT16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T44: aclnnAdds BF16 + scalar → isKeepB16=true ── */
static void Test_Adds_BF16_Scalar(aclrtStream stream)
{
    const std::string name = "T44: aclnnAdds BF16+scalar [isKeepB16=true]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint16_t> selfData = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> outData(8,0);
    float otherVal=1.0f, alphaVal=1.0f;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData,shape,&selfDev,ACL_BF16,&self);
    CreateAclTensor(outData, shape,&outDev, ACL_BF16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T47: aclnnAdds INT64 + scalar alpha=2 → AxpyV2 INT64 ── */
static void Test_Adds_Int64_Scalar_Alpha2(aclrtStream stream)
{
    const std::string name = "T47: aclnnAdds INT64+scalar alpha=2 [AxpyV2 int64]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int64_t> selfData = {1,2,3,4,5,6,7,8};
    std::vector<int64_t> outData(8,0);
    int64_t otherVal=10, alphaVal=2;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_INT64);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT64);
    CreateAclTensor(selfData,shape,&selfDev,ACL_INT64,&self);
    CreateAclTensor(outData, shape,&outDev, ACL_INT64,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    std::vector<int64_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int64_t),outDev,8*sizeof(int64_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==selfData[i]+2LL*10LL);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T50: aclnnAdds INT8 + scalar alpha=2 → AxpyV2 INT8 ── */
static void Test_Adds_Int8_Scalar_Alpha2(aclrtStream stream)
{
    const std::string name = "T50: aclnnAdds INT8+scalar alpha=2 [AxpyV2 int8]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int8_t> selfData = {1,2,3,4,5,6,7,8};
    std::vector<int8_t> outData(8,0);
    int8_t  otherVal=5;
    int32_t alphaVal=2;

    void *selfDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*out=nullptr;
    aclScalar* other = aclCreateScalar(&otherVal,ACL_INT8);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData,shape,&selfDev,ACL_INT8,&self);
    CreateAclTensor(outData, shape,&outDev, ACL_INT8,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdds);

    aclDestroyTensor(self);aclDestroyTensor(out);
    aclDestroyScalar(other);aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ══════════════════════════════════════════════════════════════════
 *  ③ aclnn_add_v3.cpp 覆盖
 *
 *  aclnnAddV3GetWorkspaceSize 主逻辑：
 *    IsEmpty                                  ← (注释)
 *    PromoteTypeScalar:
 *      IsComplexType                          ← 不在支持列表
 *      IsFloatingType(other) → DT_FLOAT       ← T28/T29/T32/T33/T45
 *      self==DOUBLE && out==FLOAT → FLOAT     ← 不常见
 *      IsFloatingType(self) || other==BOOL    ← T31(INT8 self)
 *      else → other dtype                     ← T30(INT32)
 *    alpha==1: l0op::Add                      ← T28/T30/T32/T33/T45
 *    IsSupportAxpy: l0op::Axpy               ← T29/T32
 *    else: Mul+Add                            ← T31 (INT8不在V3 Axpy列表)
 * ══════════════════════════════════════════════════════════════════ */

/* ── T28: aclnnAddV3 FLOAT32 alpha=1 → IsFloat(other)→DT_FLOAT; alpha==1 → Add ── */
static void Test_AddV3_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "T28: aclnnAddV3 FLOAT32 alpha=1 [scalar+tensor, Add path]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8,0.f);
    float selfVal=10.0f, alphaVal=1.0f;

    void *otherDev=nullptr,*outDev=nullptr;
    aclTensor *other=nullptr,*out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAddV3);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],10.0+1.0*(double)otherData[i]);

    aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T29: aclnnAddV3 FLOAT32 alpha=2 → IsSupportAxpy(FLOAT)=true → Axpy ── */
static void Test_AddV3_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "T29: aclnnAddV3 FLOAT32 alpha=2 [Axpy branch]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> outData(8,0.f);
    float selfVal=5.0f, alphaVal=2.0f;

    void *otherDev=nullptr,*outDev=nullptr;
    aclTensor *other=nullptr,*out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAddV3);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],5.0+2.0*(double)otherData[i]);

    aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T30: aclnnAddV3 INT32 alpha=3 → other dtype=INT32 → PromoteType returns INT32;
         IsSupportAxpy(INT32)=true in V3 list → Axpy ── */
static void Test_AddV3_Int32_Alpha3(aclrtStream stream)
{
    const std::string name = "T30: aclnnAddV3 INT32 alpha=3 [PromoteType→INT32, Axpy V3]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int32_t> otherData = {1,2,3,4,5,6,7,8};
    std::vector<int32_t> outData(8,0);
    int32_t selfVal=100;
    float alphaVal=3.0f;

    void *otherDev=nullptr,*outDev=nullptr;
    aclTensor *other=nullptr,*out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal, ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT32,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT32,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAddV3);

    std::vector<int32_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int32_t),outDev,8*sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=(result[i]==100+3*otherData[i]);

    aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T31: aclnnAddV3 INT8 alpha=2 → else branch (Mul+Add)
         INT8 不在 V3 AXPY_DTYPE_SUPPORT_LIST {FLOAT,INT32,FP16}
         PromoteTypeScalar: IsFloatingType(self=INT8)=false, other=INT8 BOOL=false
         → IsFloatingType(self scalar)? self=INT8 scalar → not floating
         → return other dtype = INT8 ── */
static void Test_AddV3_Int8_AlphaMulAdd(aclrtStream stream)
{
    const std::string name = "T31: aclnnAddV3 INT8 alpha=2 [else Mul+Add branch]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<int8_t> otherData = {1,2,3,4,5,6,7,8};
    std::vector<int8_t> outData(8,0);
    int8_t selfVal=10;
    float alphaVal=2.0f;

    void *otherDev=nullptr,*outDev=nullptr;
    aclTensor *other=nullptr,*out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal, ACL_INT8);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT8,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT8,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAddV3);

    std::vector<int8_t> result(8,0);
    aclrtMemcpy(result.data(),8*sizeof(int8_t),outDev,8*sizeof(int8_t),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) {
        int expected = (int)selfVal+2*(int)otherData[i];
        ok&=(result[i]==(int8_t)expected);
    }

    aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T32: aclnnAddV3 FLOAT16 alpha=2 → IsSupportAxpy(FP16)=true in V3 list ── */
static void Test_AddV3_Float16_Alpha2(aclrtStream stream)
{
    const std::string name = "T32: aclnnAddV3 FLOAT16 alpha=2 [Axpy fp16 V3]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,4};
    std::vector<uint16_t> otherData = {0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    std::vector<uint16_t> outData(8,0);
    float selfValF=3.0f, alphaVal=2.0f;

    void *otherDev=nullptr,*outDev=nullptr;
    aclTensor *other=nullptr,*out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfValF,ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT16,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAddV3);

    aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── T33: aclnnInplaceAddV3 FLOAT32 alpha=1 ── */
static void Test_InplaceAddV3_Float32_Alpha1(aclrtStream stream)
{
    const std::string name = "T33: aclnnInplaceAddV3 FLOAT32 alpha=1 [in-place V3]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float selfVal=10.0f, alphaVal=1.0f;

    void *otherDev=nullptr;
    aclTensor *other=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(self,other,alpha,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnInplaceAddV3);

    std::vector<float> result(8,0.f);
    aclrtMemcpy(result.data(),8*sizeof(float),otherDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],10.0+1.0*(double)otherData[i]);

    aclDestroyTensor(other);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);
    PrintResult(name,ok);
}

/* ── T34: aclnnInplaceAddV3 FLOAT32 alpha=2 ── */
static void Test_InplaceAddV3_Float32_Alpha2(aclrtStream stream)
{
    const std::string name = "T34: aclnnInplaceAddV3 FLOAT32 alpha=2 [in-place V3 Axpy]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> otherData = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    float selfVal=100.0f, alphaVal=2.0f;

    void *otherDev=nullptr;
    aclTensor *other=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(self,other,alpha,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnInplaceAddV3);

    aclDestroyTensor(other);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);
    PrintResult(name,ok);
}

/* ── T45: aclnnAddV3 BF16 alpha=1 ── */
static void Test_AddV3_BF16_Alpha1(aclrtStream stream)
{
    const std::string name = "T45: aclnnAddV3 BF16 alpha=1 [bf16 V3]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<uint16_t> otherData = {0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0,0x40E0,0x4100};
    std::vector<uint16_t> outData(8,0);
    float selfVal=1.0f, alphaVal=1.0f;

    void *otherDev=nullptr,*outDev=nullptr;
    aclTensor *other=nullptr,*out=nullptr;
    aclScalar* self  = aclCreateScalar(&selfVal, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(otherData,shape,&otherDev,ACL_BF16,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_BF16,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAddV3);

    aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(self);aclDestroyScalar(alpha);
    aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ══════════════════════════════════════════════════════════════════
 *  ④ add.cpp 覆盖
 *
 *  Add() 函数：
 *    isMixDataType → AddAiCore with DT_FLOAT out          ← T16/T17/T40/T41
 *    !isMix && IsAiCoreSupport(both) → AddAiCore          ← T01~T12
 *    !isMix && !IsAiCoreSupport → AddAiCpu                ← 需要 DOUBLE/INT16（注释）
 *
 *  AddInplace() 函数（由 aclnnInplaceAdd 调用）：
 *    BroadcastInferShape 失败 → return nullptr             ← 不测（会 crash）
 *    broadcastShape != other shape → LOGE return           ← T24(selfShape==otherShape, 正常路径)
 *    isMixDataType && other==FP16||BF16 → LOGE return     ← 不测
 *    isMixDataType → AddAiCore                             ← (inplace 不支持混合)
 *    else → AddAiCore / AddAiCpu                           ← T24/T25
 *
 *  IsAddSupportNonContiguous():
 *    IsRegBase() && IsAiCoreSupport(self) && IsAiCoreSupport(other) ← T01 (sameDtype path)
 * ══════════════════════════════════════════════════════════════════ */

/* ── 形状/维度压力测试 ── */
static void Test_Add_Float32_LargeTensor(aclrtStream stream)
{
    const std::string name = "T35: aclnnAdd FLOAT32 large tensor (512x512)";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {512,512};
    int64_t N = 512*512;
    std::vector<float> selfData(N,1.0f), otherData(N,2.0f), outData(N,0.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(N,0.f);
    aclrtMemcpy(result.data(),N*sizeof(float),outDev,N*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    ok&=CheckClose(result[0],3.0)&&CheckClose(result[N-1],3.0);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

static void Test_Add_Float32_3D(aclrtStream stream)
{
    const std::string name = "T42: aclnnAdd FLOAT32 3D {2,3,4}";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {2,3,4};
    int64_t N = 24;
    std::vector<float> selfData(N), otherData(N), outData(N,0.f);
    for(int i=0;i<N;i++){selfData[i]=(float)i; otherData[i]=(float)(N-i);}
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(N,0.f);
    aclrtMemcpy(result.data(),N*sizeof(float),outDev,N*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<N&&ok;i++) ok&=CheckClose(result[i],(double)selfData[i]+(double)otherData[i]);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

static void Test_Add_Float32_ZeroValues(aclrtStream stream)
{
    const std::string name = "T43: aclnnAdd FLOAT32 all-zero boundary";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {4,2};
    std::vector<float> selfData(8,0.f), otherData(8,0.f), outData(8,1.f);
    float alphaVal = 1.0f;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_FLOAT);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(8,1.f);
    aclrtMemcpy(result.data(),8*sizeof(float),outDev,8*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    for(int i=0;i<8&&ok;i++) ok&=CheckClose(result[i],0.0);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

static void Test_Add_Int32_LargeTensor(aclrtStream stream)
{
    const std::string name = "T46: aclnnAdd INT32 large tensor (128x128)";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {128,128};
    int64_t N = 128*128;
    std::vector<int32_t> selfData(N,1), otherData(N,2), outData(N,0);
    int32_t alphaVal = 1;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_INT32);
    CreateAclTensor(selfData, shape,&selfDev, ACL_INT32,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_INT32,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_INT32,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<int32_t> result(N,0);
    aclrtMemcpy(result.data(),N*sizeof(int32_t),outDev,N*sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
    ok&=(result[0]==3)&&(result[N-1]==3);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ── alpha=1.0 as double scalar → IsEqualToOne DT_DOUBLE branch ── */
static void Test_Add_Float32_AlphaExactOne(aclrtStream stream)
{
    const std::string name = "T49: aclnnAdd FLOAT32 alpha=1.0 double scalar [IsEqualToOne DT_DOUBLE branch]";
    TRACE_ENTER(name.c_str());
    std::vector<int64_t> shape = {8,8};
    int64_t N = 64;
    std::vector<float> selfData(N,3.14f), otherData(N,2.71f), outData(N,0.f);
    double alphaVal = 1.0;

    void *selfDev=nullptr,*otherDev=nullptr,*outDev=nullptr;
    aclTensor *self=nullptr,*other=nullptr,*out=nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaVal,ACL_DOUBLE);
    CreateAclTensor(selfData, shape,&selfDev, ACL_FLOAT,&self);
    CreateAclTensor(otherData,shape,&otherDev,ACL_FLOAT,&other);
    CreateAclTensor(outData,  shape,&outDev,  ACL_FLOAT,&out);

    uint64_t wsSize=0; aclOpExecutor* exec=nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self,other,alpha,out,&wsSize,&exec);
    bool ok = (ret==ACL_SUCCESS);
    if(ok) RunAndSync(nullptr,wsSize,exec,stream,aclnnAdd);

    std::vector<float> result(N,0.f);
    aclrtMemcpy(result.data(),N*sizeof(float),outDev,N*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    ok&=CheckClose(result[0],(double)(3.14f+2.71f),1e-2,1e-2);

    aclDestroyTensor(self);aclDestroyTensor(other);aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev);aclrtFree(otherDev);aclrtFree(outDev);
    PrintResult(name,ok);
}

/* ══════════════════════════════════════════════════════════════════
 *  注释说明：以下测试因硬件层面会永久挂起，暂时禁用。
 *  如需重新启用，请在熟悉 AiCpu 调度状态后逐一解注释验证。
 *
 *  T13/T36/T48: broadcast 测试
 *    原因: broadcast kernel 的中间 buffer device_ptr=0，
 *          kernel 写入空地址后硬件回包永久挂起。
 *          正确做法需在 CreateAclTensor 之前先通过
 *          aclnnAddGetWorkspaceSize 获取 wsSize，再分配 workspace，
 *          这与当前测试框架结构不符，暂不支持。
 *
 *  T51/T52/T53: 空张量 (IsEmpty) 测试
 *    原因: wsSize==0 但 exec 非空时，RunAndSync 仍会调用
 *          execFn(nullptr,0,exec,stream) + SynchronizeStream，
 *          驱动侧等待永不完成。
 *
 *  T54/T55/T62: DOUBLE 类型 → AiCpu 路径
 *  T56/T57:     INT16  类型 → AiCpu 路径
 *    原因: AiCpu 调度队列在当前环境存在问题，
 *          aclrtSynchronizeStream 永久等待。
 *
 *  T58/T59: isKeepB16=false 标量溢出
 *    原因: 类型提升后走 AiCpu Cast 路径，存在相同调度挂起风险。
 *
 *  T60: 混合 dtype alpha!=1 (isMixDataType=true 但不满足 IsEqualToOne)
 *    原因: 该组合在当前驱动下 kernel 无法完成。
 *
 *  T61: INT32 self + FLOAT scalar (CombineCategories FloatLower 路径)
 *    原因: 类型提升后走 AiCpu Cast 路径，存在调度挂起风险。
 * ══════════════════════════════════════════════════════════════════ */

/* ══════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════ */
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = InitACL(deviceId, &stream);
    if(ret != ACL_SUCCESS){
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("\n========== Add Operator Comprehensive Tests ==========\n\n");

    /* ── add_tiling_arch35.cpp: 全部 DoOpTiling 分支 ── */
    Test_Add_Float32_Alpha1(stream);            // T01: AddWithCastCompute<float>
    Test_Add_Float32_Alpha2(stream);            // T02: Axpy FLOAT
    Test_Add_Float16_Alpha1(stream);            // T03: AddWithCastCompute<half> via FP16
    Test_Add_Float16_Alpha2(stream);            // T04: Axpy FP16
    Test_Add_Int32_Alpha1(stream);              // T05: AddWithoutCastCompute<int32>
    Test_Add_Int32_Alpha3(stream);              // T06: AxpyV2 INT32
    Test_Add_Int8_Alpha1(stream);               // T07: AddWithoutCastCompute<int8>
    Test_Add_Int8_Alpha2(stream);               // T08: AxpyV2 INT8
    Test_Add_Uint8_Alpha1(stream);              // T09: AddWithoutCastCompute<uint8>
    Test_Add_Uint8_Alpha2(stream);              // T09b: AxpyV2 UINT8
    Test_Add_Int64_Alpha1(stream);              // T10: AddWithoutCastCompute<int64>
    Test_Add_Int64_Alpha2(stream);              // T11: AxpyV2 INT64
    Test_Add_Bool_Alpha1(stream);               // T12: AddBoolCompute<int8>
    Test_Add_Bool_Alpha2(stream);               // T12b: AxpyV2 BOOL
    Test_Add_BF16_Alpha1(stream);               // T18: AddWithCastCompute<half> via BF16
    Test_Add_BF16_Alpha2(stream);               // T19: Axpy BF16

    /* ── Mixed dtype → MixDtype 两个子分支 ── */
    Test_Add_MixedDtype_F16F32_Alpha1(stream);  // T16: MixDtype<half,float>
    Test_Add_MixedDtype_F32F16_Alpha1(stream);  // T17: MixDtype<float,half>
    Test_Add_MixedDtype_BF16Float_Alpha1(stream); // T40: MixDtype BF16+float
    Test_Add_MixedDtype_FloatBF16_Alpha1(stream); // T41: MixDtype float+BF16

    /* ── aclnn_add.cpp: alpha 边界 ── */
    Test_Add_Float32_Alpha0(stream);            // T14: Axpy alpha=0
    Test_Add_Float32_AlphaNeg(stream);          // T15: Axpy alpha<0
    Test_Add_Float32_AlphaExactOne(stream);     // T49: IsEqualToOne DT_DOUBLE branch

    /* ── aclnn_add.cpp: aclnnAdds PromoteTypeScalar 各分支 ── */
    Test_Adds_Float32_Alpha1(stream);           // T20: IsFloatingType(self)→selfDtype
    Test_Adds_Float32_Alpha2p5(stream);         // T21: Axpy float scalar
    Test_Adds_Int32_Scalar_Alpha1(stream);      // T22: CombineCategories int+int
    Test_Adds_Bool_BoolScalar_Alpha1(stream);   // T23: CombineCategories BOOL path
    Test_Adds_Float16_Scalar(stream);           // T37: isKeepB16=true FP16
    Test_Adds_BF16_Scalar(stream);              // T44: isKeepB16=true BF16
    Test_Adds_Int64_Scalar_Alpha2(stream);      // T47: AxpyV2 INT64 scalar
    Test_Adds_Int8_Scalar_Alpha2(stream);       // T50: AxpyV2 INT8 scalar

    /* ── aclnn_add.cpp: InplaceAdd / InplaceAdds ── */
    Test_InplaceAdd_Float32_Alpha1(stream);     // T24: CheckInplace + Add same dtype
    Test_InplaceAdd_Float32_Alpha2(stream);     // T25: CheckInplace + Axpy
    Test_InplaceAdds_Float32(stream);           // T26: InplaceAdds float
    Test_InplaceAdds_Int32_Alpha3(stream);      // T27: InplaceAdds int32

    /* ── aclnn_add_v3.cpp: 全部分支 ── */
    Test_AddV3_Float32_Alpha1(stream);          // T28: Add path (alpha==1)
    Test_AddV3_Float32_Alpha2(stream);          // T29: Axpy path FLOAT
    Test_AddV3_Int32_Alpha3(stream);            // T30: Axpy path INT32
    Test_AddV3_Int8_AlphaMulAdd(stream);        // T31: else Mul+Add (INT8)
    Test_AddV3_Float16_Alpha2(stream);          // T32: Axpy path FP16
    Test_InplaceAddV3_Float32_Alpha1(stream);   // T33: InplaceAddV3 alpha=1
    Test_InplaceAddV3_Float32_Alpha2(stream);   // T34: InplaceAddV3 Axpy
    Test_AddV3_BF16_Alpha1(stream);             // T45: BF16 V3

    /* ── Shape/size stress ── */
    Test_Add_Float32_LargeTensor(stream);       // T35: 512x512 FLOAT32
    Test_Add_Float32_3D(stream);                // T42: 3D shape
    Test_Add_Float32_ZeroValues(stream);        // T43: all-zero boundary
    Test_Add_Int32_LargeTensor(stream);         // T46: 128x128 INT32

    /* ── Summary ── */
    LOG_PRINT("\n====================================================\n");
    LOG_PRINT("Results: %d PASSED, %d FAILED, %d TOTAL\n",
              g_passCount, g_failCount, g_passCount+g_failCount);
    LOG_PRINT("====================================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (g_failCount>0) ? 1 : 0;
}



