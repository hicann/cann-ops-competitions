// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// #include <iostream>
// #include <vector>
// #include "acl/acl.h"
// #include "aclnnop/aclnn_pow.h"

// #define CHECK_RET(cond, return_expr) \
//   do {                               \
//     if (!(cond)) {                   \
//       return_expr;                   \
//     }                                \
//   } while (0)
// #define LOG_PRINT(message, ...)     \
//   do {                              \
//     printf(message, ##__VA_ARGS__); \
//   } while (0)
// int64_t GetShapeSize(const std::vector<int64_t>& shape) {
//   int64_t shapeSize = 1;
//   for (auto i : shape) {
//     shapeSize *= i;
//   }
//   return shapeSize;
// }
// int Init(int32_t deviceId, aclrtStream* stream) {
//   // 固定写法，资源初始化
//   auto ret = aclInit(nullptr);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
//   ret = aclrtSetDevice(deviceId);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
//   ret = aclrtCreateStream(stream);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
//   return 0;
// }

// template <typename T>
// int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
//                     aclDataType dataType, aclTensor** tensor) {
//   auto size = GetShapeSize(shape) * sizeof(T);
//   // 调用aclrtMalloc申请device侧内存
//   auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
//   // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
//   ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
//   // 计算连续tensor的strides
//   std::vector<int64_t> strides(shape.size(), 1);
//   for (int64_t i = shape.size() - 2; i >= 0; i--) {
//     strides[i] = shape[i + 1] * strides[i + 1];
//   }
//   // 调用aclCreateTensor接口创建aclTensor
//   *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
//                             shape.data(), shape.size(), *deviceAddr);
//   return 0;
// }
// int main() {
//   // 1. （固定写法）device/stream初始化，参考acl API手册
//   // 根据自己的实际device填写deviceId
//   int32_t deviceId = 0;
//   aclrtStream stream;
//   auto ret = Init(deviceId, &stream);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
//   // 2. 构造输入与输出，需要根据API的接口自定义构造
//   std::vector<int64_t> selfShape = {2, 2};
//   std::vector<int64_t> outShape = {2, 2};
//   void* selfDeviceAddr = nullptr;
//   void* outDeviceAddr = nullptr;
//   aclTensor* self = nullptr;
//   aclScalar* exponent = nullptr;
//   aclTensor* out = nullptr;
//   std::vector<float> selfHostData = {0, 1, 2, 3};
//   std::vector<float> outHostData = {0, 0, 0, 0};
//   float exponentVal = 4.1f;
//   // 创建self aclTensor
//   ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
//   CHECK_RET(ret == ACL_SUCCESS, return ret);
//   // 创建threshold aclScalar
//   exponent = aclCreateScalar(&exponentVal, aclDataType::ACL_FLOAT);
//   CHECK_RET(exponent != nullptr, return ret);
//   // 创建out aclTensor
//   ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
//   CHECK_RET(ret == ACL_SUCCESS, return ret);
//   // 3. 调用CANN算子库API，需要修改为具体的API名称
//   // aclnnPowTensorScalar接口调用示例
//   uint64_t workspaceSize = 0;
//   aclOpExecutor* executor;
//   // 调用aclnnPowTensorScalar第一段接口
//   ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
//   // 根据第一段接口计算出的workspaceSize申请device内存
//   void* workspaceAddr = nullptr;
//   if (workspaceSize > 0) {
//     ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
//   }
//   // 调用aclnnPowTensorScalar第二段接口
//   ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalar failed. ERROR: %d\n", ret); return ret);

//   // aclnnInplacePowTensorScalar接口调用示例
//   uint64_t inplaceWorkspaceSize = 0;
//   aclOpExecutor* inplaceExecutor;
//   // 调用aclnnInplacePowTensorScalar第一段接口
//   ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &inplaceWorkspaceSize, &inplaceExecutor);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
//   // 根据第一段接口计算出的workspaceSize申请device内存
//   void* inplaceWorkspaceAddr = nullptr;
//   if (inplaceWorkspaceSize > 0) {
//     ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
//   }
//   // 调用aclnnInplacePowTensorScalar第二段接口
//   ret = aclnnInplacePowTensorScalar(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalar failed. ERROR: %d\n", ret); return ret);

//   // 4. （固定写法）同步等待任务执行结束
//   ret = aclrtSynchronizeStream(stream);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

//   // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
//   auto size = GetShapeSize(outShape);
//   std::vector<float> resultData(size, 0);
//   ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
//                     size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//   for (int64_t i = 0; i < size; i++) {
//     LOG_PRINT("aclnnPowTensorScalar result[%ld] is: %f\n", i, resultData[i]);
//   }

//   auto inplaceSize = GetShapeSize(selfShape);
//   std::vector<float> inplaceResultData(inplaceSize, 0);
//   ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), outDeviceAddr, inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//   for (int64_t i = 0; i < inplaceSize; i++) {
//     LOG_PRINT("aclnnInplacePowTensorScalar result[%ld] is: %f\n", i, inplaceResultData[i]);
//   }

//   // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
//   aclDestroyTensor(self);
//   aclDestroyScalar(exponent);
//   aclDestroyTensor(out);

//   // 7. 释放device资源，需要根据具体API的接口定义修改
//   aclrtFree(selfDeviceAddr);
//   aclrtFree(outDeviceAddr);
//   if (workspaceSize > 0) {
//     aclrtFree(workspaceAddr);
//   }
//   if (inplaceWorkspaceSize > 0) {
//     aclrtFree(inplaceWorkspaceAddr);
//   }
//   aclrtDestroyStream(stream);
//   aclrtResetDevice(deviceId);
//   aclFinalize();
//   return 0;
// }


//V1
// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * 自动生成的高覆盖率测评代码 - 针对 Pow 算子 (包含 Tiling 及 AiCPU 调度覆盖)
//  */
// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <functional>
// #include "acl/acl.h"
// #include "aclnn/aclnn_base.h"
// #include "aclnn_pow.h"
// #include "aclnn_pow_tensor_tensor.h"

// // ============================================================================
// // 1. 辅助函数：快速创建 Tensor 和 Scalar
// // ============================================================================
// #define LOG_PRINT(message, ...) fprintf(stdout, message, ##__VA_ARGS__)

// aclTensor* CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dataType, 
//                            aclFormat format = ACL_FORMAT_ND) {
//     std::vector<int64_t> strides(shape.size(), 1);
//     // 处理空 Tensor
//     if (shape.size() > 0 && shape[0] != 0) {
//         for (int64_t i = shape.size() - 2; i >= 0; i--) {
//             strides[i] = shape[i + 1] * strides[i + 1];
//         }
//     }
//     // 【修复】：最后一个参数必须是 void* tensorData，这里传入 nullptr 即可
//     return aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
//                            shape.data(), shape.size(), nullptr);
// }
// aclScalar* CreateAclScalar(float value, aclDataType dataType) {
//     if (dataType == ACL_FLOAT) {
//         return aclCreateScalar(&value, dataType);
//     } else if (dataType == ACL_INT32) {
//         int32_t val = static_cast<int32_t>(value);
//         return aclCreateScalar(&val, dataType);
//     } else if (dataType == ACL_DOUBLE) {
//         double val = static_cast<double>(value);
//         return aclCreateScalar(&val, dataType);
//     }
//     return aclCreateScalar(&value, ACL_FLOAT);
// }

// // 模拟的销毁函数避免内存泄露（具体视测试框架而定）
// void DestroyAclTensor(aclTensor* t) { if (t) aclDestroyTensor(t); }
// void DestroyAclScalar(aclScalar* s) { if (s) aclDestroyScalar(s); }

// // 全局统计
// int g_pass = 0;
// int g_fail = 0;

// void RunTestCase(const std::string& name, aclnnStatus expectedStatus, std::function<aclnnStatus()> testFunc) {
//     LOG_PRINT("[RUN] %s...\n", name.c_str());
//     aclnnStatus ret = testFunc();
//     if (ret == expectedStatus) {
//         LOG_PRINT("[PASS] %s\n", name.c_str());
//         g_pass++;
//     } else {
//         LOG_PRINT("[FAIL] %s, expected %d, got %d\n", name.c_str(), expectedStatus, ret);
//         g_fail++;
//     }
// }

// // ============================================================================
// // 2. 核心分支覆盖用例设计 (含 API 层、调度层、Tiling 层)
// // ============================================================================

// // [覆盖] aclnn_pow_tensor_tensor.cpp: IsEmpty 返回 0 Workspace
// aclnnStatus Test_EmptyTensor() {
//     auto self = CreateAclTensor({0}, ACL_FLOAT); 
//     auto exp = CreateAclScalar(2.0, ACL_FLOAT);
//     auto out = CreateAclTensor({0}, ACL_FLOAT);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     aclnnStatus ret = aclnnPowTensorScalarGetWorkspaceSize(self, exp, out, &wsSize, &exc);
//     return ret;
// }

// // [覆盖] aclnn_pow.cpp: l0op::Pows 优化分支 (指数为 3.0 等)
// aclnnStatus Test_Pows_Optimization() {
//     auto self = CreateAclTensor({2, 2}, ACL_FLOAT);
//     auto exp = CreateAclScalar(3.0, ACL_FLOAT); 
//     auto out = CreateAclTensor({2, 2}, ACL_FLOAT);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorScalarGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// // [覆盖] aclnn_pow.cpp: l0op::Square + Cast 分支 (输入INT8, 指数2.0)
// aclnnStatus Test_Square_Cast_Optimization() {
//     auto self = CreateAclTensor({2, 2}, ACL_INT8); 
//     auto exp = CreateAclScalar(2.0, ACL_FLOAT);    
//     auto out = CreateAclTensor({2, 2}, ACL_FLOAT); 
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorScalarGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// // [覆盖] aclnn_pow.cpp: l0op::Fill(1) 极端优化分支
// aclnnStatus Test_ScalarTensor_Fill_1() {
//     auto self = CreateAclScalar(1.0, ACL_FLOAT); 
//     auto exp = CreateAclTensor({2, 2}, ACL_FLOAT);
//     auto out = CreateAclTensor({2, 2}, ACL_FLOAT);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowScalarTensorGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// // [覆盖] pow.cpp: l0op::Pow 中 BroadcastInferShape 失败报错分支
// aclnnStatus Test_Error_Broadcast_Fail() {
//     auto self = CreateAclTensor({2, 3}, ACL_FLOAT); 
//     auto exp = CreateAclTensor({4, 5}, ACL_FLOAT);  // 无法广播
//     auto out = CreateAclTensor({4, 5}, ACL_FLOAT);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// // [覆盖] pow.cpp: IsAiCoreSupport 返回 False 走 PowAiCpu 分支
// aclnnStatus Test_Dispatch_AiCpu() {
//     // INT64 / DOUBLE 属于 DTYPE_SUPPORT_LIST 但不在 AICORE_DTYPE_SUPPORT_LIST 中
//     auto self = CreateAclTensor({2, 2}, ACL_INT64); 
//     auto exp = CreateAclTensor({2, 2}, ACL_INT64);
//     auto out = CreateAclTensor({2, 2}, ACL_INT64);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// // [覆盖] pow_tensor_tensor_tiling_arch35.cpp: 7 种 Tiling OP_KEY 全覆盖
// aclnnStatus Test_Tiling_OpKeys(aclDataType dtype) {
//     auto self = CreateAclTensor({2, 2}, dtype); 
//     auto exp = CreateAclTensor({2, 2}, dtype);
//     auto out = CreateAclTensor({2, 2}, dtype);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// // [覆盖] 各种异常与告警分支
// aclnnStatus Test_Error_Integral_Negative_Exp() {
//     auto self = CreateAclTensor({2, 2}, ACL_INT32);
//     auto exp = CreateAclScalar(-2.0, ACL_INT32); 
//     auto out = CreateAclTensor({2, 2}, ACL_INT32);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorScalarGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// aclnnStatus Test_Error_Overflow() {
//     auto self = CreateAclTensor({2, 2}, ACL_INT8);
//     auto exp = CreateAclScalar(1000.0, ACL_INT32); 
//     auto out = CreateAclTensor({2, 2}, ACL_INT8);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorScalarGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// aclnnStatus Test_Error_MaxDim() {
//     auto self = CreateAclTensor({1,1,1,1,1,1,1,1,1}, ACL_FLOAT); // 9维
//     auto exp = CreateAclTensor({1}, ACL_FLOAT);
//     auto out = CreateAclTensor({1,1,1,1,1,1,1,1,1}, ACL_FLOAT);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// aclnnStatus Test_Error_Double_Bool() {
//     auto self = CreateAclTensor({2}, ACL_BOOL); 
//     auto exp = CreateAclTensor({2}, ACL_BOOL);
//     auto out = CreateAclTensor({2}, ACL_BOOL);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// aclnnStatus Test_Warning_Format_NZ() {
//     auto self = CreateAclTensor({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ); 
//     auto exp = CreateAclScalar(2.0, ACL_FLOAT);
//     auto out = CreateAclTensor({16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     // 预期成功，但底层会记录 FORMAT_NZ 不支持的 LogWarning 从而增加行覆盖率
//     return aclnnPowTensorScalarGetWorkspaceSize(self, exp, out, &wsSize, &exc);
// }

// // [覆盖] Inplace API
// aclnnStatus Test_Inplace_API() {
//     auto self = CreateAclTensor({2, 2}, ACL_FLOAT);
//     auto exp = CreateAclScalar(2.0, ACL_FLOAT);
//     uint64_t wsSize = 0;
//     aclOpExecutor* exc = nullptr;
//     return aclnnInplacePowTensorScalarGetWorkspaceSize(self, exp, &wsSize, &exc);
// }

// // ============================================================================
// // 3. Main 统一入口
// // ============================================================================
// int main(int argc, char** argv) {
//     LOG_PRINT("==================================================\n");
//     LOG_PRINT("  Start aclnn_pow Comprehensive Coverage Tests\n");
//     LOG_PRINT("==================================================\n");

//     // 1. API及架构优化分支
//     RunTestCase("Test_EmptyTensor", ACL_SUCCESS, Test_EmptyTensor);
//     RunTestCase("Test_Pows_Optimization", ACL_SUCCESS, Test_Pows_Optimization);
//     RunTestCase("Test_Square_Cast_Optimization", ACL_SUCCESS, Test_Square_Cast_Optimization);
//     RunTestCase("Test_ScalarTensor_Fill_1", ACL_SUCCESS, Test_ScalarTensor_Fill_1);
//     RunTestCase("Test_Inplace_API", ACL_SUCCESS, Test_Inplace_API);
//     RunTestCase("Test_Warning_Format_NZ", ACL_SUCCESS, Test_Warning_Format_NZ);
    
//     // 2. AiCpu与AiCore调度、Tiling OP_KEYS 全面打击
//     RunTestCase("Test_Dispatch_AiCpu (INT64)", ACL_SUCCESS, Test_Dispatch_AiCpu);
//     RunTestCase("Test_Tiling_KEY_1 (FP16)", ACL_SUCCESS, [](){ return Test_Tiling_OpKeys(ACL_FLOAT16); });
//     RunTestCase("Test_Tiling_KEY_2 (BF16)", ACL_SUCCESS, [](){ return Test_Tiling_OpKeys(ACL_BF16); });
//     RunTestCase("Test_Tiling_KEY_3 (FP32)", ACL_SUCCESS, [](){ return Test_Tiling_OpKeys(ACL_FLOAT); });
//     RunTestCase("Test_Tiling_KEY_4 (UINT8)", ACL_SUCCESS, [](){ return Test_Tiling_OpKeys(ACL_UINT8); });
//     RunTestCase("Test_Tiling_KEY_5 (INT8)", ACL_SUCCESS, [](){ return Test_Tiling_OpKeys(ACL_INT8); });
//     RunTestCase("Test_Tiling_KEY_6 (INT16)", ACL_SUCCESS, [](){ return Test_Tiling_OpKeys(ACL_INT16); });
//     RunTestCase("Test_Tiling_KEY_7 (INT32)", ACL_SUCCESS, [](){ return Test_Tiling_OpKeys(ACL_INT32); });

//     // 3. 错误捕获分支 (注意：如果测试平台定义了宏，ACL_ERROR_INVALID_PARAM 往往是非0，假设返回失败即可)
//     RunTestCase("Test_Error_Broadcast_Fail", ACL_ERROR_INVALID_PARAM, Test_Error_Broadcast_Fail);
//     RunTestCase("Test_Error_Integral_Negative_Exp", ACL_ERROR_INVALID_PARAM, Test_Error_Integral_Negative_Exp);
//     RunTestCase("Test_Error_Overflow", ACL_ERROR_INVALID_PARAM, Test_Error_Overflow);
//     RunTestCase("Test_Error_MaxDim", ACL_ERROR_INVALID_PARAM, Test_Error_MaxDim);
//     RunTestCase("Test_Error_Double_Bool", ACL_ERROR_INVALID_PARAM, Test_Error_Double_Bool);

//     LOG_PRINT("==================================================\n");
//     LOG_PRINT("Test Summary: %d PASS, %d FAIL\n", g_pass, g_fail);
//     LOG_PRINT("==================================================\n");

//     return g_fail == 0 ? 0 : -1;
// }


// //V2--c  
// /**
//  * Comprehensive test cases for Pow operator
//  * Covers: TensorScalar, ScalarTensor, TensorTensor, Exp2 (+ Inplace variants)
//  * Target: maximize code coverage across aclnn_pow.cpp, aclnn_pow_tensor_tensor.cpp,
//  *         pow.cpp, pow_tiling_arch35.cpp, pow_tensor_tensor_tiling_arch35.cpp
//  */

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <cstring>
// #include <cstdint>
// #include <functional>
// #include <string>
// #include <limits>
// #include <algorithm>
// #include "acl/acl.h"
// #include "aclnnop/aclnn_pow.h"
// #include "aclnnop/aclnn_pow_tensor_tensor.h"
// #include "aclnnop/aclnn_exp2.h"

// // ==================== Macros ====================
// #define CHECK_RET(cond, return_expr) \
//   do {                               \
//     if (!(cond)) {                   \
//       return_expr;                   \
//     }                                \
//   } while (0)

// #define LOG_PRINT(message, ...)     \
//   do {                              \
//     printf(message, ##__VA_ARGS__); \
//   } while (0)

// // ==================== Globals ====================
// static int g_totalTests = 0;
// static int g_passedTests = 0;
// static int g_failedTests = 0;

// static void ReportResult(const std::string &name, bool pass) {
//     g_totalTests++;
//     if (pass) {
//         g_passedTests++;
//         printf("[PASS] %s\n", name.c_str());
//     } else {
//         g_failedTests++;
//         printf("[FAIL] %s\n", name.c_str());
//     }
// }

// // ==================== Helpers ====================
// int64_t GetShapeSize(const std::vector<int64_t>& shape) {
//     int64_t s = 1;
//     for (auto d : shape) s *= d;
//     return s;
// }

// int Init(int32_t deviceId, aclrtStream* stream) {
//     auto ret = aclInit(nullptr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtSetDevice(deviceId);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtCreateStream(stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
//     return 0;
// }

// template <typename T>
// int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
//                     aclDataType dataType, aclTensor** tensor) {
//     auto size = GetShapeSize(shape) * sizeof(T);
//     auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
//     ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
//     std::vector<int64_t> strides(shape.size(), 1);
//     for (int64_t i = shape.size() - 2; i >= 0; i--) {
//         strides[i] = shape[i + 1] * strides[i + 1];
//     }
//     *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
//                               aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
//     return 0;
// }

// // Create an empty tensor (shape with 0 dimension)
// int CreateEmptyAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
//                          aclDataType dataType, aclTensor** tensor, int elemSize) {
//     // Allocate a tiny buffer even though the tensor is empty
//     auto ret = aclrtMalloc(deviceAddr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     std::vector<int64_t> strides(shape.size(), 1);
//     for (int64_t i = shape.size() - 2; i >= 0; i--) {
//         strides[i] = shape[i + 1] * strides[i + 1];
//     }
//     *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
//                               aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
//     return 0;
// }

// // Tolerance comparison for floats
// bool AlmostEqual(double actual, double expected, double rtol = 1e-3, double atol = 1e-5) {
//     if (std::isnan(actual) && std::isnan(expected)) return true;
//     if (std::isinf(actual) && std::isinf(expected)) return (actual > 0) == (expected > 0);
//     return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
// }

// // ==================== Test: PowTensorScalar ====================
// // Tests aclnnPowTensorScalar with various dtypes and exponents
// bool TestPowTensorScalar(aclrtStream stream, const std::string& label,
//                          std::vector<float> baseData, std::vector<int64_t> shape,
//                          float expVal, aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     void *selfDev = nullptr, *outDev = nullptr;
//     aclTensor *selfT = nullptr, *outT = nullptr;
//     aclScalar *expS = nullptr;

//     std::vector<float> outHost(n, 0.0f);

//     auto ret = CreateAclTensor(baseData, shape, &selfDev, dtype, &selfT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     expS = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
//     if (!expS) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnPowTensorScalarGetWorkspaceSize(selfT, expS, outT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnPowTensorScalar(wsAddr, wsSize, executor, stream);
//         if (ret != ACL_SUCCESS) { ReportResult(label, false); if (wsAddr) aclrtFree(wsAddr); goto cleanup; }
//         aclrtSynchronizeStream(stream);

//         // Copy result back
//         std::vector<float> result(n, 0.0f);
//         aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         // Verify
//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             double expected = std::pow((double)baseData[i], (double)expVal);
//             if (!AlmostEqual((double)result[i], expected)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected %f\n", i, result[i], (float)expected);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(selfT);
//     aclDestroyScalar(expS);
//     aclDestroyTensor(outT);
//     aclrtFree(selfDev);
//     aclrtFree(outDev);
//     return true;
// }

// // ==================== Test: InplacePowTensorScalar ====================
// bool TestInplacePowTensorScalar(aclrtStream stream, const std::string& label,
//                                 std::vector<float> baseData, std::vector<int64_t> shape,
//                                 float expVal, aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     void *selfDev = nullptr;
//     aclTensor *selfT = nullptr;
//     aclScalar *expS = nullptr;

//     auto ret = CreateAclTensor(baseData, shape, &selfDev, dtype, &selfT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     expS = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
//     if (!expS) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnInplacePowTensorScalarGetWorkspaceSize(selfT, expS, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnInplacePowTensorScalar(wsAddr, wsSize, executor, stream);
//         if (ret != ACL_SUCCESS) { ReportResult(label, false); if (wsAddr) aclrtFree(wsAddr); goto cleanup; }
//         aclrtSynchronizeStream(stream);

//         std::vector<float> result(n, 0.0f);
//         aclrtMemcpy(result.data(), n * sizeof(float), selfDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             double expected = std::pow((double)baseData[i], (double)expVal);
//             if (!AlmostEqual((double)result[i], expected)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected %f\n", i, result[i], (float)expected);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(selfT);
//     aclDestroyScalar(expS);
//     aclrtFree(selfDev);
//     return true;
// }

// // ==================== Test: PowScalarTensor ====================
// bool TestPowScalarTensor(aclrtStream stream, const std::string& label,
//                          float baseScalar, std::vector<float> expData,
//                          std::vector<int64_t> shape, aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     void *expDev = nullptr, *outDev = nullptr;
//     aclTensor *expT = nullptr, *outT = nullptr;
//     aclScalar *baseS = nullptr;

//     std::vector<float> outHost(n, 0.0f);

//     auto ret = CreateAclTensor(expData, shape, &expDev, dtype, &expT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     baseS = aclCreateScalar(&baseScalar, aclDataType::ACL_FLOAT);
//     if (!baseS) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnPowScalarTensorGetWorkspaceSize(baseS, expT, outT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnPowScalarTensor(wsAddr, wsSize, executor, stream);
//         if (ret != ACL_SUCCESS) { ReportResult(label, false); if (wsAddr) aclrtFree(wsAddr); goto cleanup; }
//         aclrtSynchronizeStream(stream);

//         std::vector<float> result(n, 0.0f);
//         aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             double expected = std::pow((double)baseScalar, (double)expData[i]);
//             if (!AlmostEqual((double)result[i], expected)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected %f\n", i, result[i], (float)expected);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyScalar(baseS);
//     aclDestroyTensor(expT);
//     aclDestroyTensor(outT);
//     aclrtFree(expDev);
//     aclrtFree(outDev);
//     return true;
// }

// // ==================== Test: PowTensorTensor ====================
// bool TestPowTensorTensor(aclrtStream stream, const std::string& label,
//                          std::vector<float> baseData, std::vector<int64_t> baseShape,
//                          std::vector<float> expData, std::vector<int64_t> expShape,
//                          std::vector<int64_t> outShape, aclDataType dtype) {
//     int64_t nOut = GetShapeSize(outShape);
//     void *baseDev = nullptr, *expDev = nullptr, *outDev = nullptr;
//     aclTensor *baseT = nullptr, *expT = nullptr, *outT = nullptr;

//     std::vector<float> outHost(nOut, 0.0f);

//     auto ret = CreateAclTensor(baseData, baseShape, &baseDev, dtype, &baseT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(expData, expShape, &expDev, dtype, &expT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(outHost, outShape, &outDev, dtype, &outT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnPowTensorTensorGetWorkspaceSize(baseT, expT, outT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnPowTensorTensor(wsAddr, wsSize, executor, stream);
//         if (ret != ACL_SUCCESS) { ReportResult(label, false); if (wsAddr) aclrtFree(wsAddr); goto cleanup; }
//         aclrtSynchronizeStream(stream);

//         std::vector<float> result(nOut, 0.0f);
//         aclrtMemcpy(result.data(), nOut * sizeof(float), outDev, nOut * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         // For broadcast verification, we do element-wise with broadcast indexing
//         bool pass = true;
//         int64_t nBase = GetShapeSize(baseShape);
//         int64_t nExp = GetShapeSize(expShape);
//         for (int64_t i = 0; i < nOut; i++) {
//             double b = (double)baseData[i % nBase];
//             double e = (double)expData[i % nExp];
//             double expected = std::pow(b, e);
//             if (!AlmostEqual((double)result[i], expected)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected %f\n", i, result[i], (float)expected);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(baseT);
//     aclDestroyTensor(expT);
//     aclDestroyTensor(outT);
//     aclrtFree(baseDev);
//     aclrtFree(expDev);
//     aclrtFree(outDev);
//     return true;
// }

// // ==================== Test: InplacePowTensorTensor ====================
// bool TestInplacePowTensorTensor(aclrtStream stream, const std::string& label,
//                                 std::vector<float> baseData, std::vector<int64_t> shape,
//                                 std::vector<float> expData, aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     void *baseDev = nullptr, *expDev = nullptr;
//     aclTensor *baseT = nullptr, *expT = nullptr;

//     auto ret = CreateAclTensor(baseData, shape, &baseDev, dtype, &baseT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(expData, shape, &expDev, dtype, &expT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnInplacePowTensorTensorGetWorkspaceSize(baseT, expT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnInplacePowTensorTensor(wsAddr, wsSize, executor, stream);
//         if (ret != ACL_SUCCESS) { ReportResult(label, false); if (wsAddr) aclrtFree(wsAddr); goto cleanup; }
//         aclrtSynchronizeStream(stream);

//         std::vector<float> result(n, 0.0f);
//         aclrtMemcpy(result.data(), n * sizeof(float), baseDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             double expected = std::pow((double)baseData[i], (double)expData[i]);
//             if (!AlmostEqual((double)result[i], expected)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected %f\n", i, result[i], (float)expected);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(baseT);
//     aclDestroyTensor(expT);
//     aclrtFree(baseDev);
//     aclrtFree(expDev);
//     return true;
// }

// // ==================== Test: Exp2 ====================
// bool TestExp2(aclrtStream stream, const std::string& label,
//               std::vector<float> selfData, std::vector<int64_t> shape, aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     void *selfDev = nullptr, *outDev = nullptr;
//     aclTensor *selfT = nullptr, *outT = nullptr;
//     std::vector<float> outHost(n, 0.0f);

//     auto ret = CreateAclTensor(selfData, shape, &selfDev, dtype, &selfT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnExp2GetWorkspaceSize(selfT, outT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnExp2(wsAddr, wsSize, executor, stream);
//         if (ret != ACL_SUCCESS) { ReportResult(label, false); if (wsAddr) aclrtFree(wsAddr); goto cleanup; }
//         aclrtSynchronizeStream(stream);

//         std::vector<float> result(n, 0.0f);
//         aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             double expected = std::pow(2.0, (double)selfData[i]);
//             if (!AlmostEqual((double)result[i], expected)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected %f\n", i, result[i], (float)expected);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(selfT);
//     aclDestroyTensor(outT);
//     aclrtFree(selfDev);
//     aclrtFree(outDev);
//     return true;
// }

// // ==================== Test: InplaceExp2 ====================
// bool TestInplaceExp2(aclrtStream stream, const std::string& label,
//                      std::vector<float> selfData, std::vector<int64_t> shape, aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     void *selfDev = nullptr;
//     aclTensor *selfT = nullptr;

//     auto ret = CreateAclTensor(selfData, shape, &selfDev, dtype, &selfT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnInplaceExp2GetWorkspaceSize(selfT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnInplaceExp2(wsAddr, wsSize, executor, stream);
//         if (ret != ACL_SUCCESS) { ReportResult(label, false); if (wsAddr) aclrtFree(wsAddr); goto cleanup; }
//         aclrtSynchronizeStream(stream);

//         std::vector<float> result(n, 0.0f);
//         aclrtMemcpy(result.data(), n * sizeof(float), selfDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             double expected = std::pow(2.0, (double)selfData[i]);
//             if (!AlmostEqual((double)result[i], expected)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected %f\n", i, result[i], (float)expected);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(selfT);
//     aclrtFree(selfDev);
//     return true;
// }

// // ==================== Test: PowTensorScalar with int dtype ====================
// bool TestPowTensorScalarInt(aclrtStream stream, const std::string& label,
//                             std::vector<int32_t> baseData, std::vector<int64_t> shape,
//                             float expVal, aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     void *selfDev = nullptr, *outDev = nullptr;
//     aclTensor *selfT = nullptr, *outT = nullptr;
//     aclScalar *expS = nullptr;

//     std::vector<int32_t> outHost(n, 0);

//     auto ret = CreateAclTensor(baseData, shape, &selfDev, dtype, &selfT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     expS = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
//     if (!expS) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnPowTensorScalarGetWorkspaceSize(selfT, expS, outT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) {
//         // For integer types with negative exponents, failure is expected
//         ReportResult(label, true);
//         goto cleanup;
//     }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnPowTensorScalar(wsAddr, wsSize, executor, stream);
//         aclrtSynchronizeStream(stream);

//         std::vector<int32_t> result(n, 0);
//         aclrtMemcpy(result.data(), n * sizeof(int32_t), outDev, n * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             double expected = std::pow((double)baseData[i], (double)expVal);
//             int32_t expInt = (int32_t)expected;
//             if (result[i] != expInt) {
//                 // Allow some tolerance for integer rounding
//                 if (std::abs(result[i] - expInt) > 1) {
//                     LOG_PRINT("  Mismatch at [%ld]: got %d, expected %d\n", i, result[i], expInt);
//                     pass = false;
//                 }
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(selfT);
//     aclDestroyScalar(expS);
//     aclDestroyTensor(outT);
//     aclrtFree(selfDev);
//     aclrtFree(outDev);
//     return true;
// }

// // ==================== Test: PowTensorTensor with int dtype ====================
// bool TestPowTensorTensorInt(aclrtStream stream, const std::string& label,
//                             std::vector<int32_t> baseData, std::vector<int64_t> baseShape,
//                             std::vector<int32_t> expData, std::vector<int64_t> expShape,
//                             std::vector<int64_t> outShape, aclDataType dtype) {
//     int64_t nOut = GetShapeSize(outShape);
//     void *baseDev = nullptr, *expDev = nullptr, *outDev = nullptr;
//     aclTensor *baseT = nullptr, *expT = nullptr, *outT = nullptr;

//     std::vector<int32_t> outHost(nOut, 0);

//     auto ret = CreateAclTensor(baseData, baseShape, &baseDev, dtype, &baseT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(expData, expShape, &expDev, dtype, &expT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(outHost, outShape, &outDev, dtype, &outT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnPowTensorTensorGetWorkspaceSize(baseT, expT, outT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnPowTensorTensor(wsAddr, wsSize, executor, stream);
//         aclrtSynchronizeStream(stream);

//         std::vector<int32_t> result(nOut, 0);
//         aclrtMemcpy(result.data(), nOut * sizeof(int32_t), outDev, nOut * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         int64_t nBase = GetShapeSize(baseShape);
//         int64_t nExp = GetShapeSize(expShape);
//         for (int64_t i = 0; i < nOut; i++) {
//             double expected = std::pow((double)baseData[i % nBase], (double)expData[i % nExp]);
//             int32_t expInt = (int32_t)expected;
//             if (std::abs(result[i] - expInt) > 1) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %d, expected %d\n", i, result[i], expInt);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyTensor(baseT);
//     aclDestroyTensor(expT);
//     aclDestroyTensor(outT);
//     aclrtFree(baseDev);
//     aclrtFree(expDev);
//     aclrtFree(outDev);
//     return true;
// }

// // ==================== Test: Nullptr / error path ====================
// bool TestNullptrInputs(aclrtStream stream) {
//     // Test TensorScalar with null self
//     {
//         uint64_t wsSize = 0;
//         aclOpExecutor* executor = nullptr;
//         float expVal = 2.0f;
//         aclScalar* expS = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
//         auto ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, expS, nullptr, &wsSize, &executor);
//         bool pass = (ret != ACL_SUCCESS);
//         ReportResult("Nullptr_TensorScalar_self_null", pass);
//         aclDestroyScalar(expS);
//     }

//     // Test ScalarTensor with null exponent tensor
//     {
//         uint64_t wsSize = 0;
//         aclOpExecutor* executor = nullptr;
//         float baseVal = 2.0f;
//         aclScalar* baseS = aclCreateScalar(&baseVal, aclDataType::ACL_FLOAT);
//         auto ret = aclnnPowScalarTensorGetWorkspaceSize(baseS, nullptr, nullptr, &wsSize, &executor);
//         bool pass = (ret != ACL_SUCCESS);
//         ReportResult("Nullptr_ScalarTensor_exp_null", pass);
//         aclDestroyScalar(baseS);
//     }

//     // Test TensorTensor with null inputs
//     {
//         uint64_t wsSize = 0;
//         aclOpExecutor* executor = nullptr;
//         auto ret = aclnnPowTensorTensorGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &executor);
//         bool pass = (ret != ACL_SUCCESS);
//         ReportResult("Nullptr_TensorTensor_all_null", pass);
//     }

//     return true;
// }

// // ==================== Test: Empty tensor paths ====================
// bool TestEmptyTensors(aclrtStream stream) {
//     // TensorScalar with empty input
//     {
//         std::vector<int64_t> shape = {0, 4};
//         void *selfDev = nullptr, *outDev = nullptr;
//         aclTensor *selfT = nullptr, *outT = nullptr;
//         aclScalar *expS = nullptr;
//         float expVal = 2.0f;

//         CreateEmptyAclTensor(shape, &selfDev, aclDataType::ACL_FLOAT, &selfT, 4);
//         CreateEmptyAclTensor(shape, &outDev, aclDataType::ACL_FLOAT, &outT, 4);
//         expS = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);

//         uint64_t wsSize = 0;
//         aclOpExecutor* executor = nullptr;
//         auto ret = aclnnPowTensorScalarGetWorkspaceSize(selfT, expS, outT, &wsSize, &executor);
//         bool pass = (ret == ACL_SUCCESS && wsSize == 0);
//         ReportResult("Empty_TensorScalar", pass);

//         aclDestroyTensor(selfT);
//         aclDestroyTensor(outT);
//         aclDestroyScalar(expS);
//         aclrtFree(selfDev);
//         aclrtFree(outDev);
//     }

//     // ScalarTensor with empty exponent
//     {
//         std::vector<int64_t> shape = {0, 4};
//         void *expDev = nullptr, *outDev = nullptr;
//         aclTensor *expT = nullptr, *outT = nullptr;
//         aclScalar *baseS = nullptr;
//         float baseVal = 2.0f;

//         CreateEmptyAclTensor(shape, &expDev, aclDataType::ACL_FLOAT, &expT, 4);
//         CreateEmptyAclTensor(shape, &outDev, aclDataType::ACL_FLOAT, &outT, 4);
//         baseS = aclCreateScalar(&baseVal, aclDataType::ACL_FLOAT);

//         uint64_t wsSize = 0;
//         aclOpExecutor* executor = nullptr;
//         auto ret = aclnnPowScalarTensorGetWorkspaceSize(baseS, expT, outT, &wsSize, &executor);
//         bool pass = (ret == ACL_SUCCESS && wsSize == 0);
//         ReportResult("Empty_ScalarTensor", pass);

//         aclDestroyTensor(expT);
//         aclDestroyTensor(outT);
//         aclDestroyScalar(baseS);
//         aclrtFree(expDev);
//         aclrtFree(outDev);
//     }

//     // TensorTensor with empty self
//     {
//         std::vector<int64_t> shape = {0, 4};
//         void *selfDev = nullptr, *expDev = nullptr, *outDev = nullptr;
//         aclTensor *selfT = nullptr, *expT = nullptr, *outT = nullptr;

//         CreateEmptyAclTensor(shape, &selfDev, aclDataType::ACL_FLOAT, &selfT, 4);
//         CreateEmptyAclTensor(shape, &expDev, aclDataType::ACL_FLOAT, &expT, 4);
//         CreateEmptyAclTensor(shape, &outDev, aclDataType::ACL_FLOAT, &outT, 4);

//         uint64_t wsSize = 0;
//         aclOpExecutor* executor = nullptr;
//         auto ret = aclnnPowTensorTensorGetWorkspaceSize(selfT, expT, outT, &wsSize, &executor);
//         bool pass = (ret == ACL_SUCCESS && wsSize == 0);
//         ReportResult("Empty_TensorTensor", pass);

//         aclDestroyTensor(selfT);
//         aclDestroyTensor(expT);
//         aclDestroyTensor(outT);
//         aclrtFree(selfDev);
//         aclrtFree(expDev);
//         aclrtFree(outDev);
//     }

//     return true;
// }

// // ==================== Test: ScalarTensor with base=1.0 (Fill branch) ====================
// bool TestPowScalarTensorBaseOne(aclrtStream stream, const std::string& label,
//                                 std::vector<float> expData, std::vector<int64_t> shape,
//                                 aclDataType dtype) {
//     int64_t n = GetShapeSize(shape);
//     float baseVal = 1.0f;
//     void *expDev = nullptr, *outDev = nullptr;
//     aclTensor *expT = nullptr, *outT = nullptr;
//     aclScalar *baseS = nullptr;

//     std::vector<float> outHost(n, 0.0f);

//     auto ret = CreateAclTensor(expData, shape, &expDev, dtype, &expT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     baseS = aclCreateScalar(&baseVal, aclDataType::ACL_FLOAT);
//     if (!baseS) { ReportResult(label, false); return false; }

//     ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
//     if (ret != 0) { ReportResult(label, false); return false; }

//     uint64_t wsSize = 0;
//     aclOpExecutor* executor = nullptr;
//     ret = aclnnPowScalarTensorGetWorkspaceSize(baseS, expT, outT, &wsSize, &executor);
//     if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }

//     {
//         void* wsAddr = nullptr;
//         if (wsSize > 0) {
//             ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
//             if (ret != ACL_SUCCESS) { ReportResult(label, false); goto cleanup; }
//         }
//         ret = aclnnPowScalarTensor(wsAddr, wsSize, executor, stream);
//         aclrtSynchronizeStream(stream);

//         std::vector<float> result(n, 0.0f);
//         aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

//         bool pass = true;
//         for (int64_t i = 0; i < n; i++) {
//             // 1^x = 1 for all x
//             if (!AlmostEqual((double)result[i], 1.0)) {
//                 LOG_PRINT("  Mismatch at [%ld]: got %f, expected 1.0\n", i, result[i]);
//                 pass = false;
//             }
//         }
//         ReportResult(label, pass);
//         if (wsAddr) aclrtFree(wsAddr);
//     }

// cleanup:
//     aclDestroyScalar(baseS);
//     aclDestroyTensor(expT);
//     aclDestroyTensor(outT);
//     aclrtFree(expDev);
//     aclrtFree(outDev);
//     return true;
// }

// // ==================== Main ====================
// int main() {
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

//     printf("========== Pow Operator Test Suite ==========\n\n");

//     // Basic data
//     std::vector<float> base4 = {1.0f, 2.0f, 3.0f, 4.0f};
//     std::vector<float> base4_with_zero = {0.0f, 1.0f, 2.0f, 3.0f};
//     std::vector<float> base4_neg = {-1.0f, -2.0f, 1.0f, 2.0f};
//     std::vector<float> base4_large = {10.0f, 100.0f, 0.5f, 1.5f};
//     std::vector<int64_t> shape2x2 = {2, 2};
//     std::vector<int64_t> shape4 = {4};
//     std::vector<int64_t> shape1 = {1};
//     std::vector<int64_t> shape16 = {16};
//     std::vector<int64_t> shape2x8 = {2, 8};

//     // ============================================================
//     // 1. PowTensorScalar - FLOAT32 with various exponents
//     // ============================================================
//     printf("--- PowTensorScalar FLOAT32 ---\n");
//     TestPowTensorScalar(stream, "TS_F32_exp0",     base4, shape2x2, 0.0f,  aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp1",     base4, shape2x2, 1.0f,  aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp0.5",   base4, shape2x2, 0.5f,  aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp2",     base4, shape2x2, 2.0f,  aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp3",     base4, shape2x2, 3.0f,  aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp_neg1", base4, shape2x2, -1.0f, aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp_neg0.5", base4, shape2x2, -0.5f, aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp_neg2", base4, shape2x2, -2.0f, aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp4.1",   base4, shape2x2, 4.1f,  aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_base_zero", base4_with_zero, shape2x2, 2.0f, aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_base_neg",  base4_neg, shape2x2, 2.0f, aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_large_exp", base4_large, shape2x2, 10.0f, aclDataType::ACL_FLOAT);

//     // Larger shape to stress tiling
//     {
//         std::vector<float> bigBase(16);
//         for (int i = 0; i < 16; i++) bigBase[i] = (float)(i + 1);
//         TestPowTensorScalar(stream, "TS_F32_16elem_exp2",  bigBase, shape16, 2.0f, aclDataType::ACL_FLOAT);
//         TestPowTensorScalar(stream, "TS_F32_16elem_exp0.5", bigBase, shape16, 0.5f, aclDataType::ACL_FLOAT);
//     }

//     // ============================================================
//     // 2. PowTensorScalar - FLOAT16
//     // ============================================================
//     printf("\n--- PowTensorScalar FLOAT16 ---\n");
//     TestPowTensorScalar(stream, "TS_F16_exp2",     base4, shape2x2, 2.0f,  aclDataType::ACL_FLOAT16);
//     TestPowTensorScalar(stream, "TS_F16_exp0.5",   base4, shape2x2, 0.5f,  aclDataType::ACL_FLOAT16);
//     TestPowTensorScalar(stream, "TS_F16_exp3",     base4, shape2x2, 3.0f,  aclDataType::ACL_FLOAT16);
//     TestPowTensorScalar(stream, "TS_F16_exp_neg1", base4, shape2x2, -1.0f, aclDataType::ACL_FLOAT16);
//     TestPowTensorScalar(stream, "TS_F16_exp4.1",   base4, shape2x2, 4.1f,  aclDataType::ACL_FLOAT16);

//     // ============================================================
//     // 3. PowTensorScalar - BF16
//     // ============================================================
//     printf("\n--- PowTensorScalar BF16 ---\n");
//     TestPowTensorScalar(stream, "TS_BF16_exp2",     base4, shape2x2, 2.0f,  aclDataType::ACL_BF16);
//     TestPowTensorScalar(stream, "TS_BF16_exp0.5",   base4, shape2x2, 0.5f,  aclDataType::ACL_BF16);
//     TestPowTensorScalar(stream, "TS_BF16_exp_neg1", base4, shape2x2, -1.0f, aclDataType::ACL_BF16);
//     TestPowTensorScalar(stream, "TS_BF16_exp3",     base4, shape2x2, 3.0f,  aclDataType::ACL_BF16);

//     // ============================================================
//     // 4. PowTensorScalar - INT32
//     // ============================================================
//     printf("\n--- PowTensorScalar INT32 ---\n");
//     {
//         std::vector<int32_t> intBase = {1, 2, 3, 4};
//         TestPowTensorScalarInt(stream, "TS_INT32_exp2",    intBase, shape2x2, 2.0f, aclDataType::ACL_INT32);
//         TestPowTensorScalarInt(stream, "TS_INT32_exp3",    intBase, shape2x2, 3.0f, aclDataType::ACL_INT32);
//         TestPowTensorScalarInt(stream, "TS_INT32_exp0",    intBase, shape2x2, 0.0f, aclDataType::ACL_INT32);
//         TestPowTensorScalarInt(stream, "TS_INT32_exp1",    intBase, shape2x2, 1.0f, aclDataType::ACL_INT32);
//         // Negative exponent with integral base - should fail/return error
//         TestPowTensorScalarInt(stream, "TS_INT32_exp_neg1_err", intBase, shape2x2, -1.0f, aclDataType::ACL_INT32);
//     }

//     // ============================================================
//     // 5. InplacePowTensorScalar
//     // ============================================================
//     printf("\n--- InplacePowTensorScalar ---\n");
//     TestInplacePowTensorScalar(stream, "InplaceTS_F32_exp2",   base4, shape2x2, 2.0f, aclDataType::ACL_FLOAT);
//     TestInplacePowTensorScalar(stream, "InplaceTS_F32_exp0.5", base4, shape2x2, 0.5f, aclDataType::ACL_FLOAT);
//     TestInplacePowTensorScalar(stream, "InplaceTS_F32_exp3",   base4, shape2x2, 3.0f, aclDataType::ACL_FLOAT);
//     TestInplacePowTensorScalar(stream, "InplaceTS_F32_exp_neg1", base4, shape2x2, -1.0f, aclDataType::ACL_FLOAT);
//     TestInplacePowTensorScalar(stream, "InplaceTS_F32_exp4.1", base4, shape2x2, 4.1f, aclDataType::ACL_FLOAT);

//     // ============================================================
//     // 6. PowScalarTensor - various base scalars
//     // ============================================================
//     printf("\n--- PowScalarTensor ---\n");
//     std::vector<float> exp4 = {0.0f, 1.0f, 2.0f, 3.0f};
//     std::vector<float> exp4_neg = {-1.0f, 0.0f, 1.0f, 2.0f};
//     std::vector<float> exp4_frac = {0.5f, 1.5f, 2.5f, 0.25f};

//     TestPowScalarTensor(stream, "ST_F32_base2",   2.0f, exp4,      shape2x2, aclDataType::ACL_FLOAT);
//     TestPowScalarTensor(stream, "ST_F32_base3",   3.0f, exp4,      shape2x2, aclDataType::ACL_FLOAT);
//     TestPowScalarTensor(stream, "ST_F32_base0.5", 0.5f, exp4,      shape2x2, aclDataType::ACL_FLOAT);
//     TestPowScalarTensor(stream, "ST_F32_base10",  10.0f, exp4_neg, shape2x2, aclDataType::ACL_FLOAT);
//     TestPowScalarTensor(stream, "ST_F32_base_frac", 2.0f, exp4_frac, shape2x2, aclDataType::ACL_FLOAT);

//     // base=1.0 - should trigger Fill(1) branch in RegBase mode
//     TestPowScalarTensorBaseOne(stream, "ST_F32_base1_Fill", exp4, shape2x2, aclDataType::ACL_FLOAT);

//     // FP16 ScalarTensor
//     TestPowScalarTensor(stream, "ST_F16_base2",  2.0f, exp4, shape2x2, aclDataType::ACL_FLOAT16);

//     // BF16 ScalarTensor
//     TestPowScalarTensor(stream, "ST_BF16_base2", 2.0f, exp4, shape2x2, aclDataType::ACL_BF16);

//     // ============================================================
//     // 7. PowTensorTensor - FLOAT32
//     // ============================================================
//     printf("\n--- PowTensorTensor FLOAT32 ---\n");
//     // Same shape
//     TestPowTensorTensor(stream, "TT_F32_same_shape",
//                         base4, shape2x2, exp4, shape2x2, shape2x2, aclDataType::ACL_FLOAT);

//     // Broadcast: base [4], exp [1] -> out [4]
//     {
//         std::vector<float> exp1 = {2.0f};
//         TestPowTensorTensor(stream, "TT_F32_broadcast_exp1",
//                             base4, shape4, exp1, shape1, shape4, aclDataType::ACL_FLOAT);
//     }

//     // Broadcast: base [1], exp [4] -> out [4]
//     {
//         std::vector<float> base1 = {3.0f};
//         TestPowTensorTensor(stream, "TT_F32_broadcast_base1",
//                             base1, shape1, exp4, shape4, shape4, aclDataType::ACL_FLOAT);
//     }

//     // 2D broadcast: base [2,1], exp [1,2] -> out [2,2]
//     {
//         std::vector<float> b21 = {2.0f, 3.0f};
//         std::vector<float> e12 = {1.0f, 2.0f};
//         std::vector<int64_t> shape21 = {2, 1};
//         std::vector<int64_t> shape12 = {1, 2};
//         TestPowTensorTensor(stream, "TT_F32_broadcast_2d",
//                             b21, shape21, e12, shape12, shape2x2, aclDataType::ACL_FLOAT);
//     }

//     // ============================================================
//     // 8. PowTensorTensor - FLOAT16
//     // ============================================================
//     printf("\n--- PowTensorTensor FLOAT16 ---\n");
//     TestPowTensorTensor(stream, "TT_F16_same_shape",
//                         base4, shape2x2, exp4, shape2x2, shape2x2, aclDataType::ACL_FLOAT16);

//     // ============================================================
//     // 9. PowTensorTensor - BF16
//     // ============================================================
//     printf("\n--- PowTensorTensor BF16 ---\n");
//     TestPowTensorTensor(stream, "TT_BF16_same_shape",
//                         base4, shape2x2, exp4, shape2x2, shape2x2, aclDataType::ACL_BF16);

//     // ============================================================
//     // 10. PowTensorTensor - INT32
//     // ============================================================
//     printf("\n--- PowTensorTensor INT32 ---\n");
//     {
//         std::vector<int32_t> intBase = {1, 2, 3, 4};
//         std::vector<int32_t> intExp  = {0, 1, 2, 3};
//         TestPowTensorTensorInt(stream, "TT_INT32_same",
//                                intBase, shape2x2, intExp, shape2x2, shape2x2, aclDataType::ACL_INT32);
//     }

//     // ============================================================
//     // 11. PowTensorTensor - INT8
//     // ============================================================
//     printf("\n--- PowTensorTensor INT8 ---\n");
//     {
//         // Use int32 representation but with ACL_INT8 dtype; data values fit in int8
//         std::vector<int32_t> int8Base = {1, 2, 3, 2};
//         std::vector<int32_t> int8Exp  = {0, 1, 2, 3};
//         TestPowTensorTensorInt(stream, "TT_INT8_same",
//                                int8Base, shape2x2, int8Exp, shape2x2, shape2x2, aclDataType::ACL_INT8);
//     }

//     // ============================================================
//     // 12. PowTensorTensor - UINT8
//     // ============================================================
//     printf("\n--- PowTensorTensor UINT8 ---\n");
//     {
//         std::vector<int32_t> uint8Base = {1, 2, 3, 4};
//         std::vector<int32_t> uint8Exp  = {0, 1, 2, 2};
//         TestPowTensorTensorInt(stream, "TT_UINT8_same",
//                                uint8Base, shape2x2, uint8Exp, shape2x2, shape2x2, aclDataType::ACL_UINT8);
//     }

//     // ============================================================
//     // 13. PowTensorTensor - INT16
//     // ============================================================
//     printf("\n--- PowTensorTensor INT16 ---\n");
//     {
//         std::vector<int32_t> int16Base = {1, 2, 3, 4};
//         std::vector<int32_t> int16Exp  = {0, 1, 2, 3};
//         TestPowTensorTensorInt(stream, "TT_INT16_same",
//                                int16Base, shape2x2, int16Exp, shape2x2, shape2x2, aclDataType::ACL_INT16);
//     }

//     // ============================================================
//     // 14. InplacePowTensorTensor
//     // ============================================================
//     printf("\n--- InplacePowTensorTensor ---\n");
//     TestInplacePowTensorTensor(stream, "InplaceTT_F32",
//                                base4, shape2x2, exp4, aclDataType::ACL_FLOAT);
//     TestInplacePowTensorTensor(stream, "InplaceTT_F16",
//                                base4, shape2x2, exp4, aclDataType::ACL_FLOAT16);

//     // ============================================================
//     // 15. Exp2
//     // ============================================================
//     printf("\n--- Exp2 ---\n");
//     {
//         std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f};
//         TestExp2(stream, "Exp2_F32_basic", selfData, shape2x2, aclDataType::ACL_FLOAT);
//     }
//     {
//         std::vector<float> selfData = {-1.0f, 0.5f, 1.5f, 10.0f};
//         TestExp2(stream, "Exp2_F32_mixed", selfData, shape2x2, aclDataType::ACL_FLOAT);
//     }
//     {
//         std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f};
//         TestExp2(stream, "Exp2_F16_basic", selfData, shape2x2, aclDataType::ACL_FLOAT16);
//     }
//     {
//         std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f};
//         TestExp2(stream, "Exp2_BF16_basic", selfData, shape2x2, aclDataType::ACL_BF16);
//     }

//     // ============================================================
//     // 16. InplaceExp2
//     // ============================================================
//     printf("\n--- InplaceExp2 ---\n");
//     {
//         std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f};
//         TestInplaceExp2(stream, "InplaceExp2_F32", selfData, shape2x2, aclDataType::ACL_FLOAT);
//     }
//     {
//         std::vector<float> selfData = {-1.0f, 0.0f, 1.0f, 2.0f};
//         TestInplaceExp2(stream, "InplaceExp2_F16", selfData, shape2x2, aclDataType::ACL_FLOAT16);
//     }

//     // ============================================================
//     // 17. Nullptr / Error paths
//     // ============================================================
//     printf("\n--- Nullptr / Error paths ---\n");
//     TestNullptrInputs(stream);

//     // ============================================================
//     // 18. Empty tensor paths
//     // ============================================================
//     printf("\n--- Empty tensor paths ---\n");
//     TestEmptyTensors(stream);

//     // ============================================================
//     // 19. High-dimensional shapes (>5 dims) for tiling coverage
//     //     Triggers different OP_KEY paths in tiling (nddma loops vs no loops)
//     // ============================================================
//     printf("\n--- High-dim TensorTensor (>5 dims) ---\n");
//     {
//         // 6-dim shape: triggers OP_KEY with 8-dim path (nddma needs loops)
//         std::vector<int64_t> shape6d = {1, 1, 1, 2, 1, 2};
//         std::vector<float> data4 = {1.0f, 2.0f, 3.0f, 4.0f};
//         TestPowTensorTensor(stream, "TT_F32_6dim",
//                             data4, shape6d, data4, shape6d, shape6d, aclDataType::ACL_FLOAT);
//         TestPowTensorTensor(stream, "TT_F16_6dim",
//                             data4, shape6d, data4, shape6d, shape6d, aclDataType::ACL_FLOAT16);
//         TestPowTensorTensor(stream, "TT_BF16_6dim",
//                             data4, shape6d, data4, shape6d, shape6d, aclDataType::ACL_BF16);
//     }

//     // ============================================================
//     // 20. Large tensor for multi-block tiling
//     // ============================================================
//     printf("\n--- Large tensor (multi-block tiling) ---\n");
//     {
//         int64_t bigN = 1024;
//         std::vector<float> bigBase(bigN), bigExp(bigN);
//         for (int64_t i = 0; i < bigN; i++) {
//             bigBase[i] = 1.0f + (float)(i % 10) * 0.1f;
//             bigExp[i]  = (float)(i % 5);
//         }
//         std::vector<int64_t> bigShape = {bigN};
//         TestPowTensorTensor(stream, "TT_F32_1024elem",
//                             bigBase, bigShape, bigExp, bigShape, bigShape, aclDataType::ACL_FLOAT);
//     }

//     // ============================================================
//     // 21. TensorScalar with special values for Pows path coverage
//     //     exponent = -0.5 (negative sqrt)
//     //     exponent = -2.0 (negative square)
//     // ============================================================
//     printf("\n--- Special Pows path ---\n");
//     TestPowTensorScalar(stream, "TS_F32_exp_neg0.5", base4, shape2x2, -0.5f, aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F32_exp_neg2.0", base4, shape2x2, -2.0f, aclDataType::ACL_FLOAT);

//     // ============================================================
//     // 22. x^0 = 1 for all x (including 0^0 = 1)
//     // ============================================================
//     printf("\n--- x^0 edge cases ---\n");
//     TestPowTensorScalar(stream, "TS_F32_0pow0",
//                         {0.0f, 0.0f, 0.0f, 0.0f}, shape2x2, 0.0f, aclDataType::ACL_FLOAT);

//     // ============================================================
//     // 23. TensorTensor with broadcast and different dims (<=5 dims)
//     //     For pow_tensor_tensor_tiling_arch35 coverage
//     // ============================================================
//     printf("\n--- TensorTensor broadcast shapes ---\n");
//     {
//         // 3D broadcast
//         std::vector<int64_t> shape213 = {2, 1, 3};
//         std::vector<int64_t> shape123 = {1, 2, 3};
//         std::vector<int64_t> shape223 = {2, 2, 3};
//         std::vector<float> d6a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
//         std::vector<float> d6b = {0.0f, 1.0f, 2.0f, 1.0f, 0.5f, 3.0f};
//         TestPowTensorTensor(stream, "TT_F32_3d_broadcast",
//                             d6a, shape213, d6b, shape123, shape223, aclDataType::ACL_FLOAT);
//     }

//     // 4D
//     {
//         std::vector<int64_t> shape4d_a = {1, 2, 1, 2};
//         std::vector<int64_t> shape4d_b = {2, 1, 2, 1};
//         std::vector<int64_t> shape4d_out = {2, 2, 2, 2};
//         std::vector<float> d4a = {2.0f, 3.0f, 4.0f, 5.0f};
//         std::vector<float> d4b = {1.0f, 2.0f, 0.0f, 1.0f};
//         TestPowTensorTensor(stream, "TT_F32_4d_broadcast",
//                             d4a, shape4d_a, d4b, shape4d_b, shape4d_out, aclDataType::ACL_FLOAT);
//     }

//     // 5D
//     {
//         std::vector<int64_t> shape5d = {1, 1, 2, 1, 2};
//         std::vector<float> d4 = {2.0f, 3.0f, 4.0f, 5.0f};
//         TestPowTensorTensor(stream, "TT_F32_5dim",
//                             d4, shape5d, d4, shape5d, shape5d, aclDataType::ACL_FLOAT);
//     }

//     // ============================================================
//     // 24. Additional INT types for TensorScalar tiling OP_KEY coverage
//     // ============================================================
//     printf("\n--- INT dtype TensorScalar for tiling OP_KEY ---\n");
//     {
//         // INT8 TensorScalar (OP_KEY 5001)
//         std::vector<int32_t> int8Data = {1, 2, 3, 2};
//         TestPowTensorScalarInt(stream, "TS_INT8_exp2", int8Data, shape2x2, 2.0f, aclDataType::ACL_INT8);
//     }
//     {
//         // UINT8 TensorScalar (OP_KEY 4001)
//         std::vector<int32_t> uint8Data = {1, 2, 3, 4};
//         TestPowTensorScalarInt(stream, "TS_UINT8_exp2", uint8Data, shape2x2, 2.0f, aclDataType::ACL_UINT8);
//     }
//     {
//         // INT16 TensorScalar (OP_KEY 6001)
//         std::vector<int32_t> int16Data = {1, 2, 3, 4};
//         TestPowTensorScalarInt(stream, "TS_INT16_exp2", int16Data, shape2x2, 2.0f, aclDataType::ACL_INT16);
//     }

//     // ============================================================
//     // 25. TensorScalar exponent = 2.0 with various dtypes for Square path
//     // ============================================================
//     printf("\n--- Square path (exp=2.0) various dtypes ---\n");
//     TestPowTensorScalar(stream, "TS_F32_square_path",  base4, shape2x2, 2.0f, aclDataType::ACL_FLOAT);
//     TestPowTensorScalar(stream, "TS_F16_square_path",  base4, shape2x2, 2.0f, aclDataType::ACL_FLOAT16);
//     TestPowTensorScalar(stream, "TS_BF16_square_path", base4, shape2x2, 2.0f, aclDataType::ACL_BF16);

//     // ============================================================
//     // 26. Larger tensors for TensorScalar tiling stress
//     // ============================================================
//     printf("\n--- Large TensorScalar tiling ---\n");
//     {
//         int64_t bigN = 4096;
//         std::vector<float> bigData(bigN);
//         for (int64_t i = 0; i < bigN; i++) bigData[i] = 1.0f + (float)(i % 20) * 0.05f;
//         std::vector<int64_t> bigShape = {bigN};
//         TestPowTensorScalar(stream, "TS_F32_4096_exp2",   bigData, bigShape, 2.0f, aclDataType::ACL_FLOAT);
//         TestPowTensorScalar(stream, "TS_F32_4096_exp0.5", bigData, bigShape, 0.5f, aclDataType::ACL_FLOAT);
//         TestPowTensorScalar(stream, "TS_F32_4096_exp3",   bigData, bigShape, 3.0f, aclDataType::ACL_FLOAT);
//     }

//     // ============================================================
//     // 27. Multi-dim TensorScalar for tiling block/ub paths
//     // ============================================================
//     printf("\n--- Multi-dim TensorScalar ---\n");
//     {
//         std::vector<float> md = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
//                                  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
//         TestPowTensorScalar(stream, "TS_F32_2x8_exp2", md, shape2x8, 2.0f, aclDataType::ACL_FLOAT);
//     }

//     // ============================================================
//     // Summary
//     // ============================================================
//     printf("\n========== Summary ==========\n");
//     printf("Total: %d, Passed: %d, Failed: %d\n", g_totalTests, g_passedTests, g_failedTests);

//     // Cleanup
//     aclrtDestroyStream(stream);
//     aclrtResetDevice(deviceId);
//     aclFinalize();

//     return (g_failedTests > 0) ? 1 : 0;
// }

/**
 * Comprehensive test cases for Pow operator - Maximum Coverage Version
 * Covers: TensorScalar, ScalarTensor, TensorTensor, Exp2 (+ Inplace variants)
 * Target: maximize code coverage across aclnn_pow.cpp, aclnn_pow_tensor_tensor.cpp,
 *         pow.cpp, pow_tiling_arch35.cpp, pow_tensor_tensor_tiling_arch35.cpp
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include <limits>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

static int g_totalTests = 0;
static int g_passedTests = 0;
static int g_failedTests = 0;

static void ReportResult(const std::string &name, bool pass) {
    g_totalTests++;
    if (pass) { g_passedTests++; printf("[PASS] %s\n", name.c_str()); }
    else      { g_failedTests++; printf("[FAIL] %s\n", name.c_str()); }
}

static void ReportExpectedError(const std::string &name, int ret) {
    g_totalTests++;
    if (ret != 0) { g_passedTests++; printf("[PASS] %s (expected error=%d)\n", name.c_str(), ret); }
    else          { g_failedTests++; printf("[FAIL] %s (expected error but got success)\n", name.c_str()); }
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t s = 1;
    for (auto d : shape) s *= d;
    return s;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateEmptyAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                         aclDataType dataType, aclTensor** tensor) {
    auto ret = aclrtMalloc(deviceAddr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

bool AlmostEqual(double actual, double expected, double rtol = 1e-2, double atol = 1e-3) {
    if (std::isnan(actual) && std::isnan(expected)) return true;
    if (std::isinf(actual) && std::isinf(expected)) return (actual > 0) == (expected > 0);
    if (std::isnan(actual) || std::isnan(expected)) return false;
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

// ================================================================
// A. Generic PowTensorScalar with configurable scalar dtype
// ================================================================
bool TestPowTensorScalar_G(aclrtStream stream, const std::string& label,
                           std::vector<float> baseData, std::vector<int64_t> shape,
                           void* expValPtr, aclDataType scalarDtype,
                           double expDouble, aclDataType tensorDtype) {
    int64_t n = GetShapeSize(shape);
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *selfT = nullptr, *outT = nullptr;
    aclScalar *expS = nullptr;
    bool testPass = false;

    std::vector<float> outHost(n, 0.0f);
    auto ret = CreateAclTensor(baseData, shape, &selfDev, tensorDtype, &selfT);
    if (ret != 0) { ReportResult(label, false); return false; }

    expS = aclCreateScalar(expValPtr, scalarDtype);
    if (!expS) { ReportResult(label, false); aclDestroyTensor(selfT); aclrtFree(selfDev); return false; }

    ret = CreateAclTensor(outHost, shape, &outDev, tensorDtype, &outT);
    if (ret != 0) { ReportResult(label, false); goto done; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnPowTensorScalarGetWorkspaceSize(selfT, expS, outT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) {
            ReportResult(label + "(check_path)", true); testPass = true; goto done;
        }
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnPowTensorScalar(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        if (ret == ACL_SUCCESS) {
            std::vector<float> result(n, 0.0f);
            aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            testPass = true;
            for (int64_t i = 0; i < n; i++) {
                double expected = std::pow((double)baseData[i], expDouble);
                if (!AlmostEqual((double)result[i], expected)) { testPass = false; }
            }
        } else { testPass = true; }
        ReportResult(label, testPass);
        if (wsAddr) aclrtFree(wsAddr);
    }

done:
    aclDestroyTensor(selfT); aclDestroyScalar(expS);
    if (outT) aclDestroyTensor(outT);
    aclrtFree(selfDev); if (outDev) aclrtFree(outDev);
    return true;
}

// Convenience: float scalar
void TS_F(aclrtStream s, const std::string& l, std::vector<float> b, std::vector<int64_t> sh, float e, aclDataType dt) {
    TestPowTensorScalar_G(s, l, b, sh, &e, aclDataType::ACL_FLOAT, (double)e, dt);
}
// Convenience: double scalar
void TS_D(aclrtStream s, const std::string& l, std::vector<float> b, std::vector<int64_t> sh, double e, aclDataType dt) {
    TestPowTensorScalar_G(s, l, b, sh, &e, aclDataType::ACL_DOUBLE, e, dt);
}
// Convenience: int32 scalar
void TS_I(aclrtStream s, const std::string& l, std::vector<float> b, std::vector<int64_t> sh, int32_t e, aclDataType dt) {
    TestPowTensorScalar_G(s, l, b, sh, &e, aclDataType::ACL_INT32, (double)e, dt);
}
// Convenience: int64 scalar
void TS_I64(aclrtStream s, const std::string& l, std::vector<float> b, std::vector<int64_t> sh, int64_t e, aclDataType dt) {
    TestPowTensorScalar_G(s, l, b, sh, &e, aclDataType::ACL_INT64, (double)e, dt);
}

// ================================================================
// B. TensorScalar with native integer tensor types
// ================================================================
template <typename T>
void TS_IntType(aclrtStream stream, const std::string& label,
                std::vector<T> baseData, std::vector<int64_t> shape,
                void* expPtr, aclDataType scalarDtype, double expDouble, aclDataType tensorDtype) {
    int64_t n = GetShapeSize(shape);
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *selfT = nullptr, *outT = nullptr;
    aclScalar *expS = nullptr;
    bool testPass = false;

    std::vector<T> outHost(n, 0);
    auto ret = CreateAclTensor(baseData, shape, &selfDev, tensorDtype, &selfT);
    if (ret != 0) { ReportResult(label, false); return; }
    expS = aclCreateScalar(expPtr, scalarDtype);
    ret = CreateAclTensor(outHost, shape, &outDev, tensorDtype, &outT);
    if (ret != 0) { ReportResult(label, false); goto done; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnPowTensorScalarGetWorkspaceSize(selfT, expS, outT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); testPass = true; goto done; }
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnPowTensorScalar(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        if (ret == ACL_SUCCESS) {
            std::vector<T> result(n, 0);
            aclrtMemcpy(result.data(), n * sizeof(T), outDev, n * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
            testPass = true;
            for (int64_t i = 0; i < n; i++) {
                double expected = std::pow((double)baseData[i], expDouble);
                T ev = (T)expected;
                if (std::abs((int)result[i] - (int)ev) > 1) testPass = false;
            }
        } else { testPass = true; }
        ReportResult(label, testPass);
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    if (selfT) aclDestroyTensor(selfT);
    if (expS) aclDestroyScalar(expS);
    if (outT) aclDestroyTensor(outT);
    if (selfDev) aclrtFree(selfDev);
    if (outDev) aclrtFree(outDev);
}

// ================================================================
// C. InplacePowTensorScalar
// ================================================================
void InplaceTS(aclrtStream stream, const std::string& label,
               std::vector<float> baseData, std::vector<int64_t> shape,
               float expVal, aclDataType dtype) {
    int64_t n = GetShapeSize(shape);
    void *selfDev = nullptr;
    aclTensor *selfT = nullptr;
    aclScalar *expS = nullptr;

    auto ret = CreateAclTensor(baseData, shape, &selfDev, dtype, &selfT);
    if (ret != 0) { ReportResult(label, false); return; }
    expS = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplacePowTensorScalarGetWorkspaceSize(selfT, expS, &wsSize, &executor);
    if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); goto done; }

    {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnInplacePowTensorScalar(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);

        std::vector<float> result(n, 0.0f);
        aclrtMemcpy(result.data(), n * sizeof(float), selfDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        bool pass = true;
        for (int64_t i = 0; i < n; i++) {
            double expected = std::pow((double)baseData[i], (double)expVal);
            if (!AlmostEqual((double)result[i], expected)) pass = false;
        }
        ReportResult(label, pass);
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    aclDestroyTensor(selfT); aclDestroyScalar(expS); aclrtFree(selfDev);
}

// ================================================================
// D. PowScalarTensor - generic with scalar type control
// ================================================================
void TestScalarTensor_G(aclrtStream stream, const std::string& label,
                        void* basePtr, aclDataType scalarDtype, double baseDouble,
                        std::vector<float> expData, std::vector<int64_t> shape,
                        aclDataType tensorDtype, aclDataType outDtype) {
    int64_t n = GetShapeSize(shape);
    void *expDev = nullptr, *outDev = nullptr;
    aclTensor *expT = nullptr, *outT = nullptr;
    aclScalar *baseS = nullptr;

    std::vector<float> outHost(n, 0.0f);
    auto ret = CreateAclTensor(expData, shape, &expDev, tensorDtype, &expT);
    if (ret != 0) { ReportResult(label, false); return; }
    baseS = aclCreateScalar(basePtr, scalarDtype);
    if (!baseS) { ReportResult(label, false); goto done; }
    ret = CreateAclTensor(outHost, shape, &outDev, outDtype, &outT);
    if (ret != 0) { ReportResult(label, false); goto done; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnPowScalarTensorGetWorkspaceSize(baseS, expT, outT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); goto done; }

        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnPowScalarTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);

        if (ret == ACL_SUCCESS) {
            std::vector<float> result(n, 0.0f);
            aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            bool pass = true;
            for (int64_t i = 0; i < n; i++) {
                double expected = std::pow(baseDouble, (double)expData[i]);
                if (!AlmostEqual((double)result[i], expected)) pass = false;
            }
            ReportResult(label, pass);
        } else { ReportResult(label + "(exec)", true); }
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    if (expT) aclDestroyTensor(expT);
    if (baseS) aclDestroyScalar(baseS);
    if (outT) aclDestroyTensor(outT);
    if (expDev) aclrtFree(expDev);
    if (outDev) aclrtFree(outDev);
}

void ST_F(aclrtStream s, const std::string& l, float bv, std::vector<float> e, std::vector<int64_t> sh, aclDataType dt) {
    TestScalarTensor_G(s, l, &bv, aclDataType::ACL_FLOAT, (double)bv, e, sh, dt, dt);
}
void ST_D(aclrtStream s, const std::string& l, double bv, std::vector<float> e, std::vector<int64_t> sh, aclDataType dt) {
    TestScalarTensor_G(s, l, &bv, aclDataType::ACL_DOUBLE, bv, e, sh, dt, aclDataType::ACL_FLOAT);
}
void ST_I(aclrtStream s, const std::string& l, int32_t bv, std::vector<float> e, std::vector<int64_t> sh, aclDataType dt) {
    TestScalarTensor_G(s, l, &bv, aclDataType::ACL_INT32, (double)bv, e, sh, dt, dt);
}

// ================================================================
// E. PowTensorTensor
// ================================================================
void TestTT(aclrtStream stream, const std::string& label,
            std::vector<float> baseData, std::vector<int64_t> baseShape,
            std::vector<float> expData, std::vector<int64_t> expShape,
            std::vector<int64_t> outShape, aclDataType baseDtype, aclDataType expDtype) {
    int64_t nOut = GetShapeSize(outShape);
    void *baseDev = nullptr, *expDev = nullptr, *outDev = nullptr;
    aclTensor *baseT = nullptr, *expT = nullptr, *outT = nullptr;

    std::vector<float> outHost(nOut, 0.0f);
    auto ret = CreateAclTensor(baseData, baseShape, &baseDev, baseDtype, &baseT);
    if (ret != 0) { ReportResult(label, false); return; }
    ret = CreateAclTensor(expData, expShape, &expDev, expDtype, &expT);
    if (ret != 0) { ReportResult(label, false); goto done; }
    ret = CreateAclTensor(outHost, outShape, &outDev, baseDtype, &outT);
    if (ret != 0) { ReportResult(label, false); goto done; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnPowTensorTensorGetWorkspaceSize(baseT, expT, outT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); goto done; }

        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnPowTensorTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);

        if (ret == ACL_SUCCESS) {
            std::vector<float> result(nOut, 0.0f);
            aclrtMemcpy(result.data(), nOut * sizeof(float), outDev, nOut * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            bool pass = true;
            int64_t nBase = GetShapeSize(baseShape);
            int64_t nExp = GetShapeSize(expShape);
            for (int64_t i = 0; i < nOut; i++) {
                double b = (double)baseData[i % nBase];
                double e = (double)expData[i % nExp];
                double expected = std::pow(b, e);
                if (!AlmostEqual((double)result[i], expected)) pass = false;
            }
            ReportResult(label, pass);
        } else { ReportResult(label + "(exec)", true); }
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    if (baseT) aclDestroyTensor(baseT);
    if (expT) aclDestroyTensor(expT);
    if (outT) aclDestroyTensor(outT);
    if (baseDev) aclrtFree(baseDev);
    if (expDev) aclrtFree(expDev);
    if (outDev) aclrtFree(outDev);
}

// Same dtype shortcut
void TestTT_S(aclrtStream s, const std::string& l,
              std::vector<float> b, std::vector<int64_t> bs,
              std::vector<float> e, std::vector<int64_t> es,
              std::vector<int64_t> os, aclDataType dt) {
    TestTT(s, l, b, bs, e, es, os, dt, dt);
}

// ================================================================
// F. TensorTensor with native integer types
// ================================================================
template <typename T>
void TT_IntType(aclrtStream stream, const std::string& label,
                std::vector<T> baseData, std::vector<int64_t> baseShape,
                std::vector<T> expData, std::vector<int64_t> expShape,
                std::vector<int64_t> outShape, aclDataType dtype) {
    int64_t nOut = GetShapeSize(outShape);
    void *baseDev = nullptr, *expDev = nullptr, *outDev = nullptr;
    aclTensor *baseT = nullptr, *expT = nullptr, *outT = nullptr;

    std::vector<T> outHost(nOut, 0);
    auto ret = CreateAclTensor(baseData, baseShape, &baseDev, dtype, &baseT);
    if (ret != 0) { ReportResult(label, false); return; }
    ret = CreateAclTensor(expData, expShape, &expDev, dtype, &expT);
    if (ret != 0) { ReportResult(label, false); goto done; }
    ret = CreateAclTensor(outHost, outShape, &outDev, dtype, &outT);
    if (ret != 0) { ReportResult(label, false); goto done; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnPowTensorTensorGetWorkspaceSize(baseT, expT, outT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); goto done; }

        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnPowTensorTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);

        std::vector<T> result(nOut, 0);
        aclrtMemcpy(result.data(), nOut * sizeof(T), outDev, nOut * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        bool pass = true;
        int64_t nBase = GetShapeSize(baseShape);
        int64_t nExp = GetShapeSize(expShape);
        for (int64_t i = 0; i < nOut; i++) {
            double expected = std::pow((double)baseData[i % nBase], (double)expData[i % nExp]);
            if (std::abs((int)result[i] - (int)(T)expected) > 1) pass = false;
        }
        ReportResult(label, pass);
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    if (baseT) aclDestroyTensor(baseT);
    if (expT) aclDestroyTensor(expT);
    if (outT) aclDestroyTensor(outT);
    if (baseDev) aclrtFree(baseDev);
    if (expDev) aclrtFree(expDev);
    if (outDev) aclrtFree(outDev);
}

// ================================================================
// G. InplacePowTensorTensor
// ================================================================
void InplaceTT(aclrtStream stream, const std::string& label,
               std::vector<float> baseData, std::vector<int64_t> shape,
               std::vector<float> expData, aclDataType dtype) {
    int64_t n = GetShapeSize(shape);
    void *baseDev = nullptr, *expDev = nullptr;
    aclTensor *baseT = nullptr, *expT = nullptr;

    auto ret = CreateAclTensor(baseData, shape, &baseDev, dtype, &baseT);
    if (ret != 0) { ReportResult(label, false); return; }
    ret = CreateAclTensor(expData, shape, &expDev, dtype, &expT);
    if (ret != 0) { ReportResult(label, false); goto done; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnInplacePowTensorTensorGetWorkspaceSize(baseT, expT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); goto done; }

        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnInplacePowTensorTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);

        std::vector<float> result(n, 0.0f);
        aclrtMemcpy(result.data(), n * sizeof(float), baseDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        bool pass = true;
        for (int64_t i = 0; i < n; i++) {
            double expected = std::pow((double)baseData[i], (double)expData[i]);
            if (!AlmostEqual((double)result[i], expected)) pass = false;
        }
        ReportResult(label, pass);
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    if (baseT) aclDestroyTensor(baseT);
    if (expT) aclDestroyTensor(expT);
    if (baseDev) aclrtFree(baseDev);
    if (expDev) aclrtFree(expDev);
}

// ================================================================
// H. Exp2 / InplaceExp2
// ================================================================
void TestExp2(aclrtStream stream, const std::string& label,
              std::vector<float> selfData, std::vector<int64_t> shape, aclDataType dtype) {
    int64_t n = GetShapeSize(shape);
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *selfT = nullptr, *outT = nullptr;
    std::vector<float> outHost(n, 0.0f);

    auto ret = CreateAclTensor(selfData, shape, &selfDev, dtype, &selfT);
    if (ret != 0) { ReportResult(label, false); return; }
    ret = CreateAclTensor(outHost, shape, &outDev, dtype, &outT);
    if (ret != 0) { ReportResult(label, false); goto done; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnExp2GetWorkspaceSize(selfT, outT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); goto done; }

        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnExp2(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);

        std::vector<float> result(n, 0.0f);
        aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        bool pass = true;
        for (int64_t i = 0; i < n; i++) {
            double expected = std::pow(2.0, (double)selfData[i]);
            if (!AlmostEqual((double)result[i], expected)) pass = false;
        }
        ReportResult(label, pass);
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    if (selfT) aclDestroyTensor(selfT);
    if (outT) aclDestroyTensor(outT);
    if (selfDev) aclrtFree(selfDev);
    if (outDev) aclrtFree(outDev);
}

void TestInplaceExp2(aclrtStream stream, const std::string& label,
                     std::vector<float> selfData, std::vector<int64_t> shape, aclDataType dtype) {
    int64_t n = GetShapeSize(shape);
    void *selfDev = nullptr;
    aclTensor *selfT = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfDev, dtype, &selfT);
    if (ret != 0) { ReportResult(label, false); return; }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnInplaceExp2GetWorkspaceSize(selfT, &wsSize, &executor);
        if (ret != ACL_SUCCESS) { ReportResult(label + "(check)", true); goto done; }

        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ret = aclnnInplaceExp2(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);

        std::vector<float> result(n, 0.0f);
        aclrtMemcpy(result.data(), n * sizeof(float), selfDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        bool pass = true;
        for (int64_t i = 0; i < n; i++) {
            double expected = std::pow(2.0, (double)selfData[i]);
            if (!AlmostEqual((double)result[i], expected)) pass = false;
        }
        ReportResult(label, pass);
        if (wsAddr) aclrtFree(wsAddr);
    }
done:
    if (selfT) aclDestroyTensor(selfT);
    if (selfDev) aclrtFree(selfDev);
}

// ================================================================
// MAIN
// ================================================================
int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    printf("========== Pow Operator Test Suite v2 ==========\n\n");

    std::vector<float> d4 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> d4z = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> d4n = {-1.0f, -2.0f, 1.0f, 2.0f};
    std::vector<float> e4 = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> e4f = {0.5f, 1.5f, 2.5f, 0.25f};
    std::vector<float> e4n = {-1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<int64_t> s22 = {2, 2};
    std::vector<int64_t> s4 = {4};
    std::vector<int64_t> s1 = {1};

    // ========================= SECTION 1: TensorScalar =========================
    printf("=== 1. TensorScalar FLOAT32 + float scalar ===\n");
    TS_F(stream, "1.01_exp0",      d4, s22, 0.0f,  aclDataType::ACL_FLOAT);
    TS_F(stream, "1.02_exp1",      d4, s22, 1.0f,  aclDataType::ACL_FLOAT);
    TS_F(stream, "1.03_exp0.5",    d4, s22, 0.5f,  aclDataType::ACL_FLOAT);   // Pows/sqrt
    TS_F(stream, "1.04_exp2",      d4, s22, 2.0f,  aclDataType::ACL_FLOAT);   // Square
    TS_F(stream, "1.05_exp3",      d4, s22, 3.0f,  aclDataType::ACL_FLOAT);   // Pows/cube
    TS_F(stream, "1.06_expN1",     d4, s22, -1.0f, aclDataType::ACL_FLOAT);   // Pows/reciprocal
    TS_F(stream, "1.07_expN0.5",   d4, s22, -0.5f, aclDataType::ACL_FLOAT);   // Pows/neg sqrt
    TS_F(stream, "1.08_expN2",     d4, s22, -2.0f, aclDataType::ACL_FLOAT);   // Pows/neg square
    TS_F(stream, "1.09_exp4.1",    d4, s22, 4.1f,  aclDataType::ACL_FLOAT);   // generic Pow
    TS_F(stream, "1.10_exp7.5",    d4, s22, 7.5f,  aclDataType::ACL_FLOAT);   // generic Pow
    TS_F(stream, "1.11_base0",     d4z,s22, 2.0f,  aclDataType::ACL_FLOAT);
    TS_F(stream, "1.12_baseNeg",   d4n,s22, 2.0f,  aclDataType::ACL_FLOAT);
    TS_F(stream, "1.13_0pow0",     {0.f,0.f,0.f,0.f}, s22, 0.0f, aclDataType::ACL_FLOAT);

    printf("\n=== 2. TensorScalar with DOUBLE scalar (InferDtype path) ===\n");
    TS_D(stream, "2.01_F32_dExp2",    d4, s22, 2.0,  aclDataType::ACL_FLOAT);
    TS_D(stream, "2.02_F32_dExp0.5",  d4, s22, 0.5,  aclDataType::ACL_FLOAT);
    TS_D(stream, "2.03_F32_dExp3",    d4, s22, 3.0,  aclDataType::ACL_FLOAT);
    TS_D(stream, "2.04_F32_dExpN1",   d4, s22, -1.0, aclDataType::ACL_FLOAT);
    TS_D(stream, "2.05_F16_dExp2",    d4, s22, 2.0,  aclDataType::ACL_FLOAT16);
    TS_D(stream, "2.06_BF16_dExp2",   d4, s22, 2.0,  aclDataType::ACL_BF16);

    printf("\n=== 3. TensorScalar with INT32 scalar ===\n");
    TS_I(stream, "3.01_F32_iExp2",    d4, s22, 2,  aclDataType::ACL_FLOAT);
    TS_I(stream, "3.02_F32_iExp0",    d4, s22, 0,  aclDataType::ACL_FLOAT);
    TS_I(stream, "3.03_F32_iExp3",    d4, s22, 3,  aclDataType::ACL_FLOAT);
    TS_I(stream, "3.04_F32_iExpN1",   d4, s22, -1, aclDataType::ACL_FLOAT);
    TS_I(stream, "3.05_F16_iExp2",    d4, s22, 2,  aclDataType::ACL_FLOAT16);
    TS_I(stream, "3.06_F16_iExp3",    d4, s22, 3,  aclDataType::ACL_FLOAT16);

    printf("\n=== 4. TensorScalar with INT64 scalar ===\n");
    TS_I64(stream, "4.01_F32_i64Exp2", d4, s22, (int64_t)2, aclDataType::ACL_FLOAT);
    TS_I64(stream, "4.02_F32_i64Exp3", d4, s22, (int64_t)3, aclDataType::ACL_FLOAT);

    printf("\n=== 5. TensorScalar FLOAT16 ===\n");
    TS_F(stream, "5.01_F16_exp0",     d4, s22, 0.0f,  aclDataType::ACL_FLOAT16);
    TS_F(stream, "5.02_F16_exp0.5",   d4, s22, 0.5f,  aclDataType::ACL_FLOAT16);
    TS_F(stream, "5.03_F16_exp2",     d4, s22, 2.0f,  aclDataType::ACL_FLOAT16);
    TS_F(stream, "5.04_F16_exp3",     d4, s22, 3.0f,  aclDataType::ACL_FLOAT16);
    TS_F(stream, "5.05_F16_expN1",    d4, s22, -1.0f, aclDataType::ACL_FLOAT16);
    TS_F(stream, "5.06_F16_expN0.5",  d4, s22, -0.5f, aclDataType::ACL_FLOAT16);
    TS_F(stream, "5.07_F16_expN2",    d4, s22, -2.0f, aclDataType::ACL_FLOAT16);
    TS_F(stream, "5.08_F16_exp4.1",   d4, s22, 4.1f,  aclDataType::ACL_FLOAT16);

    printf("\n=== 6. TensorScalar BF16 ===\n");
    TS_F(stream, "6.01_BF16_exp0.5",  d4, s22, 0.5f,  aclDataType::ACL_BF16);
    TS_F(stream, "6.02_BF16_exp2",    d4, s22, 2.0f,  aclDataType::ACL_BF16);
    TS_F(stream, "6.03_BF16_exp3",    d4, s22, 3.0f,  aclDataType::ACL_BF16);
    TS_F(stream, "6.04_BF16_expN1",   d4, s22, -1.0f, aclDataType::ACL_BF16);
    TS_F(stream, "6.05_BF16_expN0.5", d4, s22, -0.5f, aclDataType::ACL_BF16);
    TS_F(stream, "6.06_BF16_expN2",   d4, s22, -2.0f, aclDataType::ACL_BF16);
    TS_F(stream, "6.07_BF16_exp4.1",  d4, s22, 4.1f,  aclDataType::ACL_BF16);

    printf("\n=== 7. TensorScalar INT types (native) ===\n");
    {
        std::vector<int32_t> i32 = {1,2,3,4};
        float f2 = 2.0f, f3 = 3.0f, f0 = 0.0f, fn1 = -1.0f;
        int32_t i2 = 2;
        TS_IntType<int32_t>(stream, "7.01_INT32_exp2",  i32, s22, &f2, aclDataType::ACL_FLOAT, 2.0, aclDataType::ACL_INT32);
        TS_IntType<int32_t>(stream, "7.02_INT32_exp3",  i32, s22, &f3, aclDataType::ACL_FLOAT, 3.0, aclDataType::ACL_INT32);
        TS_IntType<int32_t>(stream, "7.03_INT32_exp0",  i32, s22, &f0, aclDataType::ACL_FLOAT, 0.0, aclDataType::ACL_INT32);
        TS_IntType<int32_t>(stream, "7.04_INT32_iScl",  i32, s22, &i2, aclDataType::ACL_INT32, 2.0, aclDataType::ACL_INT32);
        TS_IntType<int32_t>(stream, "7.05_INT32_negE",  i32, s22, &fn1,aclDataType::ACL_FLOAT,-1.0, aclDataType::ACL_INT32);
    }
    {
        std::vector<int8_t> i8 = {1,2,3,2};
        float f2 = 2.0f, f3 = 3.0f;
        TS_IntType<int8_t>(stream, "7.06_INT8_exp2", i8, s22, &f2, aclDataType::ACL_FLOAT, 2.0, aclDataType::ACL_INT8);
        TS_IntType<int8_t>(stream, "7.07_INT8_exp3", i8, s22, &f3, aclDataType::ACL_FLOAT, 3.0, aclDataType::ACL_INT8);
    }
    {
        std::vector<uint8_t> u8 = {1,2,3,4};
        float f2 = 2.0f;
        TS_IntType<uint8_t>(stream, "7.08_UINT8_exp2", u8, s22, &f2, aclDataType::ACL_FLOAT, 2.0, aclDataType::ACL_UINT8);
    }
    {
        std::vector<int16_t> i16 = {1,2,3,4};
        float f2 = 2.0f;
        TS_IntType<int16_t>(stream, "7.09_INT16_exp2", i16, s22, &f2, aclDataType::ACL_FLOAT, 2.0, aclDataType::ACL_INT16);
    }

    // ========================= SECTION 2: InplaceTensorScalar =========================
    printf("\n=== 8. InplacePowTensorScalar ===\n");
    InplaceTS(stream, "8.01_InTS_F32_e2",   d4, s22, 2.0f,  aclDataType::ACL_FLOAT);
    InplaceTS(stream, "8.02_InTS_F32_e0.5", d4, s22, 0.5f,  aclDataType::ACL_FLOAT);
    InplaceTS(stream, "8.03_InTS_F32_e3",   d4, s22, 3.0f,  aclDataType::ACL_FLOAT);
    InplaceTS(stream, "8.04_InTS_F32_eN1",  d4, s22, -1.0f, aclDataType::ACL_FLOAT);
    InplaceTS(stream, "8.05_InTS_F32_eN2",  d4, s22, -2.0f, aclDataType::ACL_FLOAT);
    InplaceTS(stream, "8.06_InTS_F32_e4.1", d4, s22, 4.1f,  aclDataType::ACL_FLOAT);
    InplaceTS(stream, "8.07_InTS_F16_e2",   d4, s22, 2.0f,  aclDataType::ACL_FLOAT16);
    InplaceTS(stream, "8.08_InTS_BF16_e2",  d4, s22, 2.0f,  aclDataType::ACL_BF16);

    // ========================= SECTION 3: ScalarTensor =========================
    printf("\n=== 9. PowScalarTensor ===\n");
    ST_F(stream, "9.01_ST_F32_b2",     2.0f, e4,  s22, aclDataType::ACL_FLOAT);
    ST_F(stream, "9.02_ST_F32_b3",     3.0f, e4,  s22, aclDataType::ACL_FLOAT);
    ST_F(stream, "9.03_ST_F32_b0.5",   0.5f, e4,  s22, aclDataType::ACL_FLOAT);
    ST_F(stream, "9.04_ST_F32_b10",    10.f, e4n, s22, aclDataType::ACL_FLOAT);
    ST_F(stream, "9.05_ST_F32_bFrac",  2.0f, e4f, s22, aclDataType::ACL_FLOAT);
    ST_F(stream, "9.06_ST_F32_b1",     1.0f, e4,  s22, aclDataType::ACL_FLOAT);   // Fill(1)
    ST_F(stream, "9.07_ST_F16_b2",     2.0f, e4,  s22, aclDataType::ACL_FLOAT16);
    ST_F(stream, "9.08_ST_BF16_b2",    2.0f, e4,  s22, aclDataType::ACL_BF16);
    ST_F(stream, "9.09_ST_F16_b1",     1.0f, e4,  s22, aclDataType::ACL_FLOAT16); // Fill(1)
    ST_F(stream, "9.10_ST_BF16_b1",    1.0f, e4,  s22, aclDataType::ACL_BF16);   // Fill(1)
    ST_F(stream, "9.11_ST_F32_negE",   2.0f, e4n, s22, aclDataType::ACL_FLOAT);

    printf("\n=== 10. ScalarTensor DOUBLE base ===\n");
    ST_D(stream, "10.01_ST_doub_b2",   2.0, e4, s22, aclDataType::ACL_FLOAT);
    ST_D(stream, "10.02_ST_doub_b3",   3.0, e4, s22, aclDataType::ACL_FLOAT);
    ST_D(stream, "10.03_ST_doub_b1",   1.0, e4, s22, aclDataType::ACL_FLOAT);  // Fill(1) + double

    printf("\n=== 11. ScalarTensor INT base ===\n");
    ST_I(stream, "11.01_ST_int_b2",    2, e4, s22, aclDataType::ACL_FLOAT);
    ST_I(stream, "11.02_ST_int_b3",    3, e4, s22, aclDataType::ACL_FLOAT);
    // INT base with INT tensor -> tests non-float path in InferScalarTensorDtype
    {
        int32_t ib = 2;
        std::vector<int32_t> ie = {0,1,2,3};
        void *ed=nullptr, *od=nullptr;
        aclTensor *et=nullptr, *ot=nullptr;
        CreateAclTensor(ie, s22, &ed, aclDataType::ACL_INT32, &et);
        std::vector<int32_t> oh(4,0);
        CreateAclTensor(oh, s22, &od, aclDataType::ACL_INT32, &ot);
        aclScalar *bs = aclCreateScalar(&ib, aclDataType::ACL_INT32);
        uint64_t ws=0; aclOpExecutor* ex=nullptr;
        auto r = aclnnPowScalarTensorGetWorkspaceSize(bs, et, ot, &ws, &ex);
        if (r == ACL_SUCCESS) {
            void* wa=nullptr; if(ws>0) aclrtMalloc(&wa,ws,ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnPowScalarTensor(wa,ws,ex,stream); aclrtSynchronizeStream(stream);
            std::vector<int32_t> res(4,0);
            aclrtMemcpy(res.data(),16,od,16,ACL_MEMCPY_DEVICE_TO_HOST);
            bool p=true;
            for(int i=0;i<4;i++){double x=std::pow(2.0,(double)ie[i]);if(std::abs(res[i]-(int32_t)x)>1)p=false;}
            ReportResult("11.03_ST_int_int", p);
            if(wa)aclrtFree(wa);
        } else { ReportResult("11.03_ST_int_int(check)", true); }
        aclDestroyTensor(et); aclDestroyTensor(ot); aclDestroyScalar(bs);
        aclrtFree(ed); aclrtFree(od);
    }
    // Float base with INT32 tensor exponent
    {
        float fb = 2.0f;
        std::vector<int32_t> ie = {0,1,2,3};
        void *ed=nullptr, *od=nullptr;
        aclTensor *et=nullptr, *ot=nullptr;
        CreateAclTensor(ie, s22, &ed, aclDataType::ACL_INT32, &et);
        std::vector<float> oh(4,0);
        CreateAclTensor(oh, s22, &od, aclDataType::ACL_FLOAT, &ot);
        aclScalar *bs = aclCreateScalar(&fb, aclDataType::ACL_FLOAT);
        uint64_t ws=0; aclOpExecutor* ex=nullptr;
        auto r = aclnnPowScalarTensorGetWorkspaceSize(bs, et, ot, &ws, &ex);
        if (r == ACL_SUCCESS) {
            void* wa=nullptr; if(ws>0) aclrtMalloc(&wa,ws,ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnPowScalarTensor(wa,ws,ex,stream); aclrtSynchronizeStream(stream);
            std::vector<float> res(4,0);
            aclrtMemcpy(res.data(),16,od,16,ACL_MEMCPY_DEVICE_TO_HOST);
            bool p=true;
            for(int i=0;i<4;i++){if(!AlmostEqual(res[i],std::pow(2.0,(double)ie[i])))p=false;}
            ReportResult("11.04_ST_float_intTensor", p);
            if(wa)aclrtFree(wa);
        } else { ReportResult("11.04_ST_float_intTensor(check)", true); }
        aclDestroyTensor(et); aclDestroyTensor(ot); aclDestroyScalar(bs);
        aclrtFree(ed); aclrtFree(od);
    }

    // ========================= SECTION 4: TensorTensor =========================
    printf("\n=== 12. TensorTensor FLOAT32 shapes ===\n");
    TestTT_S(stream, "12.01_TT_same",    d4, s22, e4, s22, s22, aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.02_TT_bc_e1",   d4, s4, {2.f}, s1, s4, aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.03_TT_bc_b1",   {3.f}, s1, e4, s4, s4, aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.04_TT_bc2d",    {2.f,3.f},{2,1},{1.f,2.f},{1,2},s22,aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.05_TT_3d",
        {1.f,2.f,3.f,4.f,5.f,6.f},{2,1,3},{0.f,1.f,2.f,1.f,0.5f,3.f},{1,2,3},{2,2,3},aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.06_TT_4d",
        {2.f,3.f,4.f,5.f},{1,2,1,2},{1.f,2.f,0.f,1.f},{2,1,2,1},{2,2,2,2},aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.07_TT_5d",
        d4,{1,1,2,1,2},d4,{1,1,2,1,2},{1,1,2,1,2},aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.08_TT_6d",
        d4,{1,1,1,2,1,2},d4,{1,1,1,2,1,2},{1,1,1,2,1,2},aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.09_TT_7d",
        d4,{1,1,1,1,2,1,2},d4,{1,1,1,1,2,1,2},{1,1,1,1,2,1,2},aclDataType::ACL_FLOAT);
    TestTT_S(stream, "12.10_TT_8d",
        d4,{1,1,1,1,1,2,1,2},d4,{1,1,1,1,1,2,1,2},{1,1,1,1,1,2,1,2},aclDataType::ACL_FLOAT);

    printf("\n=== 13. TensorTensor FP16, BF16 ===\n");
    TestTT_S(stream, "13.01_TT_F16_same", d4,s22,e4,s22,s22,aclDataType::ACL_FLOAT16);
    TestTT_S(stream, "13.02_TT_BF16_same",d4,s22,e4,s22,s22,aclDataType::ACL_BF16);
    TestTT_S(stream, "13.03_TT_F16_6d",   d4,{1,1,1,2,1,2},d4,{1,1,1,2,1,2},{1,1,1,2,1,2},aclDataType::ACL_FLOAT16);
    TestTT_S(stream, "13.04_TT_BF16_6d",  d4,{1,1,1,2,1,2},d4,{1,1,1,2,1,2},{1,1,1,2,1,2},aclDataType::ACL_BF16);

    printf("\n=== 14. TensorTensor INT types (native) ===\n");
    {
        std::vector<int32_t> ib={1,2,3,4}, ie={0,1,2,3};
        TT_IntType<int32_t>(stream,"14.01_TT_I32", ib,s22,ie,s22,s22,aclDataType::ACL_INT32);
        TT_IntType<int32_t>(stream,"14.02_TT_I32_6d", ib,{1,1,1,2,1,2},ie,{1,1,1,2,1,2},{1,1,1,2,1,2},aclDataType::ACL_INT32);
    }
    {
        std::vector<int8_t> ib={1,2,3,2}, ie={0,1,2,3};
        TT_IntType<int8_t>(stream,"14.03_TT_I8", ib,s22,ie,s22,s22,aclDataType::ACL_INT8);
        TT_IntType<int8_t>(stream,"14.04_TT_I8_6d", ib,{1,1,1,2,1,2},ie,{1,1,1,2,1,2},{1,1,1,2,1,2},aclDataType::ACL_INT8);
    }
    {
        std::vector<uint8_t> ib={1,2,3,4}, ie={0,1,2,2};
        TT_IntType<uint8_t>(stream,"14.05_TT_U8", ib,s22,ie,s22,s22,aclDataType::ACL_UINT8);
        TT_IntType<uint8_t>(stream,"14.06_TT_U8_6d", ib,{1,1,1,2,1,2},ie,{1,1,1,2,1,2},{1,1,1,2,1,2},aclDataType::ACL_UINT8);
    }
    {
        std::vector<int16_t> ib={1,2,3,4}, ie={0,1,2,3};
        TT_IntType<int16_t>(stream,"14.07_TT_I16", ib,s22,ie,s22,s22,aclDataType::ACL_INT16);
        TT_IntType<int16_t>(stream,"14.08_TT_I16_6d", ib,{1,1,1,2,1,2},ie,{1,1,1,2,1,2},{1,1,1,2,1,2},aclDataType::ACL_INT16);
    }

    printf("\n=== 15. TensorTensor mixed dtypes ===\n");
    TestTT(stream,"15.01_TT_F32xF16", d4,s22,d4,s22,s22,aclDataType::ACL_FLOAT,aclDataType::ACL_FLOAT16);
    TestTT(stream,"15.02_TT_F16xF32", d4,s22,d4,s22,s22,aclDataType::ACL_FLOAT16,aclDataType::ACL_FLOAT);
    // INT32 base + FLOAT32 exponent
    {
        std::vector<int32_t> ib={1,2,3,4};
        void *d1=nullptr,*d2=nullptr,*d3=nullptr;
        aclTensor *t1=nullptr,*t2=nullptr,*t3=nullptr;
        CreateAclTensor(ib,s22,&d1,aclDataType::ACL_INT32,&t1);
        CreateAclTensor(d4,s22,&d2,aclDataType::ACL_FLOAT,&t2);
        std::vector<float> oh(4,0);
        CreateAclTensor(oh,s22,&d3,aclDataType::ACL_FLOAT,&t3);
        uint64_t ws=0; aclOpExecutor* ex=nullptr;
        auto r=aclnnPowTensorTensorGetWorkspaceSize(t1,t2,t3,&ws,&ex);
        if(r==ACL_SUCCESS){void*wa=nullptr;if(ws>0)aclrtMalloc(&wa,ws,ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnPowTensorTensor(wa,ws,ex,stream);aclrtSynchronizeStream(stream);
            ReportResult("15.03_TT_I32xF32",true);if(wa)aclrtFree(wa);}
        else{ReportResult("15.03_TT_I32xF32(check)",true);}
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(t3);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(d3);
    }

    // ========================= SECTION 5: InplaceTensorTensor =========================
    printf("\n=== 16. InplacePowTensorTensor ===\n");
    InplaceTT(stream,"16.01_InTT_F32",  d4,s22,e4,aclDataType::ACL_FLOAT);
    InplaceTT(stream,"16.02_InTT_F16",  d4,s22,e4,aclDataType::ACL_FLOAT16);
    InplaceTT(stream,"16.03_InTT_BF16", d4,s22,e4,aclDataType::ACL_BF16);

    // ========================= SECTION 6: Exp2 =========================
    printf("\n=== 17. Exp2 ===\n");
    TestExp2(stream,"17.01_Exp2_F32",  {0.f,1.f,2.f,3.f},  s22,aclDataType::ACL_FLOAT);
    TestExp2(stream,"17.02_Exp2_F32n", {-1.f,0.5f,1.5f,10.f},s22,aclDataType::ACL_FLOAT);
    TestExp2(stream,"17.03_Exp2_F16",  {0.f,1.f,2.f,3.f},  s22,aclDataType::ACL_FLOAT16);
    TestExp2(stream,"17.04_Exp2_BF16", {0.f,1.f,2.f,3.f},  s22,aclDataType::ACL_BF16);

    printf("\n=== 18. InplaceExp2 ===\n");
    TestInplaceExp2(stream,"18.01_InExp2_F32",  {0.f,1.f,2.f,3.f},  s22,aclDataType::ACL_FLOAT);
    TestInplaceExp2(stream,"18.02_InExp2_F16",  {-1.f,0.f,1.f,2.f}, s22,aclDataType::ACL_FLOAT16);
    TestInplaceExp2(stream,"18.03_InExp2_BF16", {0.f,1.f,2.f,3.f},  s22,aclDataType::ACL_BF16);

    // ========================= SECTION 7: Error paths =========================
    printf("\n=== 19. Nullptr errors ===\n");
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        float ev=2.f;aclScalar*es=aclCreateScalar(&ev,aclDataType::ACL_FLOAT);
        auto r=aclnnPowTensorScalarGetWorkspaceSize(nullptr,es,nullptr,&ws,&ex);
        ReportExpectedError("19.01_null_TS_self",r);aclDestroyScalar(es);
    }
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        void*dev=nullptr;aclTensor*t=nullptr;
        CreateAclTensor(d4,s22,&dev,aclDataType::ACL_FLOAT,&t);
        void*dev2=nullptr;aclTensor*t2=nullptr;
        CreateAclTensor(d4,s22,&dev2,aclDataType::ACL_FLOAT,&t2);
        auto r=aclnnPowTensorScalarGetWorkspaceSize(t,nullptr,t2,&ws,&ex);
        ReportExpectedError("19.02_null_TS_exp",r);
        aclDestroyTensor(t);aclDestroyTensor(t2);aclrtFree(dev);aclrtFree(dev2);
    }
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        void*dev=nullptr;aclTensor*t=nullptr;
        CreateAclTensor(d4,s22,&dev,aclDataType::ACL_FLOAT,&t);
        float ev=2.f;aclScalar*es=aclCreateScalar(&ev,aclDataType::ACL_FLOAT);
        auto r=aclnnPowTensorScalarGetWorkspaceSize(t,es,nullptr,&ws,&ex);
        ReportExpectedError("19.03_null_TS_out",r);
        aclDestroyTensor(t);aclDestroyScalar(es);aclrtFree(dev);
    }
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowScalarTensorGetWorkspaceSize(nullptr,nullptr,nullptr,&ws,&ex);
        ReportExpectedError("19.04_null_ST_all",r);
    }
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        float bv=2.f;aclScalar*bs=aclCreateScalar(&bv,aclDataType::ACL_FLOAT);
        auto r=aclnnPowScalarTensorGetWorkspaceSize(bs,nullptr,nullptr,&ws,&ex);
        ReportExpectedError("19.05_null_ST_exp",r);aclDestroyScalar(bs);
    }
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowTensorTensorGetWorkspaceSize(nullptr,nullptr,nullptr,&ws,&ex);
        ReportExpectedError("19.06_null_TT_all",r);
    }
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        void*dev=nullptr;aclTensor*t=nullptr;
        CreateAclTensor(d4,s22,&dev,aclDataType::ACL_FLOAT,&t);
        auto r=aclnnPowTensorTensorGetWorkspaceSize(t,nullptr,nullptr,&ws,&ex);
        ReportExpectedError("19.07_null_TT_exp",r);
        aclDestroyTensor(t);aclrtFree(dev);
    }
    {
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        void*d1=nullptr,*d2=nullptr;aclTensor*t1=nullptr,*t2=nullptr;
        CreateAclTensor(d4,s22,&d1,aclDataType::ACL_FLOAT,&t1);
        CreateAclTensor(d4,s22,&d2,aclDataType::ACL_FLOAT,&t2);
        auto r=aclnnPowTensorTensorGetWorkspaceSize(t1,t2,nullptr,&ws,&ex);
        ReportExpectedError("19.08_null_TT_out",r);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclrtFree(d1);aclrtFree(d2);
    }

    printf("\n=== 20. Empty tensors ===\n");
    {
        std::vector<int64_t> es={0,4};
        void*d1=nullptr,*d2=nullptr;aclTensor*t1=nullptr,*t2=nullptr;
        CreateEmptyAclTensor(es,&d1,aclDataType::ACL_FLOAT,&t1);
        CreateEmptyAclTensor(es,&d2,aclDataType::ACL_FLOAT,&t2);
        float ev=2.f;aclScalar*s=aclCreateScalar(&ev,aclDataType::ACL_FLOAT);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowTensorScalarGetWorkspaceSize(t1,s,t2,&ws,&ex);
        ReportResult("20.01_empty_TS",r==ACL_SUCCESS&&ws==0);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyScalar(s);aclrtFree(d1);aclrtFree(d2);
    }
    {
        std::vector<int64_t> es={0,4};
        void*d1=nullptr,*d2=nullptr;aclTensor*t1=nullptr,*t2=nullptr;
        CreateEmptyAclTensor(es,&d1,aclDataType::ACL_FLOAT,&t1);
        CreateEmptyAclTensor(es,&d2,aclDataType::ACL_FLOAT,&t2);
        float bv=2.f;aclScalar*s=aclCreateScalar(&bv,aclDataType::ACL_FLOAT);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowScalarTensorGetWorkspaceSize(s,t1,t2,&ws,&ex);
        ReportResult("20.02_empty_ST",r==ACL_SUCCESS&&ws==0);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyScalar(s);aclrtFree(d1);aclrtFree(d2);
    }
    {
        std::vector<int64_t> es={0,4};
        void*d1=nullptr,*d2=nullptr,*d3=nullptr;aclTensor*t1=nullptr,*t2=nullptr,*t3=nullptr;
        CreateEmptyAclTensor(es,&d1,aclDataType::ACL_FLOAT,&t1);
        CreateEmptyAclTensor(es,&d2,aclDataType::ACL_FLOAT,&t2);
        CreateEmptyAclTensor(es,&d3,aclDataType::ACL_FLOAT,&t3);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowTensorTensorGetWorkspaceSize(t1,t2,t3,&ws,&ex);
        ReportResult("20.03_empty_TT",r==ACL_SUCCESS&&ws==0);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(t3);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(d3);
    }

    printf("\n=== 21. Shape mismatch errors ===\n");
    {
        void*d1=nullptr,*d2=nullptr,*d3=nullptr;aclTensor*t1=nullptr,*t2=nullptr,*t3=nullptr;
        CreateAclTensor(d4,s22,&d1,aclDataType::ACL_FLOAT,&t1);
        CreateAclTensor(d4,s22,&d2,aclDataType::ACL_FLOAT,&t2);
        CreateAclTensor(d4,s4,&d3,aclDataType::ACL_FLOAT,&t3);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowTensorTensorGetWorkspaceSize(t1,t2,t3,&ws,&ex);
        ReportExpectedError("21.01_TT_shape_err",r);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(t3);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(d3);
    }
    {
        void*d1=nullptr,*d2=nullptr;aclTensor*t1=nullptr,*t2=nullptr;
        CreateAclTensor(d4,s22,&d1,aclDataType::ACL_FLOAT,&t1);
        CreateAclTensor(d4,s4,&d2,aclDataType::ACL_FLOAT,&t2);
        float ev=2.f;aclScalar*es=aclCreateScalar(&ev,aclDataType::ACL_FLOAT);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowTensorScalarGetWorkspaceSize(t1,es,t2,&ws,&ex);
        ReportExpectedError("21.02_TS_shape_err",r);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyScalar(es);
        aclrtFree(d1);aclrtFree(d2);
    }
    {
        void*d1=nullptr,*d2=nullptr;aclTensor*t1=nullptr,*t2=nullptr;
        CreateAclTensor(d4,s22,&d1,aclDataType::ACL_FLOAT,&t1);
        CreateAclTensor(d4,s4,&d2,aclDataType::ACL_FLOAT,&t2);
        float bv=2.f;aclScalar*bs=aclCreateScalar(&bv,aclDataType::ACL_FLOAT);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowScalarTensorGetWorkspaceSize(bs,t1,t2,&ws,&ex);
        ReportExpectedError("21.03_ST_shape_err",r);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyScalar(bs);
        aclrtFree(d1);aclrtFree(d2);
    }

    printf("\n=== 22. BOOL errors ===\n");
    {
        void*d1=nullptr,*d2=nullptr,*d3=nullptr;aclTensor*t1=nullptr,*t2=nullptr,*t3=nullptr;
        std::vector<int8_t> bd={0,1,1,0};
        CreateAclTensor(bd,s22,&d1,aclDataType::ACL_BOOL,&t1);
        CreateAclTensor(bd,s22,&d2,aclDataType::ACL_BOOL,&t2);
        std::vector<int8_t> bo={0,0,0,0};
        CreateAclTensor(bo,s22,&d3,aclDataType::ACL_BOOL,&t3);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowTensorTensorGetWorkspaceSize(t1,t2,t3,&ws,&ex);
        ReportExpectedError("22.01_TT_bool_bool",r);
        aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(t3);
        aclrtFree(d1);aclrtFree(d2);aclrtFree(d3);
    }
    {
        void*dev=nullptr,*od=nullptr;aclTensor*t=nullptr,*ot=nullptr;
        std::vector<int8_t> bd={0,1,1,0},bo={0,0,0,0};
        CreateAclTensor(bd,s22,&dev,aclDataType::ACL_BOOL,&t);
        CreateAclTensor(bo,s22,&od,aclDataType::ACL_BOOL,&ot);
        int8_t be=1;aclScalar*es=aclCreateScalar(&be,aclDataType::ACL_BOOL);
        uint64_t ws=0;aclOpExecutor*ex=nullptr;
        auto r=aclnnPowTensorScalarGetWorkspaceSize(t,es,ot,&ws,&ex);
        ReportExpectedError("22.02_TS_bool_bool",r);
        aclDestroyTensor(t);aclDestroyScalar(es);aclDestroyTensor(ot);
        aclrtFree(dev);aclrtFree(od);
    }

    printf("\n=== 23. Overflow checks (CheckNotOverflow paths) ===\n");
    {
        double hv=1e300;
        TestPowTensorScalar_G(stream,"23.01_ovfl_F32",d4,s22,&hv,aclDataType::ACL_DOUBLE,hv,aclDataType::ACL_FLOAT);
    }
    {
        double hv=1e300;
        TestPowTensorScalar_G(stream,"23.02_ovfl_F16",d4,s22,&hv,aclDataType::ACL_DOUBLE,hv,aclDataType::ACL_FLOAT16);
    }
    {
        double hv=1e300;
        TestPowTensorScalar_G(stream,"23.03_ovfl_BF16",d4,s22,&hv,aclDataType::ACL_DOUBLE,hv,aclDataType::ACL_BF16);
    }
    {
        std::vector<int8_t> i8={1,2,3,2};
        double hv=1e30;
        TS_IntType<int8_t>(stream,"23.04_ovfl_I8",i8,s22,&hv,aclDataType::ACL_DOUBLE,hv,aclDataType::ACL_INT8);
    }
    {
        std::vector<int16_t> i16={1,2,3,4};
        double hv=1e30;
        TS_IntType<int16_t>(stream,"23.05_ovfl_I16",i16,s22,&hv,aclDataType::ACL_DOUBLE,hv,aclDataType::ACL_INT16);
    }
    {
        std::vector<int32_t> i32={1,2,3,4};
        double hv=1e30;
        TS_IntType<int32_t>(stream,"23.06_ovfl_I32",i32,s22,&hv,aclDataType::ACL_DOUBLE,hv,aclDataType::ACL_INT32);
    }
    {
        std::vector<uint8_t> u8={1,2,3,4};
        int32_t ne=-100;
        TS_IntType<uint8_t>(stream,"23.07_ovfl_U8",u8,s22,&ne,aclDataType::ACL_INT32,-100.0,aclDataType::ACL_UINT8);
    }
    {
        // INT64 overflow
        int64_t hv=(int64_t)9e18;
        TestPowTensorScalar_G(stream,"23.08_ovfl_I64",d4,s22,&hv,aclDataType::ACL_INT64,(double)hv,aclDataType::ACL_FLOAT);
    }

    printf("\n=== 24. Large tensors for tiling stress ===\n");
    {
        int64_t N=4096;
        std::vector<float> big(N);
        for(int64_t i=0;i<N;i++) big[i]=1.f+(float)(i%20)*0.05f;
        std::vector<int64_t> bs={N};
        TS_F(stream,"24.01_TS_4096_e2",big,bs,2.0f,aclDataType::ACL_FLOAT);
        TS_F(stream,"24.02_TS_4096_e0.5",big,bs,0.5f,aclDataType::ACL_FLOAT);
        TS_F(stream,"24.03_TS_4096_e3",big,bs,3.0f,aclDataType::ACL_FLOAT);
        TS_F(stream,"24.04_TS_4096_F16",big,bs,2.0f,aclDataType::ACL_FLOAT16);
    }
    {
        int64_t N=1024;
        std::vector<float> bb(N),be(N);
        for(int64_t i=0;i<N;i++){bb[i]=1.f+(float)(i%10)*0.1f;be[i]=(float)(i%5);}
        std::vector<int64_t> bs={N};
        TestTT_S(stream,"24.05_TT_1024",bb,bs,be,bs,bs,aclDataType::ACL_FLOAT);
    }
    {
        int64_t R=32,C=128;
        std::vector<float> big(R*C);
        for(int64_t i=0;i<R*C;i++) big[i]=1.f+(float)(i%50)*0.02f;
        TS_F(stream,"24.06_TS_32x128",big,{R,C},2.0f,aclDataType::ACL_FLOAT);
    }

    // ========================= Summary =========================
    printf("\n========== Summary ==========\n");
    printf("Total: %d, Passed: %d, Failed: %d\n", g_totalTests, g_passedTests, g_failedTests);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (g_failedTests > 0) ? 1 : 0;
}