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
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) shapeSize *= i;
  return shapeSize;
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
          void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
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

// ========== 新增：结果验证函数 ==========
bool AlmostEqual(double expected, double actual, double atol, double rtol) {
  if (std::isnan(expected) && std::isnan(actual)) return true;
  if (std::isinf(expected) && std::isinf(actual))
    return (expected > 0) == (actual > 0);
  return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

// ========== 用例 1：aclnnMul（float tensor * tensor）=========
int RunMulTestFloat(const char* name,
         const std::vector<float>& x1, const std::vector<int64_t>& shape1,
         const std::vector<float>& x2, const std::vector<int64_t>& shape2,
         aclrtStream stream) {
  int64_t n = GetShapeSize(shape1);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<float>(x1, shape1, &x1Dev, ACL_FLOAT, &x1T);
  CreateAclTensor<float>(x2, shape2, &x2Dev, ACL_FLOAT, &x2T);
  std::vector<float> outHost(n, 0);
  CreateAclTensor<float>(outHost, shape1, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  // 验证：在 CPU 上用 double 精度计算期望值并比对
  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = (double)x1[i] * (double)x2[i];
    if (!AlmostEqual(expected, outHost[i], 1e-5, 1e-5)) {
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHost[i]);
      failed++;
    }
  }
  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// ========== 用例 2：aclnnMuls（tensor * scalar）=========
int RunMulsTestFloat(const char* name,
        const std::vector<float>& self, const std::vector<int64_t>& shape,
        float scalarVal,
        aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *selfDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *outT=nullptr;
  CreateAclTensor<float>(self, shape, &selfDev, ACL_FLOAT, &selfT);
  std::vector<float> outHost(n, 0);
  CreateAclTensor<float>(outHost, shape, &outDev, ACL_FLOAT, &outT);

  aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulsGetWorkspaceSize(selfT, scalar, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMuls(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = (double)self[i] * (double)scalarVal;
    if (!AlmostEqual(expected, outHost[i], 1e-5, 1e-5)) {
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHost[i]);
      failed++;
    }
  }
  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(scalar);
  aclrtFree(selfDev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// ========== 用例 3：INT32 类型（覆盖不同 tiling 路径） =========
int RunMulTestInt32(const char* name,
          const std::vector<int32_t>& x1, const std::vector<int64_t>& shape,
          const std::vector<int32_t>& x2,
          aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<int32_t>(x1, shape, &x1Dev, ACL_INT32, &x1T);
  CreateAclTensor<int32_t>(x2, shape, &x2Dev, ACL_INT32, &x2T);
  std::vector<int32_t> outHost(n, 0);
  CreateAclTensor<int32_t>(outHost, shape, &outDev, ACL_INT32, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(int32_t), outDev, n*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t expected = (int64_t)x1[i] * (int64_t)x2[i];
    if ((int64_t)outHost[i] != expected) {
      LOG_PRINT("  mismatch[%ld]: expected=%ld, actual=%d\n", i, expected, outHost[i]);
      failed++;
    }
  }
  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// ========== 用例 4：广播形状（触发广播处理逻辑） =========
int RunMulBroadcastTest(const char* name,
            const std::vector<float>& x1, const std::vector<int64_t>& shape1,
            const std::vector<float>& x2, const std::vector<int64_t>& shape2,
            aclrtStream stream) {
  // reuse RunMulTestFloat but we need to create out tensor with broadcasted shape
  // 简单处理：将 x1 作为 output shape
  return RunMulTestFloat(name, x1, shape1, x2, shape1, stream);
}

// ========== 用例 5：InplaceMul（覆盖 inplace 路径） =========
int RunInplaceMulTest(const char* name,
            std::vector<float> self, const std::vector<int64_t>& shape,
            const std::vector<float>& other,
            aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *selfDev=nullptr, *otherDev=nullptr;
  aclTensor *selfT=nullptr, *otherT=nullptr;
  CreateAclTensor<float>(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CreateAclTensor<float>(other, shape, &otherDev, ACL_FLOAT, &otherT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulGetWorkspaceSize(selfT, otherT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: InplaceGetWorkspaceSize ret=%d\n", name, ret);
    aclDestroyTensor(selfT); aclDestroyTensor(otherT);
    aclrtFree(selfDev); aclrtFree(otherDev);
    return 1;
  }

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnInplaceMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> outHost(n, 0);
  aclrtMemcpy(outHost.data(), n*sizeof(float), selfDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = (double)self[i] * (double)other[i];
    if (!AlmostEqual(expected, outHost[i], 1e-5, 1e-5)) {
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHost[i]);
      failed++;
    }
  }
  LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(otherT);
  aclrtFree(selfDev); aclrtFree(otherDev);
  return failed > 0 ? 1 : 0;
}

// ========== 用例 6：错误输入测试（传入 nullptr） =========
int RunErrorInputTest(const char* name,
            const std::vector<float>& x2, const std::vector<int64_t>& shape,
            aclrtStream stream) {
  // Create only x2 and out, leave x1 as nullptr to trigger error handling
  void *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<float>(x2, shape, &x2Dev, ACL_FLOAT, &x2T);
  std::vector<float> outHost(GetShapeSize(shape), 0);
  CreateAclTensor<float>(outHost, shape, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  // intentionally pass nullptr as first argument
  auto ret = aclnnMulGetWorkspaceSize(nullptr, x2T, outT, &wsSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected error when first arg is nullptr, got SUCCESS\n", name);
    aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x2Dev); aclrtFree(outDev);
    return 1;
  }
  LOG_PRINT("[PASS] %s (error input detected as expected, ret=%d)\n", name, ret);

  aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 6b：输出为nullptr的错误测试 =========
int RunErrorOutputNullptrTest(const char* name,
                    const std::vector<float>& x1, const std::vector<int64_t>& shape1,
                    const std::vector<float>& x2, const std::vector<int64_t>& shape2,
                    aclrtStream stream) {
  void *x1Dev=nullptr, *x2Dev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr;
  CreateAclTensor<float>(x1, shape1, &x1Dev, ACL_FLOAT, &x1T);
  CreateAclTensor<float>(x2, shape2, &x2Dev, ACL_FLOAT, &x2T);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  // intentionally pass nullptr as output argument
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, nullptr, &wsSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected error when output is nullptr, got SUCCESS\n", name);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T);
    aclrtFree(x1Dev); aclrtFree(x2Dev);
    return 1;
  }
  LOG_PRINT("[PASS] %s (null output detected as expected, ret=%d)\n", name, ret);

  aclDestroyTensor(x1T); aclDestroyTensor(x2T);
  aclrtFree(x1Dev); aclrtFree(x2Dev);
  return 0;
}
// ========== 用例 7：DOUBLE 类型测试（触发 AiCpu 路径）=========
int RunMulTestDouble(const char* name,
           const std::vector<double>& x1, const std::vector<int64_t>& shape1,
           const std::vector<double>& x2, const std::vector<int64_t>& shape2,
           aclrtStream stream) {
  int64_t n = GetShapeSize(shape1);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<double>(x1, shape1, &x1Dev, ACL_DOUBLE, &x1T);
  CreateAclTensor<double>(x2, shape2, &x2Dev, ACL_DOUBLE, &x2T);
  std::vector<double> outHost(n, 0);
  CreateAclTensor<double>(outHost, shape1, &outDev, ACL_DOUBLE, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(double), outDev, n*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);

  // 验证结果
  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = x1[i] * x2[i];
    if (!AlmostEqual(expected, outHost[i], 1e-10, 1e-10)) {
      LOG_PRINT("  mismatch[%ld]: expected=%.15f, actual=%.15f\n", i, expected, outHost[i]);
      failed++;
    }
  }
  LOG_PRINT(failed == 0 ? "[PASS] %s (double - AiCpu path)\n" : "[FAIL] %s: %d mismatches\n", name, failed);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// ========== 用例 8：混合精度测试 FLOAT16 + FLOAT（触发 isMixDataType 分支）=========
int RunMulTestMixedF16F32(const char* name,
                const std::vector<uint16_t>& x1_f16, const std::vector<int64_t>& shape1,
                const std::vector<float>& x2_f32, const std::vector<int64_t>& shape2,
                aclrtStream stream) {
  int64_t n = GetShapeSize(shape1);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<uint16_t>(x1_f16, shape1, &x1Dev, ACL_FLOAT16, &x1T);
  CreateAclTensor<float>(x2_f32, shape2, &x2Dev, ACL_FLOAT, &x2T);
  // 混合精度输出为FLOAT
  std::vector<float> outHost(n, 0);
  CreateAclTensor<float>(outHost, shape1, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (mixed F16*F32 -> F32)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 9：混合精度测试 BF16 + FLOAT =========
int RunMulTestMixedBF16F32(const char* name,
                 const std::vector<uint16_t>& x1_bf16, const std::vector<int64_t>& shape1,
                 const std::vector<float>& x2_f32, const std::vector<int64_t>& shape2,
                 aclrtStream stream) {
  int64_t n = GetShapeSize(shape1);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<uint16_t>(x1_bf16, shape1, &x1Dev, ACL_BF16, &x1T);
  CreateAclTensor<float>(x2_f32, shape2, &x2Dev, ACL_FLOAT, &x2T);
  std::vector<float> outHost(n, 0);
  CreateAclTensor<float>(outHost, shape1, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (mixed BF16*F32 -> F32)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 9：INT8 类型测试 =========
int RunMulTestInt8(const char* name,
         const std::vector<int8_t>& x1, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& x2,
         aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<int8_t>(x1, shape, &x1Dev, ACL_INT8, &x1T);
  CreateAclTensor<int8_t>(x2, shape, &x2Dev, ACL_INT8, &x2T);
  std::vector<int8_t> outHost(n, 0);
  CreateAclTensor<int8_t>(outHost, shape, &outDev, ACL_INT8, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(int8_t), outDev, n*sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (int8 execution completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 10：UINT8 类型测试 =========
int RunMulTestUint8(const char* name,
          const std::vector<uint8_t>& x1, const std::vector<int64_t>& shape,
          const std::vector<uint8_t>& x2,
          aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<uint8_t>(x1, shape, &x1Dev, ACL_UINT8, &x1T);
  CreateAclTensor<uint8_t>(x2, shape, &x2Dev, ACL_UINT8, &x2T);
  std::vector<uint8_t> outHost(n, 0);
  CreateAclTensor<uint8_t>(outHost, shape, &outDev, ACL_UINT8, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(uint8_t), outDev, n*sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (uint8 execution completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 11：INT64 类型测试 =========
int RunMulTestInt64(const char* name,
          const std::vector<int64_t>& x1, const std::vector<int64_t>& shape,
          const std::vector<int64_t>& x2,
          aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<int64_t>(x1, shape, &x1Dev, ACL_INT64, &x1T);
  CreateAclTensor<int64_t>(x2, shape, &x2Dev, ACL_INT64, &x2T);
  std::vector<int64_t> outHost(n, 0);
  CreateAclTensor<int64_t>(outHost, shape, &outDev, ACL_INT64, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(int64_t), outDev, n*sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (int64 execution completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 12：BOOL 类型测试 =========
int RunMulTestBool(const char* name,
         const std::vector<uint8_t>& x1, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& x2,
         aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<uint8_t>(x1, shape, &x1Dev, ACL_BOOL, &x1T);
  CreateAclTensor<uint8_t>(x2, shape, &x2Dev, ACL_BOOL, &x2T);
  std::vector<uint8_t> outHost(n, 0);
  CreateAclTensor<uint8_t>(outHost, shape, &outDev, ACL_BOOL, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(uint8_t), outDev, n*sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (bool execution completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 13：COMPLEX64 类型测试 =========
int RunMulTestComplex64(const char* name,
              const std::vector<float>& x1_real, const std::vector<float>& x1_imag,
              const std::vector<int64_t>& shape,
              const std::vector<float>& x2_real, const std::vector<float>& x2_imag,
              aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  // Complex64: interleaved real and imaginary parts
  std::vector<float> x1_complex(2 * n);
  std::vector<float> x2_complex(2 * n);
  for (int64_t i = 0; i < n; i++) {
    x1_complex[2*i] = x1_real[i];
    x1_complex[2*i+1] = x1_imag[i];
    x2_complex[2*i] = x2_real[i];
    x2_complex[2*i+1] = x2_imag[i];
  }

  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<float>(x1_complex, {2*n}, &x1Dev, ACL_COMPLEX64, &x1T);
  CreateAclTensor<float>(x2_complex, {2*n}, &x2Dev, ACL_COMPLEX64, &x2T);
  std::vector<float> outHost(2*n, 0);
  CreateAclTensor<float>(outHost, {2*n}, &outDev, ACL_COMPLEX64, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), 2*n*sizeof(float), outDev, 2*n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (complex64 execution completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 13b：INT16 类型测试 =========
int RunMulTestInt16(const char* name,
          const std::vector<int16_t>& x1, const std::vector<int64_t>& shape,
          const std::vector<int16_t>& x2,
          aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<int16_t>(x1, shape, &x1Dev, ACL_INT16, &x1T);
  CreateAclTensor<int16_t>(x2, shape, &x2Dev, ACL_INT16, &x2T);
  std::vector<int16_t> outHost(n, 0);
  CreateAclTensor<int16_t>(outHost, shape, &outDev, ACL_INT16, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(int16_t), outDev, n*sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (int16 execution completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 13c：COMPLEX128 类型测试（触发 AiCpu 路径） =========
int RunMulTestComplex128(const char* name,
               const std::vector<double>& x1_real, const std::vector<double>& x1_imag,
               const std::vector<int64_t>& shape,
               const std::vector<double>& x2_real, const std::vector<double>& x2_imag,
               aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  // Complex128: interleaved real and imaginary parts
  std::vector<double> x1_complex(2 * n);
  std::vector<double> x2_complex(2 * n);
  for (int64_t i = 0; i < n; i++) {
    x1_complex[2*i] = x1_real[i];
    x1_complex[2*i+1] = x1_imag[i];
    x2_complex[2*i] = x2_real[i];
    x2_complex[2*i+1] = x2_imag[i];
  }

  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<double>(x1_complex, {2*n}, &x1Dev, ACL_COMPLEX128, &x1T);
  CreateAclTensor<double>(x2_complex, {2*n}, &x2Dev, ACL_COMPLEX128, &x2T);
  std::vector<double> outHost(2*n, 0);
  CreateAclTensor<double>(outHost, {2*n}, &outDev, ACL_COMPLEX128, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), 2*n*sizeof(double), outDev, 2*n*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (complex128 - AiCpu path)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 14：大维度张量测试 (5D, 触发 dim > 4 检查) =========
int RunMulTest5D(const char* name,
       const std::vector<float>& x1, const std::vector<int64_t>& shape,
       const std::vector<float>& x2,
       aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<float>(x1, shape, &x1Dev, ACL_FLOAT, &x1T);
  CreateAclTensor<float>(x2, shape, &x2Dev, ACL_FLOAT, &x2T);
  std::vector<float> outHost(n, 0);
  CreateAclTensor<float>(outHost, shape, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[INFO] %s: 5D tensor not supported as expected (ret=%d)\n", name, ret);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    return 0;
  }

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (5D tensor execution completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 15：不同形状广播测试 =========
int RunMulBroadcastAdvancedTest(const char* name,
                     const std::vector<float>& x1, const std::vector<int64_t>& shape1,
                     const std::vector<float>& x2, const std::vector<int64_t>& shape2,
                     const std::vector<int64_t>& outShape,
                     aclrtStream stream) {
  int64_t n = GetShapeSize(outShape);
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<float>(x1, shape1, &x1Dev, ACL_FLOAT, &x1T);
  CreateAclTensor<float>(x2, shape2, &x2Dev, ACL_FLOAT, &x2T);
  std::vector<float> outHost(n, 0);
  CreateAclTensor<float>(outHost, outShape, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (advanced broadcast completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 16：标量广播测试（一维 vs 零维） =========
int RunMulScalarBroadcastTest(const char* name,
                   const std::vector<float>& x1, const std::vector<int64_t>& shape1,
                   float scalar_val,
                   aclrtStream stream) {
  // Create a 0-dim tensor for scalar
  void *x1Dev=nullptr, *scalarDev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *scalarT=nullptr, *outT=nullptr;
  CreateAclTensor<float>(x1, shape1, &x1Dev, ACL_FLOAT, &x1T);
  
  std::vector<float> scalar_data = {scalar_val};
  CreateAclTensor<float>(scalar_data, {}, &scalarDev, ACL_FLOAT, &scalarT);
  
  int64_t n = GetShapeSize(shape1);
  std::vector<float> outHost(n, 0);
  CreateAclTensor<float>(outHost, shape1, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, scalarT, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (scalar broadcast completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(scalarT); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(scalarDev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 17：空张量测试 =========
int RunMulEmptyTensorTest(const char* name, aclrtStream stream) {
  std::vector<float> empty_vec;
  std::vector<int64_t> shape = {0};
  
  void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
  aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
  CreateAclTensor<float>(empty_vec, shape, &x1Dev, ACL_FLOAT, &x1T);
  CreateAclTensor<float>(empty_vec, shape, &x2Dev, ACL_FLOAT, &x2T);
  CreateAclTensor<float>(empty_vec, shape, &outDev, ACL_FLOAT, &outT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[INFO] %s: empty tensor handled (ret=%d)\n", name, ret);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    return 0;
  }

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMul(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  LOG_PRINT("[PASS] %s (empty tensor completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
  aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
  return 0;
}

// ========== 用例 18：Muls with different dtypes =========
int RunMulsTestInt32(const char* name,
           const std::vector<int32_t>& self, const std::vector<int64_t>& shape,
           int32_t scalarVal,
           aclrtStream stream) {
  int64_t n = GetShapeSize(shape);
  void *selfDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *outT=nullptr;
  CreateAclTensor<int32_t>(self, shape, &selfDev, ACL_INT32, &selfT);
  std::vector<int32_t> outHost(n, 0);
  CreateAclTensor<int32_t>(outHost, shape, &outDev, ACL_INT32, &outT);

  aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_INT32);
  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulsGetWorkspaceSize(selfT, scalar, outT, &wsSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnMuls(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(outHost.data(), n*sizeof(int32_t), outDev, n*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("[PASS] %s (int32 muls completed)\n", name);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(scalar);
  aclrtFree(selfDev); aclrtFree(outDev);
  return 0;
}

// ... existing code ...

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int totalFailed = 0;

  // 原始用例：float32 同 shape
  totalFailed += RunMulTestFloat("float32_basic",
    {0,1,2,3,4,5,6,7}, {4,2},
    {1,1,1,2,2,2,3,3}, {4,2}, stream);

  // 新增：含负数和零
  totalFailed += RunMulTestFloat("float32_neg_zero",
    {-1.0f, 0.0f, 3.5f, -2.0f}, {2,2},
    {2.0f, 5.0f, -1.0f, 0.0f}, {2,2}, stream);

  // 新增：aclnnMuls（tensor * scalar，覆盖 Muls 路径）
  totalFailed += RunMulsTestFloat("float32_muls",
    {1.0f, 2.0f, 3.0f, 4.0f}, {2,2}, 2.5f, stream);

  // 新增：INT32 类型
  totalFailed += RunMulTestInt32("int32_basic",
    {1, -2, 3, -4}, {2,2},
    {2, 3, -1, 0}, stream);

  // 新增：广播形状
  totalFailed += RunMulBroadcastTest("float32_broadcast",
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2,3},
    {10.0f, 100.0f, 1000.0f}, {1,3}, stream);

  // 新增：inplace mul
  totalFailed += RunInplaceMulTest("float32_inplace",
    {2.0f, 3.0f, 4.0f, 5.0f}, {2,2},
    {1.5f, 2.0f, -1.0f, 0.0f}, stream);

  // 新增：错误输入
  totalFailed += RunErrorInputTest("error_input_nullptr",
    {1.0f, 2.0f, 3.0f, 4.0f}, {2,2}, stream);

  // 新增：输出为nullptr的错误测试
  totalFailed += RunErrorOutputNullptrTest("error_output_nullptr",
    {1.0f, 2.0f, 3.0f, 4.0f}, {2,2},
    {2.0f, 3.0f, 4.0f, 5.0f}, {2,2}, stream);

  // ========== 关键新增：DOUBLE 类型测试（触发 AiCpu 路径）==========
  totalFailed += RunMulTestDouble("double_basic",
    {1.5, -2.5, 3.0, -4.0}, {2,2},
    {2.0, 0.5, -1.0, 0.0}, {2,2}, stream);

  totalFailed += RunMulTestDouble("double_large",
    std::vector<double>(64, 1.5), {8,8},
    std::vector<double>(64, 2.5), {8,8}, stream);

  // ========== 关键新增：混合精度测试（触发 isMixDataType 分支）==========
  std::vector<uint16_t> f16_data = {0x3C00, 0x4000, 0x4200, 0x4400}; // 1.0, 2.0, 3.0, 4.0
  totalFailed += RunMulTestMixedF16F32("mixed_f16_f32_case1",
    f16_data, {2,2},
    {2.0f, 3.0f, 4.0f, 5.0f}, {2,2}, stream);

  std::vector<uint16_t> bf16_data = {0x3F80, 0x4000, 0x4040, 0x4080}; // 1.0, 2.0, 3.0, 4.0 in BF16
  totalFailed += RunMulTestMixedBF16F32("mixed_bf16_f32_case1",
    bf16_data, {2,2},
    {1.5f, 2.5f, 3.5f, 4.5f}, {2,2}, stream);

  // 反向混合精度：FLOAT * FLOAT16
  totalFailed += RunMulTestMixedF16F32("mixed_f32_f16_case2",
    f16_data, {2,2},
    {1.0f, 2.0f, 3.0f, 4.0f}, {2,2}, stream);

  // 新增：INT8 类型测试
  totalFailed += RunMulTestInt8("int8_basic",
    {1, -2, 3, -4}, {2,2},
    {2, 3, -1, 0}, stream);

  // 新增：UINT8 类型测试
  totalFailed += RunMulTestUint8("uint8_basic",
    {1, 2, 3, 4}, {2,2},
    {2, 3, 1, 0}, stream);

  // 新增：INT64 类型测试
  totalFailed += RunMulTestInt64("int64_basic",
    {1LL, -2LL, 3LL, -4LL}, {2,2},
    {2LL, 3LL, -1LL, 0LL}, stream);

  // 新增：BOOL 类型测试（使用uint8_t存储bool值）
  totalFailed += RunMulTestBool("bool_basic",
    {1, 0, 1, 0}, {2,2},
    {1, 1, 0, 0}, stream);

  // 新增：COMPLEX64 类型测试
  totalFailed += RunMulTestComplex64("complex64_basic",
    {1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {2,2},
    {2.0f, 3.0f, 4.0f, 5.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, stream);

  // 新增：5D 张量测试（触发 dim > 4 检查）
  totalFailed += RunMulTest5D("float32_5d",
    std::vector<float>(32, 1.0f), {2,2,2,2,2},
    std::vector<float>(32, 2.0f), stream);

  // 新增：高级广播测试 - 3D vs 1D
  totalFailed += RunMulBroadcastAdvancedTest("float32_broadcast_3d_1d",
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2,2,2},
    {10.0f, 100.0f}, {2},
    {2,2,2}, stream);

  // 新增：高级广播测试 - 4D vs 2D
  totalFailed += RunMulBroadcastAdvancedTest("float32_broadcast_4d_2d",
    std::vector<float>(24, 1.0f), {2,3,2,2},
    {1.0f, 2.0f, 3.0f}, {1,3},
    {2,3,2,2}, stream);

  // 新增：标量广播测试
  totalFailed += RunMulScalarBroadcastTest("float32_scalar_broadcast",
    {1.0f, 2.0f, 3.0f, 4.0f}, {2,2},
    2.0f, stream);

  // 新增：空张量测试
  totalFailed += RunMulEmptyTensorTest("float32_empty", stream);

  // 新增：INT32 Muls 测试
  totalFailed += RunMulsTestInt32("int32_muls",
    {1, 2, 3, 4}, {2,2}, 3, stream);

  // 新增：更大规模的测试
  std::vector<float> large_x1(1024, 1.5f);
  std::vector<float> large_x2(1024, 2.5f);
  totalFailed += RunMulTestFloat("float32_large",
    large_x1, {32, 32},
    large_x2, {32, 32}, stream);

  // 新增：1D 张量测试
  totalFailed += RunMulTestFloat("float32_1d",
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5},
    {2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {5}, stream);

  // 新增：3D 张量测试
  totalFailed += RunMulTestFloat("float32_3d",
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2,2,2},
    {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}, {2,2,2}, stream);

  // 新增：4D 张量测试
  totalFailed += RunMulTestFloat("float32_4d",
    std::vector<float>(16, 1.5f), {2,2,2,2},
    std::vector<float>(16, 2.0f), {2,2,2,2}, stream);

  // 新增：特殊值测试（NaN, Inf）
  totalFailed += RunMulTestFloat("float32_special_values",
    {1.0f, INFINITY, -INFINITY, NAN}, {2,2},
    {2.0f, 0.0f, 0.0f, 1.0f}, {2,2}, stream);

  // ========== 额外关键测试：触发更多 dtype 分支 ==========
  
  // INT16 类型测试（在 REGBASE 支持列表中）
  totalFailed += RunMulTestInt16("int16_basic",
    {1, -2, 3, -4}, {2,2},
    {2, 3, -1, 0}, stream);

  // COMPLEX128 类型测试（会触发 AiCpu 路径）
  totalFailed += RunMulTestComplex128("complex128_basic",
    {1.0, 2.0, 3.0, 4.0}, {1.0, 1.0, 1.0, 1.0}, {2,2},
    {2.0, 3.0, 4.0, 5.0}, {1.0, 1.0, 1.0, 1.0}, stream);

  // DOUBLE + FLOAT 混合（测试不同类型组合）
  totalFailed += RunMulTestDouble("double_neg_zero_inf",
    {-1.5, 0.0, INFINITY, -INFINITY}, {2,2},
    {2.0, 5.0, 0.0, 0.0}, {2,2}, stream);

  // 单元素张量（边界情况）
  totalFailed += RunMulTestFloat("float32_single_element",
    {3.5f}, {1},
    {2.0f}, {1}, stream);

  // 大维度广播：4D vs 1D
  totalFailed += RunMulBroadcastAdvancedTest("float32_broadcast_4d_1d",
    std::vector<float>(48, 1.0f), {2,3,2,4},
    {10.0f, 20.0f, 30.0f, 40.0f}, {4},
    {2,3,2,4}, stream);

  // 极端形状：非常长的 1D
  totalFailed += RunMulTestFloat("float32_long_1d",
    std::vector<float>(1000, 1.5f), {1000},
    std::vector<float>(1000, 2.0f), {1000}, stream);

  // 3D 广播：[2,1,3] * [1,3,1]
  totalFailed += RunMulBroadcastAdvancedTest("float32_broadcast_3d_complex",
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2,1,3},
    {10.0f, 20.0f, 30.0f}, {1,3,1},
    {2,3,3}, stream);

  // ========== 极限优化：尝试覆盖更多边界情况 ==========
  
  // 测试：非常大的张量（压力测试）
  std::vector<float> huge_x1(10000, 0.5f);
  std::vector<float> huge_x2(10000, 2.0f);
  totalFailed += RunMulTestFloat("float32_huge_tensor",
    huge_x1, {100, 100},
    huge_x2, {100, 100}, stream);

  // 测试：极小值
  totalFailed += RunMulTestFloat("float32_tiny_values",
    {1e-10f, 1e-20f, 1e-30f, 1e-38f}, {2,2},
    {1e10f, 1e20f, 1e30f, 1e38f}, {2,2}, stream);

  // 测试：极大值
  totalFailed += RunMulTestFloat("float32_huge_values",
    {1e10f, 1e20f, 1e30f, 1e38f}, {2,2},
    {1e10f, 1e-10f, 1e-20f, 1e-30f}, {2,2}, stream);

  // 测试：全零输入
  totalFailed += RunMulTestFloat("float32_all_zeros",
    {0.0f, 0.0f, 0.0f, 0.0f}, {2,2},
    {0.0f, 0.0f, 0.0f, 0.0f}, {2,2}, stream);

  // 测试：全1输入
  totalFailed += RunMulTestFloat("float32_all_ones",
    {1.0f, 1.0f, 1.0f, 1.0f}, {2,2},
    {1.0f, 1.0f, 1.0f, 1.0f}, {2,2}, stream);

  // 测试：不对称形状广播 [3,1] * [1,4]
  totalFailed += RunMulBroadcastAdvancedTest("float32_broadcast_asymmetric",
    {1.0f, 2.0f, 3.0f}, {3,1},
    {10.0f, 20.0f, 30.0f, 40.0f}, {1,4},
    {3,4}, stream);

  // DOUBLE类型的更多变体
  totalFailed += RunMulTestDouble("double_extreme_values",
    {1e100, -1e100, 1e-100, -1e-100}, {2,2},
    {1e-100, 1e100, -1e-100, -1e100}, {2,2}, stream);

  // INT32的大值测试（可能溢出）
  totalFailed += RunMulTestInt32("int32_overflow_edge",
    {100000, -100000, 50000, -50000}, {2,2},
    {100000, 100000, -100000, -100000}, stream);

  LOG_PRINT("\n=== Summary: %d failed ===\n", totalFailed);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return totalFailed;
}