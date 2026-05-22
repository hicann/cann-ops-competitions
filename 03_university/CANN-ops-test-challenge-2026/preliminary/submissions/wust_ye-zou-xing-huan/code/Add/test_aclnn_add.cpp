/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Add算子稳定测试版本 - 避免使用inplace版本
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_add.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) shapeSize *= i;
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, 
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret); return ret);
  
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, 
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

bool AlmostEqual(double expected, double actual, double atol, double rtol) {
  if (std::isnan(expected) && std::isnan(actual)) return true;
  if (std::isinf(expected) && std::isinf(actual))
    return (expected > 0) == (actual > 0);
  return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

// Test 1: 基本加法 FLOAT32
int TestAdd_Float32_Basic(aclrtStream stream) {
  LOG_PRINT("[TEST] Add_Float32_Basic - ");
  std::vector<int64_t> shape = {4, 2};
  void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *otherT=nullptr, *outT=nullptr;

  std::vector<float> self = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> other = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHost(GetShapeSize(shape), 0);
  float alpha = 1.0f;

  CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CreateAclTensor(other, shape, &otherDev, ACL_FLOAT, &otherT);
  CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);
  aclScalar* alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddGetWorkspaceSize(selfT, otherT, alphaScalar, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("FAILED (GetWorkspaceSize=%d)\n", ret);
    return 1;
  }

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnAdd(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  int64_t n = GetShapeSize(shape);
  std::vector<float> result(n);
  aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = (double)self[i] + (double)alpha * (double)other[i];
    if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
  }
  LOG_PRINT(failed == 0 ? "PASS\n" : "FAIL (%d mismatches)\n", failed);

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alphaScalar);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// Test 2: 带alpha的加法
int TestAdd_Float32_Alpha(aclrtStream stream) {
  LOG_PRINT("[TEST] Add_Float32_Alpha - ");
  std::vector<int64_t> shape = {2, 2};
  void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *otherT=nullptr, *outT=nullptr;

  std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> other = {2.0f, 3.0f, 1.0f, 2.0f};
  std::vector<float> outHost(GetShapeSize(shape), 0);
  float alpha = 2.5f;

  CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CreateAclTensor(other, shape, &otherDev, ACL_FLOAT, &otherT);
  CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);
  aclScalar* alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddGetWorkspaceSize(selfT, otherT, alphaScalar, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) return 1;

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnAdd(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  int64_t n = GetShapeSize(shape);
  std::vector<float> result(n);
  aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = (double)self[i] + (double)alpha * (double)other[i];
    if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
  }
  LOG_PRINT(failed == 0 ? "PASS\n" : "FAIL\n");

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alphaScalar);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// Test 3: INT32类型
int TestAdd_Int32_Basic(aclrtStream stream) {
  LOG_PRINT("[TEST] Add_Int32_Basic - ");
  std::vector<int64_t> shape = {4};
  void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *otherT=nullptr, *outT=nullptr;

  std::vector<int32_t> self = {10, 20, 30, 40};
  std::vector<int32_t> other = {5, -3, 0, 15};
  std::vector<int32_t> outHost(GetShapeSize(shape), 0);
  float alpha = 1.0f;

  CreateAclTensor(self, shape, &selfDev, ACL_INT32, &selfT);
  CreateAclTensor(other, shape, &otherDev, ACL_INT32, &otherT);
  CreateAclTensor(outHost, shape, &outDev, ACL_INT32, &outT);
  aclScalar* alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddGetWorkspaceSize(selfT, otherT, alphaScalar, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) return 1;

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnAdd(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  int64_t n = GetShapeSize(shape);
  std::vector<int32_t> result(n);
  aclrtMemcpy(result.data(), n*sizeof(int32_t), outDev, n*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    int32_t expected = self[i] + (int32_t)alpha * other[i];
    if (result[i] != expected) failed++;
  }
  LOG_PRINT(failed == 0 ? "PASS\n" : "FAIL\n");

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alphaScalar);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// Test 4: 广播场景
int TestAdd_Broadcast(aclrtStream stream) {
  LOG_PRINT("[TEST] Add_Broadcast_2Dx1D - ");
  std::vector<int64_t> shape1 = {2, 3};
  std::vector<int64_t> shape2 = {1, 3};
  void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *otherT=nullptr, *outT=nullptr;

  std::vector<float> self = {1, 2, 3, 4, 5, 6};
  std::vector<float> other = {2, 1, 3};
  std::vector<float> outHost(GetShapeSize(shape1), 0);
  float alpha = 1.0f;

  CreateAclTensor(self, shape1, &selfDev, ACL_FLOAT, &selfT);
  CreateAclTensor(other, shape2, &otherDev, ACL_FLOAT, &otherT);
  CreateAclTensor(outHost, shape1, &outDev, ACL_FLOAT, &outT);
  aclScalar* alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddGetWorkspaceSize(selfT, otherT, alphaScalar, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) return 1;

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnAdd(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  int64_t n = GetShapeSize(shape1);
  std::vector<float> result(n);
  aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t otherIdx = i % 3;
    double expected = (double)self[i] + (double)alpha * (double)other[otherIdx];
    if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
  }
  LOG_PRINT(failed == 0 ? "PASS\n" : "FAIL\n");

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alphaScalar);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// Test 5: 负数和零值
int TestAdd_Negative_Zero(aclrtStream stream) {
  LOG_PRINT("[TEST] Add_Negative_Zero - ");
  std::vector<int64_t> shape = {2, 2};
  void *selfDev=nullptr, *otherDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *otherT=nullptr, *outT=nullptr;

  std::vector<float> self = {-1.0f, 0.0f, 3.5f, -2.0f};
  std::vector<float> other = {2.0f, -5.0f, 0.0f, 1.0f};
  std::vector<float> outHost(GetShapeSize(shape), 0);
  float alpha = 1.0f;

  CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CreateAclTensor(other, shape, &otherDev, ACL_FLOAT, &otherT);
  CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);
  aclScalar* alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddGetWorkspaceSize(selfT, otherT, alphaScalar, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) return 1;

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnAdd(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  int64_t n = GetShapeSize(shape);
  std::vector<float> result(n);
  aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = (double)self[i] + (double)alpha * (double)other[i];
    if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
  }
  LOG_PRINT(failed == 0 ? "PASS\n" : "FAIL\n");

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alphaScalar);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

// Test 6: 异常测试 - nullptr
int TestInvalid_Nullptr(aclrtStream stream) {
  LOG_PRINT("[TEST] Invalid_Nullptr - ");
  aclTensor* tensor = nullptr;
  aclScalar* scalar = nullptr;
  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;

  auto ret = aclnnAddGetWorkspaceSize(nullptr, tensor, scalar, tensor, &wsSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("PASS (correctly rejected)\n");
    return 0;
  }
  LOG_PRINT("FAIL (should reject nullptr)\n");
  return 1;
}

// Test 7: aclnnAdds测试
int TestAdds_Float32(aclrtStream stream) {
  LOG_PRINT("[TEST] Adds_Float32 - ");
  std::vector<int64_t> shape = {2, 2};
  void *selfDev=nullptr, *outDev=nullptr;
  aclTensor *selfT=nullptr, *outT=nullptr;

  std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
  float other = 5.0f;
  float alpha = 2.0f;
  std::vector<float> outHost(GetShapeSize(shape), 0);

  CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
  CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);

  aclScalar* otherScalar = aclCreateScalar(&other, ACL_FLOAT);
  aclScalar* alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);

  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddsGetWorkspaceSize(selfT, otherScalar, alphaScalar, outT, &wsSize, &executor);
  if (ret != ACL_SUCCESS) return 1;

  void* wsAddr = nullptr;
  if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnAdds(wsAddr, wsSize, executor, stream);
  aclrtSynchronizeStream(stream);

  int64_t n = GetShapeSize(shape);
  std::vector<float> result(n);
  aclrtMemcpy(result.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  int failed = 0;
  for (int64_t i = 0; i < n; i++) {
    double expected = (double)self[i] + (double)alpha * (double)other;
    if (!AlmostEqual(expected, result[i], 1e-5, 1e-5)) failed++;
  }
  LOG_PRINT(failed == 0 ? "PASS\n" : "FAIL\n");

  if (wsAddr) aclrtFree(wsAddr);
  aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(otherScalar); aclDestroyScalar(alphaScalar);
  aclrtFree(selfDev); aclrtFree(outDev);
  return failed > 0 ? 1 : 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed\n"); return ret);

  int totalFailed = 0;
  totalFailed += TestAdd_Float32_Basic(stream);
  totalFailed += TestAdd_Float32_Alpha(stream);
  totalFailed += TestAdd_Int32_Basic(stream);
  totalFailed += TestAdd_Broadcast(stream);
  totalFailed += TestAdd_Negative_Zero(stream);
  totalFailed += TestAdds_Float32(stream);
  totalFailed += TestInvalid_Nullptr(stream);

  LOG_PRINT("\n=== Summary: %d/%d tests passed ===\n", 7-totalFailed, 7);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return totalFailed;
}
