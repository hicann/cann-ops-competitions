/**
 * Pow Operator Complete Test Suite
 * Testing all 7 APIs with comprehensive coverage
 * Based on CANN official examples
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_pow.h"
#include "aclnn_pow_tensor_tensor.h"
#include "aclnn_exp2.h"

#define CHECK_RET(cond, return_expr) \
  do { \
    if (!(cond)) { \
      return_expr; \
    } \
  } while (0)

#define LOG_PRINT(message, ...) \
  do { \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int g_totalTests = 0;
int g_passedTests = 0;

#define RUN_TEST(testFunc) \
  do { \
    g_totalTests++; \
    std::cout << "[" << g_totalTests << "] Running " << #testFunc << "... "; \
    std::cout.flush(); \
    if (testFunc()) { \
      std::cout << "PASS" << std::endl; \
      g_passedTests++; \
    } else { \
      std::cout << "FAIL" << std::endl; \
    } \
  } while(0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
bool VerifyResult(const std::vector<T>& result, const std::vector<T>& expected,
                  float atol = 1e-5f, float rtol = 1e-5f) {
  if (result.size() != expected.size()) {
    printf("Size mismatch: result=%zu, expected=%zu\n", result.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < result.size(); i++) {
    if constexpr (std::is_floating_point_v<T>) {
      float diff = std::abs(result[i] - expected[i]);
      float tolerance = atol + rtol * std::abs(expected[i]);
      if (diff > tolerance) {
        printf("Mismatch at [%zu]: expected=%f, actual=%f, diff=%f\n",
               i, expected[i], result[i], diff);
        return false;
      }
    } else {
      if (result[i] != expected[i]) {
        printf("Mismatch at [%zu]: expected=%d, actual=%d\n",
               i, (int)expected[i], (int)result[i]);
        return false;
      }
    }
  }
  return true;
}

// Test 1: TensorScalar - x^2
bool Test1_TensorScalar_Power2() {
  aclrtStream stream;
  aclTensor *self, *out;
  aclScalar *exponent;
  void *selfAddr, *outAddr;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfData = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> outData(4, 0);
  float expVal = 2.0f;
  std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowTensorScalar(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(4);
  aclrtMemcpy(result.data(), 4 * sizeof(float), outAddr, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(out); aclDestroyScalar(exponent);
  aclrtFree(selfAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected);
}

// Test 2: TensorScalar - x^0 = 1
bool Test2_TensorScalar_PowerOfZero() {
  aclrtStream stream;
  aclTensor *self, *out;
  aclScalar *exponent;
  void *selfAddr, *outAddr;

  std::vector<int64_t> shape = {3};
  std::vector<float> selfData = {2.0f, 5.0f, 100.0f};
  std::vector<float> outData(3, 0);
  float expVal = 0.0f;
  std::vector<float> expected = {1.0f, 1.0f, 1.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowTensorScalar(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(3);
  aclrtMemcpy(result.data(), 3 * sizeof(float), outAddr, 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(out); aclDestroyScalar(exponent);
  aclrtFree(selfAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected);
}

// Test 3: TensorScalar - x^0.5 (sqrt)
bool Test3_TensorScalar_Sqrt() {
  aclrtStream stream;
  aclTensor *self, *out;
  aclScalar *exponent;
  void *selfAddr, *outAddr;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfData = {4.0f, 9.0f, 16.0f, 25.0f};
  std::vector<float> outData(4, 0);
  float expVal = 0.5f;
  std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowTensorScalar(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(4);
  aclrtMemcpy(result.data(), 4 * sizeof(float), outAddr, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(out); aclDestroyScalar(exponent);
  aclrtFree(selfAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.01f, 0.01f);
}

// Test 4: TensorScalar - x^-1 (reciprocal)
bool Test4_TensorScalar_Reciprocal() {
  aclrtStream stream;
  aclTensor *self, *out;
  aclScalar *exponent;
  void *selfAddr, *outAddr;

  std::vector<int64_t> shape = {3};
  std::vector<float> selfData = {2.0f, 4.0f, 10.0f};
  std::vector<float> outData(3, 0);
  float expVal = -1.0f;
  std::vector<float> expected = {0.5f, 0.25f, 0.1f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowTensorScalar(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(3);
  aclrtMemcpy(result.data(), 3 * sizeof(float), outAddr, 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(out); aclDestroyScalar(exponent);
  aclrtFree(selfAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.001f, 0.001f);
}

// Test 5: ScalarTensor - 2^x
bool Test5_ScalarTensor_Base2() {
  aclrtStream stream;
  aclTensor *exp, *out;
  aclScalar *base;
  void *expAddr, *outAddr;

  std::vector<int64_t> shape = {3};
  std::vector<float> expData = {1.0f, 2.0f, 3.0f};
  std::vector<float> outData(3, 0);
  float baseVal = 2.0f;
  std::vector<float> expected = {2.0f, 4.0f, 8.0f};

  Init(0, &stream);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  base = aclCreateScalar(&baseVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowScalarTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowScalarTensor(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(3);
  aclrtMemcpy(result.data(), 3 * sizeof(float), outAddr, 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(exp); aclDestroyTensor(out); aclDestroyScalar(base);
  aclrtFree(expAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.01f, 0.01f);
}

// Test 6: TensorTensor - basic
bool Test6_TensorTensor_Basic() {
  aclrtStream stream;
  aclTensor *self, *exp, *out;
  void *selfAddr, *expAddr, *outAddr;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfData = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> expData = {1.0f, 2.0f, 0.5f, 3.0f};
  std::vector<float> outData(4, 0);
  std::vector<float> expected;
  for (int i = 0; i < 4; i++) expected.push_back(std::pow(selfData[i], expData[i]));

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowTensorTensor(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(4);
  aclrtMemcpy(result.data(), 4 * sizeof(float), outAddr, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(exp); aclDestroyTensor(out);
  aclrtFree(selfAddr); aclrtFree(expAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.001f, 0.001f);
}

// Test 7: TensorTensor - broadcast
bool Test7_TensorTensor_Broadcast() {
  aclrtStream stream;
  aclTensor *self, *exp, *out;
  void *selfAddr, *expAddr, *outAddr;

  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> expShape = {3};
  std::vector<float> selfData = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> expData = {1.0f, 2.0f, 3.0f};
  std::vector<float> outData(6, 0);
  std::vector<float> expected;
  for (int i = 0; i < 6; i++) expected.push_back(std::pow(selfData[i], expData[i % 3]));

  Init(0, &stream);
  CreateAclTensor(selfData, selfShape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, expShape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, selfShape, &outAddr, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowTensorTensor(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(6);
  aclrtMemcpy(result.data(), 6 * sizeof(float), outAddr, 6 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(exp); aclDestroyTensor(out);
  aclrtFree(selfAddr); aclrtFree(expAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.01f, 0.01f);
}

// Test 8: TensorTensor - 0^x
bool Test8_TensorTensor_BaseOfZero() {
  aclrtStream stream;
  aclTensor *self, *exp, *out;
  void *selfAddr, *expAddr, *outAddr;

  std::vector<int64_t> shape = {3};
  std::vector<float> selfData = {0.0f, 0.0f, 0.0f};
  std::vector<float> expData = {1.0f, 2.0f, 3.0f};
  std::vector<float> outData(3, 0);
  std::vector<float> expected = {0.0f, 0.0f, 0.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnPowTensorTensor(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(3);
  aclrtMemcpy(result.data(), 3 * sizeof(float), outAddr, 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(exp); aclDestroyTensor(out);
  aclrtFree(selfAddr); aclrtFree(expAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected);
}

// Test 9: InplaceTensorScalar
bool Test9_InplaceTensorScalar() {
  aclrtStream stream;
  aclTensor *self;
  aclScalar *exponent;
  void *selfAddr;

  std::vector<int64_t> shape = {3};
  std::vector<float> selfData = {2.0f, 3.0f, 4.0f};
  float expVal = 3.0f;
  std::vector<float> expected = {8.0f, 27.0f, 64.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnInplacePowTensorScalar(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(3);
  aclrtMemcpy(result.data(), 3 * sizeof(float), selfAddr, 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyScalar(exponent);
  aclrtFree(selfAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.01f, 0.01f);
}

// Test 10: InplaceTensorTensor
bool Test10_InplaceTensorTensor() {
  aclrtStream stream;
  aclTensor *self, *exp;
  void *selfAddr, *expAddr;

  std::vector<int64_t> shape = {3};
  std::vector<float> selfData = {2.0f, 3.0f, 4.0f};
  std::vector<float> expData = {2.0f, 2.0f, 2.0f};
  std::vector<float> expected = {4.0f, 9.0f, 16.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exp);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnInplacePowTensorTensorGetWorkspaceSize(self, exp, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnInplacePowTensorTensor(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(3);
  aclrtMemcpy(result.data(), 3 * sizeof(float), selfAddr, 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(exp);
  aclrtFree(selfAddr); aclrtFree(expAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.01f, 0.01f);
}

// Test 11: Exp2 - 2^x
bool Test11_Exp2() {
  aclrtStream stream;
  aclTensor *self, *out;
  void *selfAddr, *outAddr;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> outData(4, 0);
  std::vector<float> expected = {1.0f, 2.0f, 4.0f, 8.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnExp2(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(4);
  aclrtMemcpy(result.data(), 4 * sizeof(float), outAddr, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self); aclDestroyTensor(out);
  aclrtFree(selfAddr); aclrtFree(outAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.01f, 0.01f);
}

// Test 12: InplaceExp2
bool Test12_InplaceExp2() {
  aclrtStream stream;
  aclTensor *self;
  void *selfAddr;

  std::vector<int64_t> shape = {3};
  std::vector<float> selfData = {0.0f, 1.0f, 2.0f};
  std::vector<float> expected = {1.0f, 2.0f, 4.0f};

  Init(0, &stream);
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  aclnnInplaceExp2GetWorkspaceSize(self, &workspaceSize, &executor);

  void* workspace = nullptr;
  if (workspaceSize > 0) aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclnnInplaceExp2(workspace, workspaceSize, executor, stream);
  aclrtSynchronizeStream(stream);

  std::vector<float> result(3);
  aclrtMemcpy(result.data(), 3 * sizeof(float), selfAddr, 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  aclDestroyTensor(self);
  aclrtFree(selfAddr);
  if (workspace) aclrtFree(workspace);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  return VerifyResult(result, expected, 0.01f, 0.01f);
}

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "   Pow Operator Test Suite" << std::endl;
  std::cout << "   Testing all 7 APIs" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  RUN_TEST(Test1_TensorScalar_Power2);
  RUN_TEST(Test2_TensorScalar_PowerOfZero);
  RUN_TEST(Test3_TensorScalar_Sqrt);
  RUN_TEST(Test4_TensorScalar_Reciprocal);
  RUN_TEST(Test5_ScalarTensor_Base2);
  RUN_TEST(Test6_TensorTensor_Basic);
  RUN_TEST(Test7_TensorTensor_Broadcast);
  RUN_TEST(Test8_TensorTensor_BaseOfZero);
  RUN_TEST(Test9_InplaceTensorScalar);
  RUN_TEST(Test10_InplaceTensorTensor);
  RUN_TEST(Test11_Exp2);
  RUN_TEST(Test12_InplaceExp2);

  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "           Test Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Total Tests: " << g_totalTests << std::endl;
  std::cout << "Passed:      " << g_passedTests << std::endl;
  std::cout << "Failed:      " << (g_totalTests - g_passedTests) << std::endl;
  std::cout << "Pass Rate:   " << (g_totalTests > 0 ? (100.0 * g_passedTests / g_totalTests) : 0.0) << "%" << std::endl;
  std::cout << "========================================" << std::endl;

  return (g_passedTests == g_totalTests) ? 0 : 1;
}
