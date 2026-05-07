#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "acl/acl.h"

#if __has_include("aclnnop/aclnn_mul.h")
#include "aclnnop/aclnn_mul.h"
#else
#include "../op_api/aclnn_mul.h"
#endif

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif

#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
#endif

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int64_t GetTypeSize(aclDataType dtype) {
  if (dtype == ACL_FLOAT) return 4;
  if (dtype == ACL_FLOAT16) return 2;
  if (dtype == ACL_BF16) return 2;
  if (dtype == ACL_DOUBLE) return 8;
#ifdef ACL_COMPLEX32
  if (dtype == ACL_COMPLEX32) return 4;
#endif
  if (dtype == ACL_INT64) return 8;
  if (dtype == ACL_INT32) return 4;
  if (dtype == ACL_INT16) return 2;
  if (dtype == ACL_INT8) return 1;
  if (dtype == ACL_UINT8) return 1;
  if (dtype == ACL_BOOL) return 1;
  if (dtype == ACL_COMPLEX64) return 8;
  if (dtype == ACL_COMPLEX128) return 16;
  return 4;
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

std::vector<uint8_t> ConvertToDtype(const std::vector<float>& data, aclDataType dtype) {
  std::vector<uint8_t> result;
  if (dtype == ACL_FLOAT) {
    result.resize(data.size() * sizeof(float));
    std::memcpy(result.data(), data.data(), result.size());
  } else if (dtype == ACL_FLOAT16) {
    result.resize(data.size() * sizeof(aclFloat16));
    for (size_t i = 0; i < data.size(); ++i) {
      aclFloat16 val = aclFloatToFloat16(data[i]);
      std::memcpy(result.data() + i * sizeof(aclFloat16), &val, sizeof(aclFloat16));
    }
  } else if (dtype == ACL_BF16) {
    result.resize(data.size() * 2);
    for (size_t i = 0; i < data.size(); ++i) {
      uint32_t bits;
      std::memcpy(&bits, &data[i], sizeof(float));
      uint16_t bf16 = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
      std::memcpy(result.data() + i * 2, &bf16, 2);
    }
  } else if (dtype == ACL_INT32) {
    result.resize(data.size() * sizeof(int32_t));
    for (size_t i = 0; i < data.size(); ++i) {
      int32_t val = static_cast<int32_t>(data[i]);
      std::memcpy(result.data() + i * sizeof(int32_t), &val, sizeof(int32_t));
    }
  } else if (dtype == ACL_INT16) {
    result.resize(data.size() * sizeof(int16_t));
    for (size_t i = 0; i < data.size(); ++i) {
      int16_t val = static_cast<int16_t>(data[i]);
      std::memcpy(result.data() + i * sizeof(int16_t), &val, sizeof(int16_t));
    }
  } else if (dtype == ACL_INT64) {
    result.resize(data.size() * sizeof(int64_t));
    for (size_t i = 0; i < data.size(); ++i) {
      int64_t val = static_cast<int64_t>(data[i]);
      std::memcpy(result.data() + i * sizeof(int64_t), &val, sizeof(int64_t));
    }
  } else if (dtype == ACL_INT8) {
    result.resize(data.size() * sizeof(int8_t));
    for (size_t i = 0; i < data.size(); ++i) {
      int8_t val = static_cast<int8_t>(data[i]);
      std::memcpy(result.data() + i * sizeof(int8_t), &val, sizeof(int8_t));
    }
  } else if (dtype == ACL_UINT8) {
    result.resize(data.size() * sizeof(uint8_t));
    for (size_t i = 0; i < data.size(); ++i) {
      uint8_t val = static_cast<uint8_t>(data[i]);
      std::memcpy(result.data() + i * sizeof(uint8_t), &val, sizeof(uint8_t));
    }
  } else if (dtype == ACL_DOUBLE) {
    result.resize(data.size() * sizeof(double));
    for (size_t i = 0; i < data.size(); ++i) {
      double val = static_cast<double>(data[i]);
      std::memcpy(result.data() + i * sizeof(double), &val, sizeof(double));
    }
  } else if (dtype == ACL_BOOL) {
    result.resize(data.size() * sizeof(bool));
    for (size_t i = 0; i < data.size(); ++i) {
      bool val = static_cast<bool>(data[i]);
      std::memcpy(result.data() + i * sizeof(bool), &val, sizeof(bool));
    }
  }
  return result;
}

std::vector<float> ConvertFromDtype(const std::vector<uint8_t>& data, aclDataType dtype) {
  std::vector<float> result;
  size_t count = 0;
  if (dtype == ACL_FLOAT) {
    count = data.size() / sizeof(float);
    result.resize(count);
    std::memcpy(result.data(), data.data(), data.size());
  } else if (dtype == ACL_FLOAT16) {
    count = data.size() / sizeof(aclFloat16);
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      aclFloat16 val;
      std::memcpy(&val, data.data() + i * sizeof(aclFloat16), sizeof(aclFloat16));
      result[i] = aclFloat16ToFloat(val);
    }
  } else if (dtype == ACL_BF16) {
    count = data.size() / 2;
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      uint16_t bf16;
      std::memcpy(&bf16, data.data() + i * 2, 2);
      uint32_t bits = static_cast<uint32_t>(bf16) << 16;
      float f;
      std::memcpy(&f, &bits, sizeof(float));
      result[i] = f;
    }
  } else if (dtype == ACL_INT32) {
    count = data.size() / sizeof(int32_t);
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      int32_t val;
      std::memcpy(&val, data.data() + i * sizeof(int32_t), sizeof(int32_t));
      result[i] = static_cast<float>(val);
    }
  } else if (dtype == ACL_INT16) {
    count = data.size() / sizeof(int16_t);
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      int16_t val;
      std::memcpy(&val, data.data() + i * sizeof(int16_t), sizeof(int16_t));
      result[i] = static_cast<float>(val);
    }
  } else if (dtype == ACL_INT64) {
    count = data.size() / sizeof(int64_t);
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      int64_t val;
      std::memcpy(&val, data.data() + i * sizeof(int64_t), sizeof(int64_t));
      result[i] = static_cast<float>(val);
    }
  } else if (dtype == ACL_INT8) {
    count = data.size() / sizeof(int8_t);
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      int8_t val;
      std::memcpy(&val, data.data() + i * sizeof(int8_t), sizeof(int8_t));
      result[i] = static_cast<float>(val);
    }
  } else if (dtype == ACL_UINT8) {
    count = data.size() / sizeof(uint8_t);
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      uint8_t val;
      std::memcpy(&val, data.data() + i * sizeof(uint8_t), sizeof(uint8_t));
      result[i] = static_cast<float>(val);
    }
  } else if (dtype == ACL_BOOL) {
    count = data.size() / sizeof(bool);
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
      bool val;
      std::memcpy(&val, data.data() + i * sizeof(bool), sizeof(bool));
      result[i] = static_cast<float>(val);
    }
  }
  return result;
}

int CreateAclTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, aclDataType dtype, void** deviceAddr, aclTensor** tensor) {
  int64_t count = GetShapeSize(shape);
  std::vector<uint8_t> byteData;
  if (hostData.empty()) {
    byteData.resize(count * GetTypeSize(dtype), 0);
  } else {
    byteData = ConvertToDtype(hostData, dtype);
  }
  int64_t size = byteData.size();
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, byteData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed for dtype=%d, dim=%zu\n", dtype, shape.size()); return -1);
  return 0;
}

int CreateZeroDimTensor(aclDataType dtype, void** deviceAddr, aclTensor** tensor) {
  *deviceAddr = nullptr;
  *tensor = nullptr;
  auto ret = aclrtMalloc(deviceAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed in CreateZeroDimTensor. ERROR: %d\n", ret); return ret);
  *tensor = aclCreateTensor(nullptr, 0, dtype, nullptr, 0, aclFormat::ACL_FORMAT_ND, nullptr, 0, *deviceAddr);
  CHECK_RET(*tensor != nullptr,
            LOG_PRINT("aclCreateTensor zero-dim failed for dtype=%d\n", dtype); aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
  return 0;
}

bool CheckResult(void* deviceAddr, int64_t count, aclDataType dtype, const std::vector<float>& expected, float atol = 1e-3, float rtol = 1e-3) {
  int64_t byteSize = count * GetTypeSize(dtype);
  std::vector<uint8_t> resultBytes(byteSize);
  auto ret = aclrtMemcpy(resultBytes.data(), byteSize, deviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return false);
  auto result = ConvertFromDtype(resultBytes, dtype);
  if (result.size() != expected.size()) {
    LOG_PRINT("Size mismatch: got %zu, expected %zu\n", result.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < result.size(); ++i) {
    if (dtype == ACL_INT32 || dtype == ACL_BOOL) {
      if (result[i] != expected[i]) {
        LOG_PRINT("Mismatch at %zu: got %f, expected %f\n", i, result[i], expected[i]);
        return false;
      }
    } else {
      if (std::abs(result[i] - expected[i]) > atol + rtol * std::abs(expected[i])) {
        LOG_PRINT("Mismatch at %zu: got %f, expected %f\n", i, result[i], expected[i]);
        return false;
      }
    }
  }
  return true;
}

aclScalar* CreateScalarByType(aclDataType scalarDtype, float scalarValue) {
  if (scalarDtype == ACL_FLOAT) {
    return aclCreateScalar(&scalarValue, ACL_FLOAT);
  }
  if (scalarDtype == ACL_INT32) {
    int32_t v = static_cast<int32_t>(scalarValue);
    return aclCreateScalar(&v, ACL_INT32);
  }
  if (scalarDtype == ACL_DOUBLE) {
    double v = static_cast<double>(scalarValue);
    return aclCreateScalar(&v, ACL_DOUBLE);
  }
  if (scalarDtype == ACL_COMPLEX64) {
    float v = scalarValue;
    return aclCreateScalar(&v, ACL_COMPLEX64);
  }
  return nullptr;
}

void RunMulTest(const char* name, aclrtStream stream,
                const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<float>& selfData,
                const std::vector<int64_t>& otherShape, aclDataType otherDtype, const std::vector<float>& otherData,
                const std::vector<int64_t>& outShape, aclDataType outDtype, const std::vector<float>& expectedData) {
  void* selfAddr = nullptr;
  void* otherAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  if (CreateAclTensor(selfData, selfShape, selfDtype, &selfAddr, &self) != ACL_SUCCESS) goto CLEANUP;
  if (CreateAclTensor(otherData, otherShape, otherDtype, &otherAddr, &other) != ACL_SUCCESS) goto CLEANUP;
  if (CreateAclTensor({}, outShape, outDtype, &outAddr, &out) != ACL_SUCCESS) goto CLEANUP;

  {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", name, ret);
      goto CLEANUP;
    }
    void* wsAddr = nullptr;
    if (ws > 0) {
      auto mallocRet = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
      if (mallocRet != ACL_SUCCESS) goto CLEANUP;
    }
    ret = aclnnMul(wsAddr, ws, exec, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnMul failed. ERROR: %d\n", name, ret);
      if (wsAddr) aclrtFree(wsAddr);
      goto CLEANUP;
    }
    aclrtSynchronizeStream(stream);
    if (wsAddr) aclrtFree(wsAddr);
  }
  {
    int64_t count = GetShapeSize(outShape);
    CHECK_RET(CheckResult(outAddr, count, outDtype, expectedData),
              LOG_PRINT("%s failed\n", name); goto CLEANUP;);
    LOG_PRINT("%s passed\n", name);
  }
CLEANUP:
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfAddr) aclrtFree(selfAddr);
  if (otherAddr) aclrtFree(otherAddr);
  if (outAddr) aclrtFree(outAddr);
}

void RunMulsTest(const char* name, aclrtStream stream,
                 const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<float>& selfData,
                 aclDataType scalarDtype, float scalarValue,
                 const std::vector<int64_t>& outShape, aclDataType outDtype, const std::vector<float>& expectedData) {
  void* selfAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* other = nullptr;

  if (CreateAclTensor(selfData, selfShape, selfDtype, &selfAddr, &self) != ACL_SUCCESS) goto CLEANUP;
  if (CreateAclTensor({}, outShape, outDtype, &outAddr, &out) != ACL_SUCCESS) goto CLEANUP;

  other = CreateScalarByType(scalarDtype, scalarValue);

  {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(self, other, out, &ws, &exec);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", name, ret);
      goto CLEANUP;
    }
    void* wsAddr = nullptr;
    if (ws > 0) {
      auto mallocRet = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
      if (mallocRet != ACL_SUCCESS) goto CLEANUP;
    }
    ret = aclnnMuls(wsAddr, ws, exec, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnMuls failed. ERROR: %d\n", name, ret);
      if (wsAddr) aclrtFree(wsAddr);
      goto CLEANUP;
    }
    aclrtSynchronizeStream(stream);
    if (wsAddr) aclrtFree(wsAddr);
  }
  {
    int64_t count = GetShapeSize(outShape);
    CHECK_RET(CheckResult(outAddr, count, outDtype, expectedData),
              LOG_PRINT("%s failed\n", name); goto CLEANUP;);
    LOG_PRINT("%s passed\n", name);
  }
CLEANUP:
  if (self) aclDestroyTensor(self);
  if (out) aclDestroyTensor(out);
  if (other) aclDestroyScalar(other);
  if (selfAddr) aclrtFree(selfAddr);
  if (outAddr) aclrtFree(outAddr);
}

void RunInplaceMulTest(const char* name, aclrtStream stream,
                       const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<float>& selfData,
                       const std::vector<int64_t>& otherShape, aclDataType otherDtype, const std::vector<float>& otherData,
                       const std::vector<float>& expectedData) {
  void* selfAddr = nullptr;
  void* otherAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;

  if (CreateAclTensor(selfData, selfShape, selfDtype, &selfAddr, &self) != ACL_SUCCESS) goto CLEANUP;
  if (CreateAclTensor(otherData, otherShape, otherDtype, &otherAddr, &other) != ACL_SUCCESS) goto CLEANUP;

  {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceMulGetWorkspaceSize(self, other, &ws, &exec);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", name, ret);
      goto CLEANUP;
    }
    void* wsAddr = nullptr;
    if (ws > 0) {
      auto mallocRet = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
      if (mallocRet != ACL_SUCCESS) goto CLEANUP;
    }
    ret = aclnnInplaceMul(wsAddr, ws, exec, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnInplaceMul failed. ERROR: %d\n", name, ret);
      if (wsAddr) aclrtFree(wsAddr);
      goto CLEANUP;
    }
    aclrtSynchronizeStream(stream);
    if (wsAddr) aclrtFree(wsAddr);
  }
  {
    int64_t count = GetShapeSize(selfShape);
    CHECK_RET(CheckResult(selfAddr, count, selfDtype, expectedData),
              LOG_PRINT("%s failed\n", name); goto CLEANUP;);
    LOG_PRINT("%s passed\n", name);
  }
CLEANUP:
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (selfAddr) aclrtFree(selfAddr);
  if (otherAddr) aclrtFree(otherAddr);
}

void RunInplaceMulsTest(const char* name, aclrtStream stream,
                        const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<float>& selfData,
                        aclDataType scalarDtype, float scalarValue,
                        const std::vector<float>& expectedData) {
  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;

  if (CreateAclTensor(selfData, selfShape, selfDtype, &selfAddr, &self) != ACL_SUCCESS) goto CLEANUP;

  other = CreateScalarByType(scalarDtype, scalarValue);

  {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceMulsGetWorkspaceSize(self, other, &ws, &exec);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", name, ret);
      goto CLEANUP;
    }
    void* wsAddr = nullptr;
    if (ws > 0) {
      auto mallocRet = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
      if (mallocRet != ACL_SUCCESS) goto CLEANUP;
    }
    ret = aclnnInplaceMuls(wsAddr, ws, exec, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("%s: aclnnInplaceMuls failed. ERROR: %d\n", name, ret);
      if (wsAddr) aclrtFree(wsAddr);
      goto CLEANUP;
    }
    aclrtSynchronizeStream(stream);
    if (wsAddr) aclrtFree(wsAddr);
  }
  {
    int64_t count = GetShapeSize(selfShape);
    CHECK_RET(CheckResult(selfAddr, count, selfDtype, expectedData),
              LOG_PRINT("%s failed\n", name); goto CLEANUP;);
    LOG_PRINT("%s passed\n", name);
  }
CLEANUP:
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyScalar(other);
  if (selfAddr) aclrtFree(selfAddr);
}

void RunMulsWorkspaceStatusTest(const char* name,
                                const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<float>& selfData,
                                aclDataType scalarDtype, float scalarValue,
                                const std::vector<int64_t>& outShape, aclDataType outDtype) {
  void* selfAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* other = nullptr;

  if (CreateAclTensor(selfData, selfShape, selfDtype, &selfAddr, &self) != ACL_SUCCESS) goto CLEANUP;
  if (CreateAclTensor({}, outShape, outDtype, &outAddr, &out) != ACL_SUCCESS) goto CLEANUP;
  other = CreateScalarByType(scalarDtype, scalarValue);

  {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(self, other, out, &ws, &exec);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("%s failed: ret=%d\n", name, ret); goto CLEANUP;);
    LOG_PRINT("%s passed: ws=%lu\n", name, ws);
  }

CLEANUP:
  if (self) aclDestroyTensor(self);
  if (out) aclDestroyTensor(out);
  if (other) aclDestroyScalar(other);
  if (selfAddr) aclrtFree(selfAddr);
  if (outAddr) aclrtFree(outAddr);
}

void RunMulWorkspaceStatusTest(const char* name,
                               const std::vector<int64_t>& selfShape, aclDataType selfDtype, const std::vector<float>& selfData,
                               const std::vector<int64_t>& otherShape, aclDataType otherDtype, const std::vector<float>& otherData,
                               const std::vector<int64_t>& outShape, aclDataType outDtype) {
  void* selfAddr = nullptr;
  void* otherAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  if (CreateAclTensor(selfData, selfShape, selfDtype, &selfAddr, &self) != ACL_SUCCESS) goto CLEANUP;
  if (CreateAclTensor(otherData, otherShape, otherDtype, &otherAddr, &other) != ACL_SUCCESS) goto CLEANUP;
  if (CreateAclTensor({}, outShape, outDtype, &outAddr, &out) != ACL_SUCCESS) goto CLEANUP;

  {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("%s failed: ret=%d\n", name, ret); goto CLEANUP;);
    LOG_PRINT("%s passed: ws=%lu\n", name, ws);
  }

CLEANUP:
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfAddr) aclrtFree(selfAddr);
  if (otherAddr) aclrtFree(otherAddr);
  if (outAddr) aclrtFree(outAddr);
}

void TestIsFloatTypeCoverageCandidates() {
  RunMulsWorkspaceStatusTest("IsFloatType_SelfDouble", {2}, ACL_DOUBLE, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_DOUBLE);
  RunMulsWorkspaceStatusTest("IsFloatType_SelfFloat", {2}, ACL_FLOAT, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_FLOAT);
  RunMulsWorkspaceStatusTest("IsFloatType_SelfFloat16", {2}, ACL_FLOAT16, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_FLOAT16);
  RunMulsWorkspaceStatusTest("IsFloatType_SelfBf16", {2}, ACL_BF16, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_FLOAT);

  RunMulsWorkspaceStatusTest("IsFloatType_OtherFloat_True", {2}, ACL_INT32, {1.0f, -2.0f}, ACL_FLOAT, 2.0f, {2}, ACL_FLOAT);
  RunMulsWorkspaceStatusTest("IsFloatType_OtherNonFloat_False", {2}, ACL_INT32, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_INT32);
}

void TestInferTensorScalarDtypeBranchCoverageCandidates() {
  // RegBase path candidates: promoteType == FLOAT16/BF16 and keepB16 false branch.
  RunMulsWorkspaceStatusTest("Infer_Reg_KeepB16False_Fp16", {2}, ACL_FLOAT16, {1.0f, -2.0f}, ACL_FLOAT, 0.1f, {2}, ACL_FLOAT16);
  RunMulsWorkspaceStatusTest("Infer_Reg_KeepB16False_Bf16", {2}, ACL_BF16, {1.0f, -2.0f}, ACL_FLOAT, 0.1f, {2}, ACL_BF16);

  // RegBase path candidate: promoteType == COMPLEX32 then remap to COMPLEX64.
  RunMulsWorkspaceStatusTest("Infer_Reg_Complex32Remap", {2}, ACL_FLOAT16, {1.0f, -2.0f}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX64);

  // Non-RegBase path candidates (effective when IsRegBase() == false).
  RunMulsWorkspaceStatusTest("Infer_NonReg_ComplexBranch", {2}, ACL_COMPLEX64, {}, ACL_INT32, 2.0f, {2}, ACL_COMPLEX64);
  RunMulsWorkspaceStatusTest("Infer_NonReg_SelfFloatBranch", {2}, ACL_FLOAT, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_FLOAT);
  RunMulsWorkspaceStatusTest("Infer_NonReg_SelfBf16ToFloat", {2}, ACL_BF16, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_FLOAT);
  RunMulsWorkspaceStatusTest("Infer_NonReg_BoolDoubleRule", {2}, ACL_BOOL, {1.0f, 0.0f}, ACL_DOUBLE, 2.0f, {2}, ACL_FLOAT);
  RunMulsWorkspaceStatusTest("Infer_NonReg_OtherDoubleOutFloat", {2}, ACL_INT32, {1.0f, -2.0f}, ACL_DOUBLE, 2.0f, {2}, ACL_FLOAT);
  RunMulsWorkspaceStatusTest("Infer_NonReg_OtherFloatPromote", {2}, ACL_INT32, {1.0f, -2.0f}, ACL_FLOAT, 2.0f, {2}, ACL_FLOAT);
  RunMulsWorkspaceStatusTest("Infer_NonReg_FallbackSelf", {2}, ACL_INT32, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_INT32);
}

void RunExceptionTest(const char* name, aclnnStatus expectedStatus, aclnnStatus actualStatus) {
  if (actualStatus == expectedStatus) {
    LOG_PRINT("%s passed: %d\n", name, actualStatus);
  } else {
    LOG_PRINT("%s failed: expected %d, got %d\n", name, expectedStatus, actualStatus);
  }
}

void TestEmptyTensorFastPath() {
  {
    // 尝试使用 zero-dim(shape={}) 构造真正 empty tensor，提升 IsEmpty() 快速路径命中概率
    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    if (CreateZeroDimTensor(ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
        CreateZeroDimTensor(ACL_FLOAT, &otherAddr, &other) == ACL_SUCCESS &&
        CreateZeroDimTensor(ACL_FLOAT, &outAddr, &out) == ACL_SUCCESS) {
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyZeroDim_Mul passed\n");
      } else {
        LOG_PRINT("EmptyZeroDim_Mul failed: ret=%d ws=%lu\n", ret, ws);
      }
    }
    if (self) aclDestroyTensor(self);
    if (other) aclDestroyTensor(other);
    if (out) aclDestroyTensor(out);
    if (selfAddr) aclrtFree(selfAddr);
    if (otherAddr) aclrtFree(otherAddr);
    if (outAddr) aclrtFree(outAddr);
  }
  {
    void* selfAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    if (CreateZeroDimTensor(ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
        CreateZeroDimTensor(ACL_FLOAT, &outAddr, &out) == ACL_SUCCESS) {
      float scalarValue = 2.0f;
      aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnMulsGetWorkspaceSize(self, other, out, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyZeroDim_Muls passed\n");
      } else {
        LOG_PRINT("EmptyZeroDim_Muls failed: ret=%d ws=%lu\n", ret, ws);
      }
      if (other) aclDestroyScalar(other);
    }
    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
    if (selfAddr) aclrtFree(selfAddr);
    if (outAddr) aclrtFree(outAddr);
  }
  {
    void* selfAddr = nullptr;
    aclTensor* self = nullptr;
    if (CreateZeroDimTensor(ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS) {
      float scalarValue = 2.0f;
      aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnInplaceMulsGetWorkspaceSize(self, other, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyZeroDim_InplaceMuls passed\n");
      } else {
        LOG_PRINT("EmptyZeroDim_InplaceMuls failed: ret=%d ws=%lu\n", ret, ws);
      }
      if (other) aclDestroyScalar(other);
    }
    if (self) aclDestroyTensor(self);
    if (selfAddr) aclrtFree(selfAddr);
  }
  {
    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    if (CreateZeroDimTensor(ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
        CreateZeroDimTensor(ACL_FLOAT, &otherAddr, &other) == ACL_SUCCESS) {
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnInplaceMulGetWorkspaceSize(self, other, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyZeroDim_InplaceMul passed\n");
      } else {
        LOG_PRINT("EmptyZeroDim_InplaceMul failed: ret=%d ws=%lu\n", ret, ws);
      }
    }
    if (self) aclDestroyTensor(self);
    if (other) aclDestroyTensor(other);
    if (selfAddr) aclrtFree(selfAddr);
    if (otherAddr) aclrtFree(otherAddr);
  }
  {
    std::vector<int64_t> emptyShape = {0};
    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    if (CreateAclTensor({}, emptyShape, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
        CreateAclTensor({}, emptyShape, ACL_FLOAT, &otherAddr, &other) == ACL_SUCCESS &&
        CreateAclTensor({}, emptyShape, ACL_FLOAT, &outAddr, &out) == ACL_SUCCESS) {
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyMulFastPath passed\n");
      } else {
        LOG_PRINT("EmptyMulFastPath failed: ret=%d ws=%lu\n", ret, ws);
      }
    }
    if (self) aclDestroyTensor(self);
    if (other) aclDestroyTensor(other);
    if (out) aclDestroyTensor(out);
    if (selfAddr) aclrtFree(selfAddr);
    if (otherAddr) aclrtFree(otherAddr);
    if (outAddr) aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> emptyShape = {0};
    void* selfAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    if (CreateAclTensor({}, emptyShape, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
        CreateAclTensor({}, emptyShape, ACL_FLOAT, &outAddr, &out) == ACL_SUCCESS) {
      float scalarValue = 2.0f;
      aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnMulsGetWorkspaceSize(self, other, out, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyMulsFastPath passed\n");
      } else {
        LOG_PRINT("EmptyMulsFastPath failed: ret=%d ws=%lu\n", ret, ws);
      }
      if (other) aclDestroyScalar(other);
    }
    if (self) aclDestroyTensor(self);
    if (out) aclDestroyTensor(out);
    if (selfAddr) aclrtFree(selfAddr);
    if (outAddr) aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> emptyShape = {0};
    void* selfAddr = nullptr;
    aclTensor* self = nullptr;
    if (CreateAclTensor({}, emptyShape, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS) {
      float scalarValue = 2.0f;
      aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnInplaceMulsGetWorkspaceSize(self, other, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyInplaceMulsFastPath passed\n");
      } else {
        LOG_PRINT("EmptyInplaceMulsFastPath failed: ret=%d ws=%lu\n", ret, ws);
      }
      if (other) aclDestroyScalar(other);
    }
    if (self) aclDestroyTensor(self);
    if (selfAddr) aclrtFree(selfAddr);
  }
  {
    std::vector<int64_t> emptyShape = {0};
    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    if (CreateAclTensor({}, emptyShape, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
        CreateAclTensor({}, emptyShape, ACL_FLOAT, &otherAddr, &other) == ACL_SUCCESS) {
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnInplaceMulGetWorkspaceSize(self, other, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("EmptyInplaceMulFastPath passed\n");
      } else {
        LOG_PRINT("EmptyInplaceMulFastPath failed: ret=%d ws=%lu\n", ret, ws);
      }
    }
    if (self) aclDestroyTensor(self);
    if (other) aclDestroyTensor(other);
    if (selfAddr) aclrtFree(selfAddr);
    if (otherAddr) aclrtFree(otherAddr);
  }
}

void TestPriorityDualBranchCandidates(aclrtStream stream) {
  // 1) empty tensor true-side (explicit pair for muls / inplace muls)
  RunMulsWorkspaceStatusTest("Priority_EmptyTrue_Muls", {0}, ACL_FLOAT, {}, ACL_FLOAT, 2.0f, {0}, ACL_FLOAT);

  {
    std::vector<int64_t> emptyShape = {0};
    void* selfAddr = nullptr;
    aclTensor* self = nullptr;
    if (CreateAclTensor({}, emptyShape, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS) {
      float scalarValue = 2.0f;
      aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
      uint64_t ws = 123;
      aclOpExecutor* exec = nullptr;
      auto ret = aclnnInplaceMulsGetWorkspaceSize(self, other, &ws, &exec);
      if (ret == ACLNN_SUCCESS && ws == 0) {
        LOG_PRINT("Priority_EmptyTrue_InplaceMuls passed\n");
      } else {
        LOG_PRINT("Priority_EmptyTrue_InplaceMuls failed: ret=%d ws=%lu\n", ret, ws);
      }
      if (other) aclDestroyScalar(other);
    }
    if (self) aclDestroyTensor(self);
    if (selfAddr) aclrtFree(selfAddr);
  }

  // 2) canUseMuls opposite-side pairs for aclnnMulsGetWorkspaceSize
  // cond1 true: IsRegBase && (BF16/FP16) && scalarDefaultDtype == FLOAT
  RunMulsWorkspaceStatusTest("Priority_canUseMuls_cond1_true", {2}, ACL_BF16, {1.0f, -2.0f}, ACL_FLOAT, 2.0f, {2}, ACL_BF16);
  // cond1 false (opposite): keep scalar FLOAT but self not in {BF16, FP16}
  RunMulsWorkspaceStatusTest("Priority_canUseMuls_cond1_false", {2}, ACL_INT32, {1.0f, -2.0f}, ACL_FLOAT, 2.0f, {2}, ACL_FLOAT);

  // cond2 false (opposite): keep self BF16 but scalar not DOUBLE
  // cond2 is (!IsRegBase && self BF16 && other DOUBLE), here force other!=DOUBLE.
  RunMulsWorkspaceStatusTest("Priority_canUseMuls_cond2_false", {2}, ACL_BF16, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2}, ACL_BF16);

  // 3) canUseMuls opposite-side pairs for aclnnInplaceMulsGetWorkspaceSize
  RunInplaceMulsTest("Priority_Inplace_canUseMuls_cond1_true", stream, {2}, ACL_BF16, {1.0f, -2.0f}, ACL_FLOAT, 2.0f, {2.0f, -4.0f});
  RunInplaceMulsTest("Priority_Inplace_canUseMuls_cond1_false", stream, {2}, ACL_INT32, {1.0f, -2.0f}, ACL_FLOAT, 2.0f, {2.0f, -4.0f});
  RunInplaceMulsTest("Priority_Inplace_canUseMuls_cond2_false", stream, {2}, ACL_BF16, {1.0f, -2.0f}, ACL_INT32, 2.0f, {2.0f, -4.0f});
}

void TestExceptions(aclrtStream stream) {
  {
    std::vector<int64_t> shape = {2, 3};
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &otherAddr, &other);
    CreateAclTensor({0,0,0,0,0,0}, shape, ACL_FLOAT, &outAddr, &out);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(nullptr, other, out, &ws, &exec);
    RunExceptionTest("NullSelf", ACLNN_ERR_PARAM_NULLPTR, ret);
    aclDestroyTensor(other); aclDestroyTensor(out);
    aclrtFree(otherAddr); aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> shape = {2, 3};
    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &otherAddr, &other);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, nullptr, &ws, &exec);
    RunExceptionTest("NullOut", ACLNN_ERR_PARAM_NULLPTR, ret);
    aclDestroyTensor(self); aclDestroyTensor(other);
    aclrtFree(selfAddr); aclrtFree(otherAddr);
  }
  {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {2, 2};
    std::vector<int64_t> outShape = {2, 3};
    void* selfAddr = nullptr; void* otherAddr = nullptr; void* outAddr = nullptr;
    aclTensor* self = nullptr; aclTensor* other = nullptr; aclTensor* out = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, selfShape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({1,1,1,1}, otherShape, ACL_FLOAT, &otherAddr, &other);
    CreateAclTensor({0,0,0,0,0,0}, outShape, ACL_FLOAT, &outAddr, &out);
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    RunExceptionTest("ShapeMismatch", ACLNN_ERR_PARAM_INVALID, ret);
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclrtFree(selfAddr); aclrtFree(otherAddr); aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {4, 5};
    std::vector<int64_t> outShape = {2, 3};
    void* selfAddr = nullptr; void* otherAddr = nullptr; void* outAddr = nullptr;
    aclTensor* self = nullptr; aclTensor* other = nullptr; aclTensor* out = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, selfShape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, otherShape, ACL_FLOAT, &otherAddr, &other);
    CreateAclTensor({0,0,0,0,0,0}, outShape, ACL_FLOAT, &outAddr, &out);
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    RunExceptionTest("BroadcastMismatch_2x3_vs_4x5", ACLNN_ERR_PARAM_INVALID, ret);
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclrtFree(selfAddr); aclrtFree(otherAddr); aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int64_t> outShape = {2, 3};
    void* selfAddr = nullptr; void* otherAddr = nullptr; void* outAddr = nullptr;
    aclTensor* self = nullptr; aclTensor* other = nullptr; aclTensor* out = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, selfShape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({1}, otherShape, ACL_FLOAT, &otherAddr, &other);
    CreateAclTensor({0,0,0,0,0,0}, outShape, ACL_FLOAT, &outAddr, &out);
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    RunExceptionTest("OtherMaxDimExceeded", ACLNN_ERR_PARAM_INVALID, ret);
    if (self) aclDestroyTensor(self);
    if (other) aclDestroyTensor(other);
    if (out) aclDestroyTensor(out);
    if (selfAddr) aclrtFree(selfAddr);
    if (otherAddr) aclrtFree(otherAddr);
    if (outAddr) aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> tooManyDimsShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    CreateAclTensor({1}, tooManyDimsShape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({1}, tooManyDimsShape, ACL_FLOAT, &otherAddr, &other);
    CreateAclTensor({0}, tooManyDimsShape, ACL_FLOAT, &outAddr, &out);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    RunExceptionTest("MaxDimExceeded", ACLNN_ERR_PARAM_INVALID, ret);
    if (self) aclDestroyTensor(self);
    if (other) aclDestroyTensor(other);
    if (out) aclDestroyTensor(out);
    if (selfAddr) aclrtFree(selfAddr);
    if (otherAddr) aclrtFree(otherAddr);
    if (outAddr) aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {2, 3};
    std::vector<int64_t> outShape = {2, 2};
    void* selfAddr = nullptr; void* otherAddr = nullptr; void* outAddr = nullptr;
    aclTensor* self = nullptr; aclTensor* other = nullptr; aclTensor* out = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, selfShape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({1,1,1,1,1,1}, otherShape, ACL_FLOAT, &otherAddr, &other);
    CreateAclTensor({0,0,0,0}, outShape, ACL_FLOAT, &outAddr, &out);
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    RunExceptionTest("OutShapeMismatch", ACLNN_ERR_PARAM_INVALID, ret);
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclrtFree(selfAddr); aclrtFree(otherAddr); aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {3, 1};
    void* selfAddr = nullptr; void* otherAddr = nullptr;
    aclTensor* self = nullptr; aclTensor* other = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, selfShape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({1,1,1}, otherShape, ACL_FLOAT, &otherAddr, &other);
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceMulGetWorkspaceSize(self, other, &ws, &exec);
    RunExceptionTest("InplaceBroadcastMismatch", ACLNN_ERR_PARAM_INVALID, ret);
    aclDestroyTensor(self); aclDestroyTensor(other);
    aclrtFree(selfAddr); aclrtFree(otherAddr);
  }
  {
    std::vector<int64_t> shape = {2, 3};
    void* otherAddr = nullptr;
    aclTensor* other = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &otherAddr, &other);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceMulGetWorkspaceSize(nullptr, other, &ws, &exec);
    RunExceptionTest("InplaceMulNullSelf", ACLNN_ERR_PARAM_NULLPTR, ret);
    aclDestroyTensor(other);
    aclrtFree(otherAddr);
  }
  {
    std::vector<int64_t> shape = {2, 3};
    void* selfAddr = nullptr;
    aclTensor* self = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &selfAddr, &self);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceMulGetWorkspaceSize(self, nullptr, &ws, &exec);
    RunExceptionTest("InplaceMulNullOther", ACLNN_ERR_PARAM_NULLPTR, ret);
    aclDestroyTensor(self);
    aclrtFree(selfAddr);
  }
  {
    std::vector<int64_t> shape = {2, 3};
    void* outAddr = nullptr;
    aclTensor* out = nullptr;
    CreateAclTensor({0,0,0,0,0,0}, shape, ACL_FLOAT, &outAddr, &out);
    float scalarValue = 2.0f;
    aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(nullptr, other, out, &ws, &exec);
    RunExceptionTest("MulsNullSelf", ACLNN_ERR_PARAM_NULLPTR, ret);
    if (other) aclDestroyScalar(other);
    aclDestroyTensor(out);
    aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> shape = {2, 3};
    void* selfAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({0,0,0,0,0,0}, shape, ACL_FLOAT, &outAddr, &out);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(self, nullptr, out, &ws, &exec);
    RunExceptionTest("MulsNullOther", ACLNN_ERR_PARAM_NULLPTR, ret);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfAddr);
    aclrtFree(outAddr);
  }
  {
    std::vector<int64_t> shape = {2, 3};
    void* selfAddr = nullptr;
    aclTensor* self = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &selfAddr, &self);
    float scalarValue = 2.0f;
    aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(self, other, nullptr, &ws, &exec);
    RunExceptionTest("MulsNullOut", ACLNN_ERR_PARAM_NULLPTR, ret);
    if (other) aclDestroyScalar(other);
    aclDestroyTensor(self);
    aclrtFree(selfAddr);
  }
  {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> outShape = {2, 2};
    void* selfAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, selfShape, ACL_FLOAT, &selfAddr, &self);
    CreateAclTensor({0,0,0,0}, outShape, ACL_FLOAT, &outAddr, &out);
    float scalarValue = 2.0f;
    aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(self, other, out, &ws, &exec);
    RunExceptionTest("MulsOutShapeMismatch", ACLNN_ERR_PARAM_INVALID, ret);
    if (other) aclDestroyScalar(other);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfAddr);
    aclrtFree(outAddr);
  }
  {
    float scalarValue = 2.0f;
    aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceMulsGetWorkspaceSize(nullptr, other, &ws, &exec);
    RunExceptionTest("InplaceMulsNullSelf", ACLNN_ERR_PARAM_NULLPTR, ret);
    if (other) aclDestroyScalar(other);
  }
  {
    std::vector<int64_t> shape = {2, 3};
    void* selfAddr = nullptr;
    aclTensor* self = nullptr;
    CreateAclTensor({1,1,1,1,1,1}, shape, ACL_FLOAT, &selfAddr, &self);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnInplaceMulsGetWorkspaceSize(self, nullptr, &ws, &exec);
    RunExceptionTest("InplaceMulsNullOther", ACLNN_ERR_PARAM_NULLPTR, ret);
    aclDestroyTensor(self);
    aclrtFree(selfAddr);
  }
  {
    std::vector<int64_t> shape = {2};
    void* selfAddr = nullptr; void* otherAddr = nullptr; void* outAddr = nullptr;
    aclTensor* self = nullptr; aclTensor* other = nullptr; aclTensor* out = nullptr;
    CreateAclTensor({1.0f, 2.0f}, shape, ACL_BF16, &selfAddr, &self);
    CreateAclTensor({1.0f, 2.0f}, shape, ACL_BF16, &otherAddr, &other);
    CreateAclTensor({0.0f, 0.0f}, shape, ACL_BF16, &outAddr, &out);
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self, other, out, &ws, &exec);
    LOG_PRINT("DtypeNotSupported (BF16) ret: %d\n", ret);
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclrtFree(selfAddr); aclrtFree(otherAddr); aclrtFree(outAddr);
  }
}

void TestTilingAndNonContiguousCoverage(aclrtStream stream) {
  RunMulTest("NonContiguous_ShapeDimGt4_Int32", stream,
             {1, 1, 1, 1, 2}, ACL_INT32, {2, -3},
             {1, 1, 1, 1, 2}, ACL_INT32, {4, 5},
             {1, 1, 1, 1, 2}, ACL_INT32, {8, -15});

  RunMulTest("NonContiguous_ShapeDimGt4_MixFloat16Float", stream,
             {1, 1, 1, 1, 2}, ACL_FLOAT16, {2.0f, 4.0f},
             {1, 1, 1, 1, 2}, ACL_FLOAT, {0.5f, 0.25f},
             {1, 1, 1, 1, 2}, ACL_FLOAT, {1.0f, 1.0f});

  RunInplaceMulTest("InplaceMixDtype_RegFastPath", stream,
                    {2}, ACL_FLOAT16, {2.0f, 4.0f},
                    {2}, ACL_FLOAT, {0.5f, 0.25f},
                    {1.0f, 1.0f});

  RunMulTest("Tiling_Int8", stream,
             {2}, ACL_INT8, {2, -3},
             {2}, ACL_INT8, {4, 5},
             {2}, ACL_INT8, {8, -15});

  RunMulTest("Tiling_UInt8", stream,
             {2}, ACL_UINT8, {2, 3},
             {2}, ACL_UINT8, {4, 5},
             {2}, ACL_UINT8, {8, 15});

  RunMulTest("Tiling_Int16", stream,
             {2}, ACL_INT16, {2, -3},
             {2}, ACL_INT16, {4, 5},
             {2}, ACL_INT16, {8, -15});

  RunMulTest("Tiling_Int64", stream,
             {2}, ACL_INT64, {2, -3},
             {2}, ACL_INT64, {4, 5},
             {2}, ACL_INT64, {8, -15});
}

void TestCastPromotionCoverage(aclrtStream stream) {
  RunMulTest("CastPromotion_Int8xInt16", stream,
             {2}, ACL_INT8, {2, -3},
             {2}, ACL_INT16, {4, 5},
             {2}, ACL_INT16, {8, -15});

  RunMulTest("CastPromotion_Int16xInt8", stream,
             {2}, ACL_INT16, {2, -3},
             {2}, ACL_INT8, {4, 5},
             {2}, ACL_INT16, {8, -15});
}

void TestComplex32TilingCoverage() {
#ifdef ACL_COMPLEX32
  RunMulWorkspaceStatusTest("Tiling_Complex32_Workspace", {2}, ACL_COMPLEX32, {},
                            {2}, ACL_COMPLEX32, {}, {2}, ACL_COMPLEX32);
#else
  LOG_PRINT("Tiling_Complex32_Workspace skipped: ACL_COMPLEX32 not available\n");
#endif
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  RunMulTest("SameShapeBaseline", stream, {2, 3}, ACL_FLOAT, {-2, -1, 0, 1, 2, 3}, {2, 3}, ACL_FLOAT, {3, -4, 5, 0, -1, 2}, {2, 3}, ACL_FLOAT, {-6, 4, 0, 0, -2, 6});
  RunMulTest("ZeroValueSensitive", stream, {2, 2}, ACL_FLOAT, {0.0, -0.0, 1.0, -1.0}, {2, 2}, ACL_FLOAT, {-1.0, 1.0, 0.0, -0.0}, {2, 2}, ACL_FLOAT, {-0.0, -0.0, 0.0, 0.0});
  RunMulTest("Broadcast", stream, {2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}, {1, 3}, ACL_FLOAT, {1, 0, -1}, {2, 3}, ACL_FLOAT, {1, 0, -3, 4, 0, -6});
  RunMulsTest("ScalarVariant", stream, {2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}, ACL_FLOAT, 2.0f, {2, 3}, ACL_FLOAT, {2, 4, 6, 8, 10, 12});
  RunMulsTest("ScalarVariantInt32", stream, {2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}, ACL_INT32, 2.0f, {2, 3}, ACL_FLOAT, {2, 4, 6, 8, 10, 12});
  RunMulsTest("ScalarVariantDouble", stream, {2, 3}, ACL_BF16, {1, 2, 3, 4, 5, 6}, ACL_DOUBLE, 2.0f, {2, 3}, ACL_BF16, {2, 4, 6, 8, 10, 12});
  RunMulsWorkspaceStatusTest("Branch_ComplexLower_True", {2}, ACL_INT32, {1, 2}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX64);
  RunMulsTest("Branch_HigherFloating_True", stream, {2}, ACL_FLOAT, {1.5, 2.5}, ACL_INT32, 2.0f, {2}, ACL_FLOAT, {3.0, 5.0});
  RunMulsTest("Branch_HigherNotUndefined_True", stream, {2}, ACL_INT32, {3, -2}, ACL_INT32, 2.0f, {2}, ACL_INT32, {6, -4});
  RunMulsWorkspaceStatusTest("Branch_HigherNotUndefined_False_Guard", {2}, ACL_DT_UNDEFINED, {1, 2}, ACL_INT32, 2.0f, {2}, ACL_INT32);
  RunInplaceMulTest("InplaceTensor", stream, {2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}, {2, 3}, ACL_FLOAT, {1, 1, 1, 2, 2, 2}, {1, 2, 3, 8, 10, 12});
  RunInplaceMulsTest("InplaceScalar", stream, {2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}, ACL_FLOAT, 3.0f, {3, 6, 9, 12, 15, 18});
  RunInplaceMulsTest("InplaceScalarInt32", stream, {2, 3}, ACL_FLOAT, {1, 2, 3, 4, 5, 6}, ACL_INT32, 3.0f, {3, 6, 9, 12, 15, 18});
  RunInplaceMulsTest("InplaceScalarDouble", stream, {2, 3}, ACL_BF16, {1, 2, 3, 4, 5, 6}, ACL_DOUBLE, 3.0f, {3, 6, 9, 12, 15, 18});

  RunMulTest("DtypeFloat", stream, {2}, ACL_FLOAT, {1.5, 2.5}, {2}, ACL_FLOAT, {2.0, 2.0}, {2}, ACL_FLOAT, {3.0, 5.0});
  RunMulTest("DtypeInt32", stream, {2}, ACL_INT32, {3, -2}, {2}, ACL_INT32, {2, 3}, {2}, ACL_INT32, {6, -6});
  RunMulTest("DtypeFloat16", stream, {2}, ACL_FLOAT16, {1.5, 2.5}, {2}, ACL_FLOAT16, {2.0, 2.0}, {2}, ACL_FLOAT16, {3.0, 5.0});
  RunMulTest("DtypeBf16", stream, {2}, ACL_BF16, {1.5, 2.5}, {2}, ACL_BF16, {2.0, 2.0}, {2}, ACL_BF16, {3.0, 5.0});
  RunMulTest("DtypeBool", stream, {2}, ACL_BOOL, {1, 0}, {2}, ACL_BOOL, {1, 1}, {2}, ACL_BOOL, {1, 0});
  RunMulTest("MixedType1", stream, {2}, ACL_FLOAT16, {1.5, 2.5}, {2}, ACL_FLOAT, {2.0, 2.0}, {2}, ACL_FLOAT, {3.0, 5.0});
  RunMulTest("MixedType2", stream, {2}, ACL_BF16, {1.5, 2.5}, {2}, ACL_FLOAT, {2.0, 2.0}, {2}, ACL_FLOAT, {3.0, 5.0});
  RunMulTest("MixedType3", stream, {2}, ACL_FLOAT, {2.0, 4.0}, {2}, ACL_FLOAT16, {0.5, 0.25}, {2}, ACL_FLOAT, {1.0, 1.0});
  RunMulTest("MixedType4", stream, {2}, ACL_FLOAT, {2.0, 4.0}, {2}, ACL_BF16, {0.5, 0.25}, {2}, ACL_FLOAT, {1.0, 1.0});
  RunMulsTest("ScalarPromotion", stream, {2}, ACL_INT32, {3, -2}, ACL_FLOAT, 2.5f, {2}, ACL_FLOAT, {7.5, -5.0});

  RunMulsWorkspaceStatusTest("InnerTypeCase_BF16", {2}, ACL_BF16, {1.0f, -2.0f}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX64);
  RunMulsWorkspaceStatusTest("InnerTypeCase_FLOAT16", {2}, ACL_FLOAT16, {1.0f, -2.0f}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX64);
  RunMulsWorkspaceStatusTest("InnerTypeCase_FLOAT", {2}, ACL_FLOAT, {1.0f, -2.0f}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX64);
  RunMulsWorkspaceStatusTest("InnerTypeCase_DOUBLE", {2}, ACL_DOUBLE, {1.0f, -2.0f}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX128);

#ifdef ACL_COMPLEX32
  RunMulsWorkspaceStatusTest("InnerTypeCase_COMPLEX32_GuardPath", {2}, ACL_COMPLEX32, {}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX32);
#endif
  RunMulsWorkspaceStatusTest("InnerTypeCase_COMPLEX64_GuardPath", {2}, ACL_COMPLEX64, {}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX64);
  RunMulsWorkspaceStatusTest("InnerTypeCase_COMPLEX128_GuardPath", {2}, ACL_COMPLEX128, {}, ACL_COMPLEX64, 2.0f, {2}, ACL_COMPLEX128);

  TestIsFloatTypeCoverageCandidates();
  TestInferTensorScalarDtypeBranchCoverageCandidates();
  TestPriorityDualBranchCandidates(stream);

  TestEmptyTensorFastPath();

  TestExceptions(stream);
  TestTilingAndNonContiguousCoverage(stream);
  TestCastPromotionCoverage(stream);
  TestComplex32TilingCoverage();

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
