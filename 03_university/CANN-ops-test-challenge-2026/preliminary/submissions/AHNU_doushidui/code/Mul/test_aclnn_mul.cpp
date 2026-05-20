#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <complex>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
    fflush(stdout);                 \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    if (i == 0) return 0;
    shapeSize *= i;
  }
  return shapeSize;
}

// Support broadcasting and custom strides for expected calculation
template <typename T>
void ComputeExpected(const std::vector<T>& selfData, const std::vector<int64_t>& selfShape, const std::vector<int64_t>& selfStrides,
                     const std::vector<T>& otherData, const std::vector<int64_t>& otherShape, const std::vector<int64_t>& otherStrides,
                     std::vector<T>& expectedData, const std::vector<int64_t>& outShape) {
  int64_t outSize = GetShapeSize(outShape);
  expectedData.resize(outSize);
  if (outSize == 0) return;

  for (int64_t i = 0; i < outSize; ++i) {
    int64_t selfIdx = 0;
    int64_t otherIdx = 0;
    int64_t temp = i;
    
    for (int j = outShape.size() - 1; j >= 0; --j) {
      int64_t dimIdx = temp % outShape[j];
      temp /= outShape[j];

      // Broadcast logic: if dim is 1, index is 0. Else it's dimIdx % dim.
      int64_t selfDim = j >= (int)(outShape.size() - selfShape.size()) ? selfShape[j - (outShape.size() - selfShape.size())] : 1;
      int64_t otherDim = j >= (int)(outShape.size() - otherShape.size()) ? otherShape[j - (outShape.size() - otherShape.size())] : 1;

      if (selfDim > 1) {
          int64_t sDimIdx = dimIdx % selfDim;
          selfIdx += sDimIdx * selfStrides[j - (outShape.size() - selfShape.size())];
      }
      if (otherDim > 1) {
          int64_t oDimIdx = dimIdx % otherDim;
          otherIdx += oDimIdx * otherStrides[j - (outShape.size() - otherShape.size())];
      }
    }
    expectedData[i] = selfData[selfIdx] * otherData[otherIdx];
  }
}

template <typename T>
bool CompareResult(const std::vector<T>& actualData, const std::vector<T>& expectedData) {
  if (actualData.size() != expectedData.size()) return false;
  for (size_t i = 0; i < actualData.size(); ++i) {
    double diff = std::abs(static_cast<double>(actualData[i]) - static_cast<double>(expectedData[i]));
    double threshold = 1e-4 + 1e-4 * std::abs(static_cast<double>(expectedData[i]));
    if (diff > threshold) {
      LOG_PRINT("Mismatch at index %zu: expected %f but got %f\n", i, static_cast<double>(expectedData[i]), static_cast<double>(actualData[i]));
      return false;
    }
  }
  return true;
}

// Specialized for complex
template <typename T>
bool CompareResult(const std::vector<std::complex<T>>& actualData, const std::vector<std::complex<T>>& expectedData) {
  if (actualData.size() != expectedData.size()) return false;
  for (size_t i = 0; i < actualData.size(); ++i) {
    double diff_real = std::abs(static_cast<double>(actualData[i].real()) - static_cast<double>(expectedData[i].real()));
    double diff_imag = std::abs(static_cast<double>(actualData[i].imag()) - static_cast<double>(expectedData[i].imag()));
    if (diff_real > 1e-4 || diff_imag > 1e-4) {
      LOG_PRINT("Mismatch at index %zu: expected (%f, %f) but got (%f, %f)\n", i, 
                (double)expectedData[i].real(), (double)expectedData[i].imag(),
                (double)actualData[i].real(), (double)actualData[i].imag());
      return false;
    }
  }
  return true;
}

template <typename T>
bool VerifyOutResult(const std::vector<int64_t>& shape, void** deviceAddr, const std::vector<T>& expectedData) {
  auto size = GetShapeSize(shape);
  if (size == 0) return true;
  std::vector<T> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T),
                         *deviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return false);
  return CompareResult(resultData, expectedData);
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape);
  // Calculate real allocation size based on strides and shape
  int64_t max_idx = 0;
  if (size > 0) {
      for (size_t i = 0; i < shape.size(); ++i) {
          max_idx += (shape[i] - 1) * strides[i];
      }
      auto allocSize = (max_idx + 1) * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      
      // If hostData is provided, copy it. Note: hostData should be sized to cover the indexed elements.
      // For simplicity, we assume hostData is in the same layout as the tensor would be if it were contiguous.
      // To handle non-contiguous, we'd need a more complex copy.
      // But for our tests, we can just copy the whole block if hostData is large enough, or copy element by element.
      if (hostData.size() >= (size_t)size) {
          std::vector<T> deviceHostBuffer(max_idx + 1, 0);
          // Map contiguous hostData to non-contiguous deviceHostBuffer
          for (int64_t i = 0; i < size; ++i) {
              int64_t logical_idx = i;
              int64_t physical_idx = 0;
              int64_t temp = logical_idx;
              std::vector<int64_t> contig_strides(shape.size(), 1);
              for (int j = (int)shape.size() - 2; j >= 0; --j) contig_strides[j] = contig_strides[j+1] * shape[j+1];
              
              for (size_t j = 0; j < shape.size(); ++j) {
                  physical_idx += (temp / contig_strides[j]) * strides[j];
                  temp %= contig_strides[j];
              }
              deviceHostBuffer[physical_idx] = hostData[logical_idx];
          }
          ret = aclrtMemcpy(*deviceAddr, allocSize, deviceHostBuffer.data(), allocSize, ACL_MEMCPY_HOST_TO_DEVICE);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      }
  } else {
    *deviceAddr = nullptr;
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
void GenData(std::vector<T>& data, size_t size, T start = 1) {
  data.resize(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = start + (T)(static_cast<double>(i % 10));
  }
}

std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
double GetRealPart(T val) {
    return static_cast<double>(val);
}

template <typename T>
double GetRealPart(std::complex<T> val) {
    return static_cast<double>(val.real());
}

template <typename T>
int RunTestAll(int32_t deviceId, aclrtStream stream, aclDataType dataType, 
               const std::vector<int64_t>& selfShape, const std::vector<int64_t>& selfStrides,
               const std::vector<int64_t>& otherShape, const std::vector<int64_t>& otherStrides,
               const std::vector<int64_t>& outShape, const std::vector<int64_t>& outStrides,
               bool isInplace = false, bool isMuls = false) {
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclScalar* otherScalar = nullptr;

  std::vector<T> selfHostData, otherHostData, outHostData, expectedData;
  GenData(selfHostData, GetShapeSize(selfShape), (T)1);
  if (!isMuls) GenData(otherHostData, GetShapeSize(otherShape), (T)2);
  else otherHostData = {(T)3};

  if (isInplace) {
      ComputeExpected(selfHostData, selfShape, GetContiguousStrides(selfShape), 
                      otherHostData, otherShape, isMuls ? std::vector<int64_t>{1} : GetContiguousStrides(otherShape), 
                      expectedData, selfShape);
  } else {
      ComputeExpected(selfHostData, selfShape, GetContiguousStrides(selfShape), 
                      otherHostData, otherShape, isMuls ? std::vector<int64_t>{1} : GetContiguousStrides(otherShape), 
                      expectedData, outShape);
  }

  int ret = CreateAclTensor(selfHostData, selfShape, selfStrides, &selfDeviceAddr, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  if (!isMuls) {
      ret = CreateAclTensor(otherHostData, otherShape, otherStrides, &otherDeviceAddr, dataType, &other);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  } else {
      T val = otherHostData[0];
      if (dataType == ACL_DOUBLE) { double v = GetRealPart(val); otherScalar = aclCreateScalar(&v, dataType); }
      else if (dataType == ACL_INT32) { int32_t v = static_cast<int32_t>(GetRealPart(val)); otherScalar = aclCreateScalar(&v, dataType); }
      else { float v = static_cast<float>(GetRealPart(val)); otherScalar = aclCreateScalar(&v, dataType); }
  }

  if (!isInplace) {
      outHostData.resize(GetShapeSize(outShape), 0);
      ret = CreateAclTensor(outHostData, outShape, outStrides, &outDeviceAddr, dataType, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  if (isMuls) {
      if (isInplace) ret = aclnnInplaceMulsGetWorkspaceSize(self, otherScalar, &workspaceSize, &executor);
      else ret = aclnnMulsGetWorkspaceSize(self, otherScalar, out, &workspaceSize, &executor);
  } else {
      if (isInplace) ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
      else ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  }

  if (ret != ACL_SUCCESS) {
      LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret);
      goto cleanup;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }

  if (isMuls) {
      if (isInplace) ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
      else ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  } else {
      if (isInplace) ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
      else ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }
  
  if (ret != ACL_SUCCESS) {
      LOG_PRINT("Execution failed. ERROR: %d\n", ret);
      goto cleanup;
  }

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
      bool check = false;
      if (isInplace) check = VerifyOutResult(selfShape, &selfDeviceAddr, expectedData);
      else check = VerifyOutResult(outShape, &outDeviceAddr, expectedData);
      
      if (!check) {
          LOG_PRINT("Verify failed!\n");
          ret = -1;
      } else {
          LOG_PRINT("Verify passed!\n");
          ret = 0;
      }
  }

  if (workspaceSize > 0) aclrtFree(workspaceAddr);

cleanup:
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (otherScalar) aclDestroyScalar(otherScalar);
  if (selfDeviceAddr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr) aclrtFree(outDeviceAddr);
  return ret;
}

int RunComplexTests(int32_t deviceId, aclrtStream stream) {
    LOG_PRINT("--- Complex Tests ---\n");
    // Complex64
    RunTestAll<std::complex<float>>(deviceId, stream, ACL_COMPLEX64, {2, 2}, {2, 1}, {2, 2}, {2, 1}, {2, 2}, {2, 1});
    // Complex128 (Should fail in Tiling as per our analysis, or might work if platform supports it)
    RunTestAll<std::complex<double>>(deviceId, stream, ACL_COMPLEX128, {2, 2}, {2, 1}, {2, 2}, {2, 1}, {2, 2}, {2, 1});
    return 0;
}

int RunNonContiguousTests(int32_t deviceId, aclrtStream stream) {
    LOG_PRINT("--- Non-Contiguous Tests ---\n");
    // Self non-contiguous: shape {2, 2}, strides {1, 2} (Transposed logically)
    RunTestAll<float>(deviceId, stream, ACL_FLOAT, {2, 2}, {1, 2}, {2, 2}, {2, 1}, {2, 2}, {2, 1});
    // Other non-contiguous
    RunTestAll<float>(deviceId, stream, ACL_FLOAT, {2, 2}, {2, 1}, {2, 2}, {1, 2}, {2, 2}, {2, 1});
    // Out non-contiguous
    RunTestAll<float>(deviceId, stream, ACL_FLOAT, {2, 2}, {2, 1}, {2, 2}, {2, 1}, {2, 2}, {1, 2});
    return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  if (ret != 0) return ret;

  // Existing tests refactored
  LOG_PRINT("--- Basic Tests ---\n");
  RunTestAll<float>(deviceId, stream, ACL_FLOAT, {4, 2}, {2, 1}, {4, 2}, {2, 1}, {4, 2}, {2, 1});
  RunTestAll<int32_t>(deviceId, stream, ACL_INT32, {4, 2}, {2, 1}, {4, 2}, {2, 1}, {4, 2}, {2, 1});
  
  LOG_PRINT("--- Inplace Tests ---\n");
  RunTestAll<float>(deviceId, stream, ACL_FLOAT, {4, 2}, {2, 1}, {4, 2}, {2, 1}, {}, {}, true);
  
  LOG_PRINT("--- Muls Tests ---\n");
  RunTestAll<float>(deviceId, stream, ACL_FLOAT, {4, 2}, {2, 1}, {1}, {1}, {4, 2}, {2, 1}, false, true);
  RunTestAll<float>(deviceId, stream, ACL_FLOAT, {4, 2}, {2, 1}, {1}, {1}, {}, {}, true, true);

  LOG_PRINT("--- Broadcast Tests ---\n");
  RunTestAll<float>(deviceId, stream, ACL_FLOAT, {4, 2}, {2, 1}, {2}, {1}, {4, 2}, {2, 1});
  RunTestAll<float>(deviceId, stream, ACL_FLOAT, {4, 1}, {1, 1}, {4, 2}, {2, 1}, {4, 2}, {2, 1});
  RunTestAll<float>(deviceId, stream, ACL_FLOAT, {2, 1, 3}, {3, 3, 1}, {4, 3}, {3, 1}, {2, 4, 3}, {12, 3, 1});

  RunNonContiguousTests(deviceId, stream);
  RunComplexTests(deviceId, stream);

  // Error cases
  LOG_PRINT("--- Extra Error Tests ---\n");
  uint64_t ws;
  aclOpExecutor* exe;
  // Unsupported Dtype
  aclTensor *t1, *t2;
  void *d1, *d2;
  std::vector<float> h = {1.0f};
  CreateAclTensor(h, {1}, {1}, &d1, ACL_FLOAT, &t1);
  // Using a dtype that is likely not supported by Mul (e.g. some obscure one)
  // Let's try ACL_INT4 if it exists, or just something invalid.
  // Actually, we can just pass nullptr to hit the OP_CHECK_NULL.
  aclnnMulGetWorkspaceSize(nullptr, t1, t1, &ws, &exe);
  
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
