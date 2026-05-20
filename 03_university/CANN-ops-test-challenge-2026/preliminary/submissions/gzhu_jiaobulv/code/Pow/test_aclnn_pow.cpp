/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstdio>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

namespace l0op {
const aclTensor *Pow(const aclTensor *self, const aclTensor *exponent, aclOpExecutor *executor);
}

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
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat format, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
  int ret = ACL_SUCCESS;
  *deviceAddr = nullptr;
  if (size > 0) {
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  return CreateAclTensor(hostData, shape, deviceAddr, dataType, ACL_FORMAT_ND, tensor);
}

template <typename T>
std::vector<T> CastData(const std::vector<double>& inData) {
  std::vector<T> outData;
  outData.reserve(inData.size());
  for (double value : inData) {
    outData.push_back(static_cast<T>(value));
  }
  return outData;
}

int CreateAclTensorFromDoubleData(const std::vector<double>& hostData, const std::vector<int64_t>& shape,
                                  aclDataType dataType, aclFormat format, void** deviceAddr, aclTensor** tensor) {
  switch (dataType) {
    case ACL_FLOAT16: {
      auto typed = CastData<uint16_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_BF16: {
      auto typed = CastData<uint16_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_FLOAT: {
      auto typed = CastData<float>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_DOUBLE: {
      auto typed = CastData<double>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_INT64: {
      auto typed = CastData<int64_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_INT32: {
      auto typed = CastData<int32_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_INT16: {
      auto typed = CastData<int16_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_INT8: {
      auto typed = CastData<int8_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_UINT8: {
      auto typed = CastData<uint8_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_UINT16: {
      auto typed = CastData<uint16_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_UINT32: {
      auto typed = CastData<uint32_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_UINT64: {
      auto typed = CastData<uint64_t>(hostData);
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_BOOL: {
      std::vector<uint8_t> typed;
      typed.reserve(hostData.size());
      for (double value : hostData) {
        typed.push_back((value != 0.0) ? 1U : 0U);
      }
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_COMPLEX64: {
      std::vector<std::complex<float>> typed;
      typed.reserve(hostData.size());
      for (double value : hostData) {
        typed.emplace_back(static_cast<float>(value), 0.0f);
      }
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    case ACL_COMPLEX128: {
      std::vector<std::complex<double>> typed;
      typed.reserve(hostData.size());
      for (double value : hostData) {
        typed.emplace_back(value, 0.0);
      }
      return CreateAclTensor(typed, shape, deviceAddr, dataType, format, tensor);
    }
    default:
      LOG_PRINT("unsupported CreateAclTensorFromDoubleData dtype: %d\n", static_cast<int>(dataType));
      return 1;
  }
}

int CreateAclTensorFromDoubleData(const std::vector<double>& hostData, const std::vector<int64_t>& shape,
                                  aclDataType dataType, void** deviceAddr, aclTensor** tensor) {
  return CreateAclTensorFromDoubleData(hostData, shape, dataType, ACL_FORMAT_ND, deviceAddr, tensor);
}

int CreateZeroAclTensor(const std::vector<int64_t>& shape, aclDataType dataType, aclFormat format, void** deviceAddr, aclTensor** tensor) {
  auto elemNum = static_cast<size_t>(GetShapeSize(shape));
  std::vector<double> zeros(elemNum, 0.0);
  return CreateAclTensorFromDoubleData(zeros, shape, dataType, format, deviceAddr, tensor);
}

int CreateZeroAclTensor(const std::vector<int64_t>& shape, aclDataType dataType, void** deviceAddr, aclTensor** tensor) {
  return CreateZeroAclTensor(shape, dataType, ACL_FORMAT_ND, deviceAddr, tensor);
}

int CreateAclScalarFromDouble(double value, aclDataType dataType, aclScalar** scalar) {
  *scalar = nullptr;
  switch (dataType) {
    case ACL_FLOAT16: {
      uint16_t v = static_cast<uint16_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_BF16: {
      uint16_t v = static_cast<uint16_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_FLOAT: {
      float v = static_cast<float>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_DOUBLE: {
      double v = value;
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_INT64: {
      int64_t v = static_cast<int64_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_INT32: {
      int32_t v = static_cast<int32_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_INT16: {
      int16_t v = static_cast<int16_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_INT8: {
      int8_t v = static_cast<int8_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_UINT8: {
      uint8_t v = static_cast<uint8_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_UINT16: {
      uint16_t v = static_cast<uint16_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_UINT32: {
      uint32_t v = static_cast<uint32_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_UINT64: {
      uint64_t v = static_cast<uint64_t>(value);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    case ACL_BOOL: {
      bool v = (value != 0.0);
      *scalar = aclCreateScalar(&v, dataType);
      break;
    }
    default:
      LOG_PRINT("unsupported CreateAclScalarFromDouble dtype: %d\n", static_cast<int>(dataType));
      return 1;
  }
  CHECK_RET(*scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return 1);
  return ACL_SUCCESS;
}

int CreateAclComplexScalar(double realValue, double imagValue, aclDataType dataType, aclScalar** scalar) {
  *scalar = nullptr;
  if (dataType == ACL_COMPLEX64) {
    std::complex<float> v(static_cast<float>(realValue), static_cast<float>(imagValue));
    *scalar = aclCreateScalar(&v, dataType);
  } else if (dataType == ACL_COMPLEX128) {
    std::complex<double> v(realValue, imagValue);
    *scalar = aclCreateScalar(&v, dataType);
  } else {
    LOG_PRINT("unsupported CreateAclComplexScalar dtype: %d\n", static_cast<int>(dataType));
    return 1;
  }
  CHECK_RET(*scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return 1);
  return ACL_SUCCESS;
}

int CopyTensorToHostDouble(const std::vector<int64_t>& shape, aclDataType dataType, void* deviceAddr,
                           std::vector<double>& hostData) {
  auto elemNum = static_cast<size_t>(GetShapeSize(shape));
  hostData.assign(elemNum, 0.0);
  if (elemNum == 0) {
    return ACL_SUCCESS;
  }

  switch (dataType) {
    case ACL_FLOAT: {
      std::vector<float> typed(elemNum, 0.0f);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(float), deviceAddr, elemNum * sizeof(float),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      for (size_t i = 0; i < elemNum; ++i) {
        hostData[i] = static_cast<double>(typed[i]);
      }
      return ACL_SUCCESS;
    }
    case ACL_DOUBLE: {
      std::vector<double> typed(elemNum, 0.0);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(double), deviceAddr, elemNum * sizeof(double),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      hostData = typed;
      return ACL_SUCCESS;
    }
    case ACL_INT64: {
      std::vector<int64_t> typed(elemNum, 0);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(int64_t), deviceAddr, elemNum * sizeof(int64_t),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      for (size_t i = 0; i < elemNum; ++i) {
        hostData[i] = static_cast<double>(typed[i]);
      }
      return ACL_SUCCESS;
    }
    case ACL_INT32: {
      std::vector<int32_t> typed(elemNum, 0);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(int32_t), deviceAddr, elemNum * sizeof(int32_t),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      for (size_t i = 0; i < elemNum; ++i) {
        hostData[i] = static_cast<double>(typed[i]);
      }
      return ACL_SUCCESS;
    }
    case ACL_INT16: {
      std::vector<int16_t> typed(elemNum, 0);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(int16_t), deviceAddr, elemNum * sizeof(int16_t),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      for (size_t i = 0; i < elemNum; ++i) {
        hostData[i] = static_cast<double>(typed[i]);
      }
      return ACL_SUCCESS;
    }
    case ACL_INT8: {
      std::vector<int8_t> typed(elemNum, 0);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(int8_t), deviceAddr, elemNum * sizeof(int8_t),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      for (size_t i = 0; i < elemNum; ++i) {
        hostData[i] = static_cast<double>(typed[i]);
      }
      return ACL_SUCCESS;
    }
    case ACL_UINT8: {
      std::vector<uint8_t> typed(elemNum, 0);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(uint8_t), deviceAddr, elemNum * sizeof(uint8_t),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      for (size_t i = 0; i < elemNum; ++i) {
        hostData[i] = static_cast<double>(typed[i]);
      }
      return ACL_SUCCESS;
    }
    case ACL_BOOL: {
      std::vector<uint8_t> typed(elemNum, 0);
      auto ret = aclrtMemcpy(typed.data(), elemNum * sizeof(uint8_t), deviceAddr, elemNum * sizeof(uint8_t),
                             ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      for (size_t i = 0; i < elemNum; ++i) {
        hostData[i] = static_cast<double>(typed[i] ? 1 : 0);
      }
      return ACL_SUCCESS;
    }
    default:
      LOG_PRINT("unsupported CopyTensorToHostDouble dtype: %d\n", static_cast<int>(dataType));
      return 1;
  }
}

void LogResultVector(const std::string& caseName, const char* tag, const std::vector<double>& data) {
  for (size_t i = 0; i < data.size(); ++i) {
    LOG_PRINT("[CASE %s] %s[%lld] = %.6f\n", caseName.c_str(), tag, static_cast<long long>(i), data[i]);
  }
}

void ReleaseAclTensor(aclTensor*& tensor) {
  if (tensor != nullptr) {
    aclDestroyTensor(tensor);
    tensor = nullptr;
  }
}

void ReleaseAclScalar(aclScalar*& scalar) {
  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
    scalar = nullptr;
  }
}

void ReleaseDeviceAddr(void*& deviceAddr) {
  if (deviceAddr != nullptr) {
    aclrtFree(deviceAddr);
    deviceAddr = nullptr;
  }
}

struct PowTensorScalarCase {
  std::string name;
  aclDataType selfType;
  aclDataType exponentType;
  aclDataType outType;
  std::vector<int64_t> selfShape;
  std::vector<int64_t> outShape;
  std::vector<double> selfData;
  double exponentValue;
  bool expectSuccess;
  bool strictCheck;
  aclFormat selfFormat = ACL_FORMAT_ND;
  aclFormat outFormat = ACL_FORMAT_ND;
};

struct PowScalarTensorCase {
  std::string name;
  aclDataType selfScalarType;
  aclDataType exponentType;
  aclDataType outType;
  std::vector<int64_t> exponentShape;
  std::vector<int64_t> outShape;
  double selfScalarValue;
  std::vector<double> exponentData;
  bool expectSuccess;
  bool strictCheck;
  aclFormat exponentFormat = ACL_FORMAT_ND;
  aclFormat outFormat = ACL_FORMAT_ND;
};

bool ValidatePowTensorScalarCase(const PowTensorScalarCase& testCase) {
  auto selfSize = GetShapeSize(testCase.selfShape);
  auto outSize = GetShapeSize(testCase.outShape);
  if (selfSize < 0 || outSize < 0) {
    LOG_PRINT("[CASE %s] invalid shape size.\n", testCase.name.c_str());
    return false;
  }
  if (selfSize != static_cast<int64_t>(testCase.selfData.size())) {
    LOG_PRINT("[CASE %s] selfData size mismatch. expect=%lld actual=%lld\n",
              testCase.name.c_str(),
              static_cast<long long>(selfSize),
              static_cast<long long>(testCase.selfData.size()));
    return false;
  }
  return true;
}

bool ValidatePowScalarTensorCase(const PowScalarTensorCase& testCase) {
  auto expSize = GetShapeSize(testCase.exponentShape);
  auto outSize = GetShapeSize(testCase.outShape);
  if (expSize < 0 || outSize < 0) {
    LOG_PRINT("[CASE %s] invalid shape size.\n", testCase.name.c_str());
    return false;
  }
  if (expSize != static_cast<int64_t>(testCase.exponentData.size())) {
    LOG_PRINT("[CASE %s] exponentData size mismatch. expect=%lld actual=%lld\n",
              testCase.name.c_str(),
              static_cast<long long>(expSize),
              static_cast<long long>(testCase.exponentData.size()));
    return false;
  }
  return true;
}

int RunPowTensorScalarCase(const PowTensorScalarCase& testCase, aclrtStream stream) {
  if (!ValidatePowTensorScalarCase(testCase)) {
    return 1;
  }

  LOG_PRINT("\n[CASE %s] tensor-scalar begin, selfType=%d, exponentType=%d, outType=%d, exponent=%.6f\n",
            testCase.name.c_str(),
            static_cast<int>(testCase.selfType),
            static_cast<int>(testCase.exponentType),
            static_cast<int>(testCase.outType),
            testCase.exponentValue);

  int ret = ACL_SUCCESS;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  void* inplaceWorkspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  uint64_t inplaceWorkspaceSize = 0;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* exponent = nullptr;
  aclOpExecutor* executor = nullptr;
  aclOpExecutor* inplaceExecutor = nullptr;

  auto cleanup = [&]() {
    ReleaseAclTensor(self);
    ReleaseAclTensor(out);
    ReleaseAclScalar(exponent);
    ReleaseDeviceAddr(selfDeviceAddr);
    ReleaseDeviceAddr(outDeviceAddr);
    ReleaseDeviceAddr(workspaceAddr);
    ReleaseDeviceAddr(inplaceWorkspaceAddr);
  };

  ret = CreateAclTensorFromDoubleData(testCase.selfData, testCase.selfShape, testCase.selfType,
                                      testCase.selfFormat, &selfDeviceAddr, &self);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }
  ret = CreateAclScalarFromDouble(testCase.exponentValue, testCase.exponentType, &exponent);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }
  ret = CreateZeroAclTensor(testCase.outShape, testCase.outType, testCase.outFormat, &outDeviceAddr, &out);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }

  ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] allocate workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
  }
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnPowTensorScalar failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &inplaceWorkspaceSize, &inplaceExecutor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnInplacePowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] allocate inplace workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
  }
  ret = aclnnInplacePowTensorScalar(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnInplacePowTensorScalar failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclrtSynchronizeStream failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  if (testCase.expectSuccess) {
    std::vector<double> outResult;
    ret = CopyTensorToHostDouble(testCase.outShape, testCase.outType, outDeviceAddr, outResult);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] copy out result failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
    LogResultVector(testCase.name, "pow", outResult);

    std::vector<double> inplaceResult;
    ret = CopyTensorToHostDouble(testCase.selfShape, testCase.selfType, selfDeviceAddr, inplaceResult);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] copy inplace result failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
    LogResultVector(testCase.name, "inplace", inplaceResult);
  }

  cleanup();
  return ret;
}

int RunPowScalarTensorCase(const PowScalarTensorCase& testCase, aclrtStream stream) {
  if (!ValidatePowScalarTensorCase(testCase)) {
    return 1;
  }

  LOG_PRINT("\n[CASE %s] scalar-tensor begin, scalarType=%d, exponentType=%d, outType=%d, scalar=%.6f\n",
            testCase.name.c_str(),
            static_cast<int>(testCase.selfScalarType),
            static_cast<int>(testCase.exponentType),
            static_cast<int>(testCase.outType),
            testCase.selfScalarValue);

  int ret = ACL_SUCCESS;
  void* exponentDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclTensor* exponent = nullptr;
  aclTensor* out = nullptr;
  aclScalar* selfScalar = nullptr;
  aclOpExecutor* executor = nullptr;

  auto cleanup = [&]() {
    ReleaseAclTensor(exponent);
    ReleaseAclTensor(out);
    ReleaseAclScalar(selfScalar);
    ReleaseDeviceAddr(exponentDeviceAddr);
    ReleaseDeviceAddr(outDeviceAddr);
    ReleaseDeviceAddr(workspaceAddr);
  };

  ret = CreateAclScalarFromDouble(testCase.selfScalarValue, testCase.selfScalarType, &selfScalar);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }
  ret = CreateAclTensorFromDoubleData(testCase.exponentData, testCase.exponentShape, testCase.exponentType,
                                      testCase.exponentFormat, &exponentDeviceAddr, &exponent);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }
  ret = CreateZeroAclTensor(testCase.outShape, testCase.outType, testCase.outFormat, &outDeviceAddr, &out);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }

  ret = aclnnPowScalarTensorGetWorkspaceSize(selfScalar, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnPowScalarTensorGetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] allocate workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
  }
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnPowScalarTensor failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclrtSynchronizeStream failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  if (testCase.expectSuccess) {
    std::vector<double> outResult;
    ret = CopyTensorToHostDouble(testCase.outShape, testCase.outType, outDeviceAddr, outResult);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] copy out result failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
    LogResultVector(testCase.name, "powScalarTensor", outResult);
  }

  cleanup();
  return ret;
}

struct PowTensorTensorCase {
  std::string name;
  aclDataType selfType;
  aclDataType exponentType;
  aclDataType outType;
  std::vector<int64_t> selfShape;
  std::vector<int64_t> exponentShape;
  std::vector<int64_t> outShape;
  std::vector<double> selfData;
  std::vector<double> exponentData;
  bool expectSuccess;
  bool strictCheck;
  aclFormat selfFormat = ACL_FORMAT_ND;
  aclFormat exponentFormat = ACL_FORMAT_ND;
  aclFormat outFormat = ACL_FORMAT_ND;
};

bool ValidatePowTensorTensorCase(const PowTensorTensorCase& testCase) {
  auto selfSize = GetShapeSize(testCase.selfShape);
  auto expSize = GetShapeSize(testCase.exponentShape);
  auto outSize = GetShapeSize(testCase.outShape);
  if (selfSize < 0 || expSize < 0 || outSize < 0) {
    LOG_PRINT("[CASE %s] invalid shape size.\n", testCase.name.c_str());
    return false;
  }
  if (selfSize != static_cast<int64_t>(testCase.selfData.size())) {
    LOG_PRINT("[CASE %s] selfData size mismatch. expect=%lld actual=%lld\n",
              testCase.name.c_str(),
              static_cast<long long>(selfSize),
              static_cast<long long>(testCase.selfData.size()));
    return false;
  }
  if (expSize != static_cast<int64_t>(testCase.exponentData.size())) {
    LOG_PRINT("[CASE %s] exponentData size mismatch. expect=%lld actual=%lld\n",
              testCase.name.c_str(),
              static_cast<long long>(expSize),
              static_cast<long long>(testCase.exponentData.size()));
    return false;
  }
  return true;
}

int RunPowTensorTensorCase(const PowTensorTensorCase& testCase, aclrtStream stream) {
  if (!ValidatePowTensorTensorCase(testCase)) {
    return 1;
  }

  LOG_PRINT("\n[CASE %s] tensor-tensor begin, selfType=%d, exponentType=%d, outType=%d\n",
            testCase.name.c_str(),
            static_cast<int>(testCase.selfType),
            static_cast<int>(testCase.exponentType),
            static_cast<int>(testCase.outType));

  int ret = ACL_SUCCESS;
  void* selfDeviceAddr = nullptr;
  void* exponentDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  void* inplaceWorkspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  uint64_t inplaceWorkspaceSize = 0;
  aclTensor* self = nullptr;
  aclTensor* exponent = nullptr;
  aclTensor* out = nullptr;
  aclOpExecutor* executor = nullptr;
  aclOpExecutor* inplaceExecutor = nullptr;

  auto cleanup = [&]() {
    ReleaseAclTensor(self);
    ReleaseAclTensor(exponent);
    ReleaseAclTensor(out);
    ReleaseDeviceAddr(selfDeviceAddr);
    ReleaseDeviceAddr(exponentDeviceAddr);
    ReleaseDeviceAddr(outDeviceAddr);
    ReleaseDeviceAddr(workspaceAddr);
    ReleaseDeviceAddr(inplaceWorkspaceAddr);
  };

  ret = CreateAclTensorFromDoubleData(testCase.selfData, testCase.selfShape, testCase.selfType,
                                      testCase.selfFormat, &selfDeviceAddr, &self);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }
  ret = CreateAclTensorFromDoubleData(testCase.exponentData, testCase.exponentShape, testCase.exponentType,
                                      testCase.exponentFormat, &exponentDeviceAddr, &exponent);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }
  ret = CreateZeroAclTensor(testCase.outShape, testCase.outType, testCase.outFormat, &outDeviceAddr, &out);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }

  ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnPowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] allocate workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
  }
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnPowTensorTensor failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  // Inplace path is only executed for shape-compatible tensor pairs.
  bool runInplace = (testCase.selfShape == testCase.exponentShape);
  if (runInplace) {
    ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self, exponent, &inplaceWorkspaceSize, &inplaceExecutor);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] aclnnInplacePowTensorTensorGetWorkspaceSize failed. ERROR: %d\n",
                testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
    if (inplaceWorkspaceSize > 0) {
      ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_SUCCESS) {
        LOG_PRINT("[CASE %s] allocate inplace workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
        cleanup();
        return ret;
      }
    }
    ret = aclnnInplacePowTensorTensor(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] aclnnInplacePowTensorTensor failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclrtSynchronizeStream failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  if (testCase.expectSuccess) {
    std::vector<double> outResult;
    ret = CopyTensorToHostDouble(testCase.outShape, testCase.outType, outDeviceAddr, outResult);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] copy out result failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
    LogResultVector(testCase.name, "powTensorTensor", outResult);

    if (runInplace) {
      std::vector<double> inplaceResult;
      ret = CopyTensorToHostDouble(testCase.selfShape, testCase.selfType, selfDeviceAddr, inplaceResult);
      if (ret != ACL_SUCCESS) {
        LOG_PRINT("[CASE %s] copy inplace result failed. ERROR: %d\n", testCase.name.c_str(), ret);
        cleanup();
        return ret;
      }
      LogResultVector(testCase.name, "inplaceTensorTensor", inplaceResult);
    }
  }

  cleanup();
  return ret;
}

int RunPowTensorTensorApiValidationTests() {
  int overallRet = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  auto update = [&](const std::string& name, aclnnStatus ret, bool expectSuccess, bool strictCheck) {
    bool matched = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    LOG_PRINT("[API %s] ret=%d, expectSuccess=%d, matched=%d\n",
              name.c_str(),
              ret,
              expectSuccess ? 1 : 0,
              matched ? 1 : 0);
    if (!matched && strictCheck) {
      overallRet = 1;
    }
  };

  void* selfAddr = nullptr;
  void* expAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  auto release = [&]() {
    ReleaseAclTensor(self);
    ReleaseAclTensor(exp);
    ReleaseAclTensor(out);
    ReleaseDeviceAddr(selfAddr);
    ReleaseDeviceAddr(expAddr);
    ReleaseDeviceAddr(outAddr);
  };

  std::vector<int64_t> shape = {2, 2};
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  if (CreateAclTensorFromDoubleData(data, shape, ACL_FLOAT, &selfAddr, &self) != ACL_SUCCESS ||
      CreateAclTensorFromDoubleData(data, shape, ACL_FLOAT, &expAddr, &exp) != ACL_SUCCESS ||
      CreateZeroAclTensor(shape, ACL_FLOAT, &outAddr, &out) != ACL_SUCCESS) {
    LOG_PRINT("[API tensor_tensor] init validation tensors failed.\n");
    release();
    return 1;
  }

  update("tensor_tensor_null_self",
         aclnnPowTensorTensorGetWorkspaceSize(nullptr, exp, out, &workspaceSize, &executor),
         false, true);
  update("tensor_tensor_null_exponent",
         aclnnPowTensorTensorGetWorkspaceSize(self, nullptr, out, &workspaceSize, &executor),
         false, true);
  update("tensor_tensor_null_out",
         aclnnPowTensorTensorGetWorkspaceSize(self, exp, nullptr, &workspaceSize, &executor),
         false, true);
  update("inplace_tensor_tensor_null_self",
         aclnnInplacePowTensorTensorGetWorkspaceSize(nullptr, exp, &workspaceSize, &executor),
         false, true);
  update("inplace_tensor_tensor_null_exponent",
         aclnnInplacePowTensorTensorGetWorkspaceSize(self, nullptr, &workspaceSize, &executor),
         false, true);
  release();

  if (CreateAclTensorFromDoubleData(data, shape, ACL_UINT64, &selfAddr, &self) == ACL_SUCCESS &&
      CreateAclTensorFromDoubleData(data, shape, ACL_UINT64, &expAddr, &exp) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape, ACL_UINT64, &outAddr, &out) == ACL_SUCCESS) {
    update("tensor_tensor_unsupported_uint64",
           aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API tensor_tensor_unsupported_uint64] skipped: create tensor failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data, shape, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
      CreateAclTensorFromDoubleData(data, shape, ACL_FLOAT, &expAddr, &exp) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape, ACL_INT32, &outAddr, &out) == ACL_SUCCESS) {
    update("tensor_tensor_out_dtype_mismatch",
           aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API tensor_tensor_out_dtype_mismatch] skipped: create tensor failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data, std::vector<int64_t>{2, 1}, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
      CreateAclTensorFromDoubleData(data, std::vector<int64_t>{2, 2}, ACL_FLOAT, &expAddr, &exp) == ACL_SUCCESS) {
    update("inplace_tensor_tensor_illegal_broadcast",
           aclnnInplacePowTensorTensorGetWorkspaceSize(self, exp, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API inplace_tensor_tensor_illegal_broadcast] skipped: create tensor failed.\n");
  }
  release();

  return overallRet;
}

struct Exp2Case {
  std::string name;
  aclDataType selfType;
  aclDataType outType;
  std::vector<int64_t> shape;
  std::vector<double> selfData;
  bool expectSuccess;
  bool strictCheck;
  aclFormat selfFormat = ACL_FORMAT_ND;
  aclFormat outFormat = ACL_FORMAT_ND;
};

bool ValidateExp2Case(const Exp2Case& testCase) {
  auto size = GetShapeSize(testCase.shape);
  if (size < 0) {
    LOG_PRINT("[CASE %s] invalid shape size.\n", testCase.name.c_str());
    return false;
  }
  if (size != static_cast<int64_t>(testCase.selfData.size())) {
    LOG_PRINT("[CASE %s] selfData size mismatch. expect=%lld actual=%lld\n",
              testCase.name.c_str(),
              static_cast<long long>(size),
              static_cast<long long>(testCase.selfData.size()));
    return false;
  }
  return true;
}

int RunExp2Case(const Exp2Case& testCase, aclrtStream stream) {
  if (!ValidateExp2Case(testCase)) {
    return 1;
  }

  LOG_PRINT("\n[CASE %s] exp2 begin, selfType=%d, outType=%d\n",
            testCase.name.c_str(),
            static_cast<int>(testCase.selfType),
            static_cast<int>(testCase.outType));

  int ret = ACL_SUCCESS;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  void* inplaceWorkspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  uint64_t inplaceWorkspaceSize = 0;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclOpExecutor* executor = nullptr;
  aclOpExecutor* inplaceExecutor = nullptr;

  auto cleanup = [&]() {
    ReleaseAclTensor(self);
    ReleaseAclTensor(out);
    ReleaseDeviceAddr(selfDeviceAddr);
    ReleaseDeviceAddr(outDeviceAddr);
    ReleaseDeviceAddr(workspaceAddr);
    ReleaseDeviceAddr(inplaceWorkspaceAddr);
  };

  ret = CreateAclTensorFromDoubleData(testCase.selfData, testCase.shape, testCase.selfType,
                                      testCase.selfFormat, &selfDeviceAddr, &self);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }
  ret = CreateZeroAclTensor(testCase.shape, testCase.outType, testCase.outFormat, &outDeviceAddr, &out);
  if (ret != ACL_SUCCESS) {
    cleanup();
    return ret;
  }

  ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnExp2GetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] allocate workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
  }
  ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnExp2 failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  ret = aclnnInplaceExp2GetWorkspaceSize(self, &inplaceWorkspaceSize, &inplaceExecutor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnInplaceExp2GetWorkspaceSize failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] allocate inplace workspace failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
  }
  ret = aclnnInplaceExp2(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclnnInplaceExp2 failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[CASE %s] aclrtSynchronizeStream failed. ERROR: %d\n", testCase.name.c_str(), ret);
    cleanup();
    return ret;
  }

  if (testCase.expectSuccess) {
    std::vector<double> outResult;
    ret = CopyTensorToHostDouble(testCase.shape, testCase.outType, outDeviceAddr, outResult);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] copy exp2 out result failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
    LogResultVector(testCase.name, "exp2", outResult);

    std::vector<double> inplaceResult;
    ret = CopyTensorToHostDouble(testCase.shape, testCase.selfType, selfDeviceAddr, inplaceResult);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[CASE %s] copy inplace exp2 result failed. ERROR: %d\n", testCase.name.c_str(), ret);
      cleanup();
      return ret;
    }
    LogResultVector(testCase.name, "inplaceExp2", inplaceResult);
  }

  cleanup();
  return ret;
}

int RunExp2ApiValidationTests() {
  int overallRet = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  auto update = [&](const std::string& name, aclnnStatus ret, bool expectSuccess, bool strictCheck) {
    bool matched = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    LOG_PRINT("[API %s] ret=%d, expectSuccess=%d, matched=%d\n",
              name.c_str(),
              ret,
              expectSuccess ? 1 : 0,
              matched ? 1 : 0);
    if (!matched && strictCheck) {
      overallRet = 1;
    }
  };

  std::vector<int64_t> shape = {2, 2};
  std::vector<double> data = {0.0, 1.0, 2.0, 3.0};
  void* selfAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  auto release = [&]() {
    ReleaseAclTensor(self);
    ReleaseAclTensor(out);
    ReleaseDeviceAddr(selfAddr);
    ReleaseDeviceAddr(outAddr);
  };

  if (CreateAclTensorFromDoubleData(data, shape, ACL_FLOAT, &selfAddr, &self) != ACL_SUCCESS ||
      CreateZeroAclTensor(shape, ACL_FLOAT, &outAddr, &out) != ACL_SUCCESS) {
    LOG_PRINT("[API exp2] init validation tensors failed.\n");
    release();
    return 1;
  }
  update("exp2_null_self", aclnnExp2GetWorkspaceSize(nullptr, out, &workspaceSize, &executor), false, true);
  update("exp2_null_out", aclnnExp2GetWorkspaceSize(self, nullptr, &workspaceSize, &executor), false, true);
  update("inplace_exp2_null_self", aclnnInplaceExp2GetWorkspaceSize(nullptr, &workspaceSize, &executor), false, true);
  release();

  if (CreateAclTensorFromDoubleData(data, shape, ACL_UINT64, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape, ACL_UINT64, &outAddr, &out) == ACL_SUCCESS) {
    update("exp2_unsupported_uint64", aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor), false, true);
  } else {
    LOG_PRINT("[API exp2_unsupported_uint64] skipped: create tensor failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data, shape, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape, ACL_INT32, &outAddr, &out) == ACL_SUCCESS) {
    update("exp2_out_dtype_mismatch", aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor), false, true);
  } else {
    LOG_PRINT("[API exp2_out_dtype_mismatch] skipped: create tensor failed.\n");
  }
  release();

  return overallRet;
}

void UpdateCaseResult(const std::string& caseName, int ret, bool expectSuccess, bool strictCheck, int& overallRet) {
  bool matched = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
  LOG_PRINT("[CASE %s] end, ret=%d, expectSuccess=%d, matched=%d\n",
            caseName.c_str(),
            ret,
            expectSuccess ? 1 : 0,
            matched ? 1 : 0);
  if (!matched && strictCheck) {
    overallRet = 1;
  }
}

int RunApiValidationTests() {
  int overallRet = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  auto update = [&](const std::string& name, aclnnStatus ret, bool expectSuccess, bool strictCheck) {
    bool matched = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    LOG_PRINT("[API %s] ret=%d, expectSuccess=%d, matched=%d\n",
              name.c_str(),
              ret,
              expectSuccess ? 1 : 0,
              matched ? 1 : 0);
    if (!matched && strictCheck) {
      overallRet = 1;
    }
  };

  void* selfAddr = nullptr;
  void* expAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;

  auto release = [&]() {
    ReleaseAclTensor(self);
    ReleaseAclTensor(exp);
    ReleaseAclTensor(out);
    ReleaseAclScalar(scalar);
    ReleaseDeviceAddr(selfAddr);
    ReleaseDeviceAddr(expAddr);
    ReleaseDeviceAddr(outAddr);
  };

  std::vector<int64_t> shape2d = {2, 2};
  std::vector<double> data2d = {1.0, 2.0, 3.0, 4.0};
  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT, &selfAddr, &self) != ACL_SUCCESS ||
      CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT, &expAddr, &exp) != ACL_SUCCESS ||
      CreateZeroAclTensor(shape2d, ACL_FLOAT, &outAddr, &out) != ACL_SUCCESS ||
      CreateAclScalarFromDouble(2.0, ACL_FLOAT, &scalar) != ACL_SUCCESS) {
    LOG_PRINT("[API] init validation tensors failed.\n");
    release();
    return 1;
  }

  update("tensor_scalar_null_self",
         aclnnPowTensorScalarGetWorkspaceSize(nullptr, scalar, out, &workspaceSize, &executor),
         false, true);
  update("tensor_scalar_null_exponent",
         aclnnPowTensorScalarGetWorkspaceSize(self, nullptr, out, &workspaceSize, &executor),
         false, true);
  update("tensor_scalar_null_out",
         aclnnPowTensorScalarGetWorkspaceSize(self, scalar, nullptr, &workspaceSize, &executor),
         false, true);

  update("scalar_tensor_null_self",
         aclnnPowScalarTensorGetWorkspaceSize(nullptr, exp, out, &workspaceSize, &executor),
         false, true);
  update("scalar_tensor_null_exponent",
         aclnnPowScalarTensorGetWorkspaceSize(scalar, nullptr, out, &workspaceSize, &executor),
         false, true);
  update("scalar_tensor_null_out",
         aclnnPowScalarTensorGetWorkspaceSize(scalar, exp, nullptr, &workspaceSize, &executor),
         false, true);

  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_UINT32, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_UINT32, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclScalarFromDouble(2.0, ACL_UINT32, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_unsupported_uint32",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API tensor_scalar_unsupported_uint32] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclScalarFromDouble(2.0, ACL_UINT64, &scalar) == ACL_SUCCESS &&
      CreateAclTensorFromDoubleData(data2d, shape2d, ACL_UINT64, &expAddr, &exp) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_UINT64, &outAddr, &out) == ACL_SUCCESS) {
    update("scalar_tensor_unsupported_uint64",
           aclnnPowScalarTensorGetWorkspaceSize(scalar, exp, out, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API scalar_tensor_unsupported_uint64] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_INT32, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclScalarFromDouble(2.0, ACL_FLOAT, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_out_dtype_mismatch",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API tensor_scalar_out_dtype_mismatch] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_FLOAT, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclScalarFromDouble(2.0, ACL_UINT32, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_exp_uint32_invalid",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API tensor_scalar_exp_uint32_invalid] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclScalarFromDouble(2.0, ACL_FLOAT, &scalar) == ACL_SUCCESS &&
      CreateAclTensorFromDoubleData(data2d, shape2d, ACL_UINT64, &expAddr, &exp) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_FLOAT, &outAddr, &out) == ACL_SUCCESS) {
    update("scalar_tensor_exp_uint64_invalid",
           aclnnPowScalarTensorGetWorkspaceSize(scalar, exp, out, &workspaceSize, &executor),
           false, true);
  } else {
    LOG_PRINT("[API scalar_tensor_exp_uint64_invalid] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT16, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_FLOAT16, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclScalarFromDouble(1.0e20, ACL_DOUBLE, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_float16_overflow_check",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           false, false);
  } else {
    LOG_PRINT("[API tensor_scalar_float16_overflow_check] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_BF16, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_BF16, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclScalarFromDouble(1.0e300, ACL_DOUBLE, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_bf16_overflow_check",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           false, false);
  } else {
    LOG_PRINT("[API tensor_scalar_bf16_overflow_check] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_INT16, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_INT16, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclScalarFromDouble(40000.0, ACL_DOUBLE, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_int16_overflow_check",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           false, false);
  } else {
    LOG_PRINT("[API tensor_scalar_int16_overflow_check] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT16, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_COMPLEX64, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclComplexScalar(2.0, 0.0, ACL_COMPLEX64, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_fp16_complex_scalar_promote",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           true, false);
  } else {
    LOG_PRINT("[API tensor_scalar_fp16_complex_scalar_promote] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT16, &selfAddr, &self) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_COMPLEX64, &outAddr, &out) == ACL_SUCCESS &&
      CreateAclComplexScalar(1.0e300, 1.0e300, ACL_COMPLEX128, &scalar) == ACL_SUCCESS) {
    update("tensor_scalar_complex_overflow_check",
           aclnnPowTensorScalarGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor),
           false, false);
  } else {
    LOG_PRINT("[API tensor_scalar_complex_overflow_check] skipped: create tensor/scalar failed.\n");
  }
  release();

  if (CreateAclComplexScalar(2.0, 0.0, ACL_COMPLEX64, &scalar) == ACL_SUCCESS &&
      CreateAclTensorFromDoubleData(data2d, shape2d, ACL_FLOAT16, &expAddr, &exp) == ACL_SUCCESS &&
      CreateZeroAclTensor(shape2d, ACL_COMPLEX64, &outAddr, &out) == ACL_SUCCESS) {
    update("scalar_tensor_complex_scalar_with_fp16_exp",
           aclnnPowScalarTensorGetWorkspaceSize(scalar, exp, out, &workspaceSize, &executor),
           true, false);
  } else {
    LOG_PRINT("[API scalar_tensor_complex_scalar_with_fp16_exp] skipped: create tensor/scalar failed.\n");
  }
  release();

  return overallRet;
}

int RunL0PowBroadcastFailCoverageCaseFromPowExample() {
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> expShape = {2};
  std::vector<double> selfData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> expData = {2.0, 3.0};

  void* selfAddr = nullptr;
  void* expAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;

  int ret = CreateAclTensorFromDoubleData(selfData, selfShape, ACL_FLOAT, &selfAddr, &self);
  if (ret != ACL_SUCCESS) {
    ReleaseAclTensor(self);
    ReleaseAclTensor(exp);
    ReleaseDeviceAddr(selfAddr);
    ReleaseDeviceAddr(expAddr);
    return ret;
  }
  ret = CreateAclTensorFromDoubleData(expData, expShape, ACL_FLOAT, &expAddr, &exp);
  if (ret != ACL_SUCCESS) {
    ReleaseAclTensor(self);
    ReleaseAclTensor(exp);
    ReleaseDeviceAddr(selfAddr);
    ReleaseDeviceAddr(expAddr);
    return ret;
  }

  // Coverage-only: force l0op::Pow broadcast failure branch in op_api/pow.cpp.
  const aclTensor* out = l0op::Pow(self, exp, nullptr);
  bool matched = (out == nullptr);
  LOG_PRINT("[L0Pow broadcast_fail from pow example] matched=%d\n", matched ? 1 : 0);

  ReleaseAclTensor(self);
  ReleaseAclTensor(exp);
  ReleaseDeviceAddr(selfAddr);
  ReleaseDeviceAddr(expAddr);
  return matched ? 0 : 1;
}

int main() {
  const int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  int ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<PowTensorScalarCase> tensorScalarCases = {
      {"tensor_scalar_float_general_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {0.0, 1.0, 2.0, 3.0},
       4.1,
       true, true},
      {"tensor_scalar_float_square_branch",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {0.0, 1.0, 2.0, 3.0},
       2.0,
       true, true},
      {"tensor_scalar_float_exp_zero_boundary",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       0.0,
       true, true},
      {"tensor_scalar_float_exp_one_boundary",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       1.0,
       true, true},
      {"tensor_scalar_float_exp_half_sqrt_branch",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 4.0, 9.0, 16.0},
       0.5,
       true, true},
      {"tensor_scalar_float_exp_cube_branch",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       3.0,
       true, true},
      {"tensor_scalar_float_exp_negative_one_branch",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 4.0, 8.0},
       -1.0,
       true, true},
      {"tensor_scalar_float_exp_negative_two_branch",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 4.0, 8.0},
       -2.0,
       true, true},
      {"tensor_scalar_float_double_scalar_branch",
       ACL_FLOAT, ACL_DOUBLE, ACL_FLOAT,
       {2, 2}, {2, 2},
       {0.5, 1.5, 2.5, 3.5},
       2.0,
       true, true},
      {"tensor_scalar_float_nan_boundary",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       std::numeric_limits<double>::quiet_NaN(),
       true, false},
      {"tensor_scalar_float_inf_boundary",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       std::numeric_limits<double>::infinity(),
       false, false},
      {"tensor_scalar_int64_square_branch",
       ACL_INT64, ACL_INT64, ACL_INT64,
       {2, 2}, {2, 2},
       {-3.0, -1.0, 0.0, 2.0},
       2.0,
       true, true},
      {"tensor_scalar_int8_int16_success",
       ACL_INT8, ACL_INT16, ACL_INT8,
       {2, 2}, {2, 2},
       {-8.0, -1.0, 0.0, 3.0},
       2.0,
       true, true},
      {"tensor_scalar_uint8_int16_success",
       ACL_UINT8, ACL_INT16, ACL_UINT8,
       {2, 2}, {2, 2},
       {0.0, 1.0, 2.0, 3.0},
       2.0,
       true, true},
      {"tensor_scalar_empty_self_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {0, 4}, {0, 4},
       {},
       2.0,
       true, true},
      {"tensor_scalar_int32_negative_exponent_expect_fail",
       ACL_INT32, ACL_INT32, ACL_INT32,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       -1.0,
       false, true},
      {"tensor_scalar_int8_int16_overflow_expect_fail",
       ACL_INT8, ACL_INT16, ACL_INT8,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       130.0,
       false, true},
      {"tensor_scalar_uint8_int16_overflow_expect_fail",
       ACL_UINT8, ACL_INT16, ACL_UINT8,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       300.0,
       false, true},
      {"tensor_scalar_float_exponent_overflow_expect_fail",
       ACL_FLOAT, ACL_DOUBLE, ACL_FLOAT,
       {2, 2}, {2, 2},
       {1.0, 2.0, 3.0, 4.0},
       1.0e39,
       false, false},
      {"tensor_scalar_bool_bool_expect_fail",
       ACL_BOOL, ACL_BOOL, ACL_BOOL,
       {2, 2}, {2, 2},
       {0.0, 1.0, 0.0, 1.0},
       1.0,
       false, true},
      {"tensor_scalar_shape_mismatch_expect_fail",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {1, 4},
       {1.0, 2.0, 3.0, 4.0},
       2.0,
       false, true},
      {"tensor_scalar_nchw_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {1, 1, 2, 2}, {1, 1, 2, 2},
       {0.0, 1.0, 2.0, 3.0},
       2.0,
       true, true, ACL_FORMAT_NCHW, ACL_FORMAT_NCHW}};

  std::vector<PowScalarTensorCase> scalarTensorCases = {
      {"scalar_tensor_fill_one_branch",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       1.0,
       {1.0, 2.0, 3.0, 4.0},
       true, true},
      {"scalar_tensor_general_compute_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       2.0,
       {1.0, 2.0, 3.0, 4.0},
       true, true},
      {"scalar_tensor_double_exp_out_float_branch",
       ACL_FLOAT, ACL_DOUBLE, ACL_FLOAT,
       {2, 2}, {2, 2},
       2.0,
       {0.5, 1.5, 2.5, 3.5},
       true, true},
      {"scalar_tensor_special_exponents_mix",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 3}, {2, 3},
       2.0,
       {0.0, 1.0, 0.5, 2.0, -1.0, -2.0},
       true, true},
      {"scalar_tensor_nan_boundary",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {2, 2},
       2.0,
       {0.0, std::numeric_limits<double>::quiet_NaN(), 1.0, -1.0},
       true, false},
      {"scalar_tensor_int32_success",
       ACL_INT32, ACL_INT32, ACL_INT32,
       {2, 2}, {2, 2},
       2.0,
       {1.0, 2.0, 3.0, 4.0},
       true, true},
      {"scalar_tensor_int64_success",
       ACL_INT64, ACL_INT64, ACL_INT64,
       {2, 2}, {2, 2},
       2.0,
       {1.0, 2.0, 3.0, 4.0},
       true, true},
      {"scalar_tensor_empty_exponent_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {0, 4}, {0, 4},
       2.0,
       {},
       true, true},
      {"scalar_tensor_bool_bool_expect_fail",
       ACL_BOOL, ACL_BOOL, ACL_BOOL,
       {2, 2}, {2, 2},
       1.0,
       {0.0, 1.0, 0.0, 1.0},
       false, true},
      {"scalar_tensor_shape_mismatch_expect_fail",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 2}, {1, 4},
       2.0,
       {1.0, 2.0, 3.0, 4.0},
       false, true},
      {"scalar_tensor_nchw_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {1, 1, 2, 2}, {1, 1, 2, 2},
       2.0,
       {0.0, 1.0, 2.0, 3.0},
       true, true, ACL_FORMAT_NCHW, ACL_FORMAT_NCHW}};

  std::vector<PowTensorTensorCase> tensorTensorCases = {
      {"tensor_tensor_float_same_shape_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 3}, {2, 3}, {2, 3},
       {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
       {0.0, 1.0, 2.0, 2.0, 3.0, 1.0},
       true, true},
      {"tensor_tensor_float_broadcast_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 1, 3}, {1, 4, 1}, {2, 4, 3},
       {-3.0, -1.0, 0.0, 1.0, 2.0, 3.0},
       {0.0, 1.0, 2.0, 3.0},
       true, true},
      {"tensor_tensor_int32_boundary_success",
       ACL_INT32, ACL_INT32, ACL_INT32,
       {2, 3}, {2, 3}, {2, 3},
       {-46340.0, -2.0, -1.0, 0.0, 1.0, 46340.0},
       {2.0, 2.0, 1.0, 0.0, 1.0, 2.0},
       true, true},
      {"tensor_tensor_int64_aicpu_success",
       ACL_INT64, ACL_INT64, ACL_INT64,
       {2, 3}, {2, 3}, {2, 3},
       {-9.0, -1.0, 0.0, 1.0, 2.0, 3.0},
       {1.0, 2.0, 1.0, 2.0, 3.0, 2.0},
       true, true},
      {"tensor_tensor_bool_bool_expect_fail",
       ACL_BOOL, ACL_BOOL, ACL_BOOL,
       {2, 2}, {2, 2}, {2, 2},
       {0.0, 1.0, 1.0, 0.0},
       {1.0, 0.0, 1.0, 0.0},
       false, true},
      {"tensor_tensor_out_shape_mismatch_expect_fail",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {2, 1}, {1, 2}, {2, 1},
       {1.0, 2.0},
       {2.0, 3.0},
       false, true},
      {"tensor_tensor_nchw_success",
       ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
       {1, 1, 2, 2}, {1, 1, 2, 2}, {1, 1, 2, 2},
       {0.0, 1.0, 2.0, 3.0},
       {2.0, 2.0, 2.0, 2.0},
       true, true, ACL_FORMAT_NCHW, ACL_FORMAT_NCHW, ACL_FORMAT_NCHW}};

  std::vector<Exp2Case> exp2Cases = {
      {"exp2_float_basic_success",
       ACL_FLOAT, ACL_FLOAT,
       {2, 2},
       {0.0, 1.0, 2.0, 3.0},
       true, true},
      {"exp2_double_basic_success",
       ACL_DOUBLE, ACL_DOUBLE,
       {2, 2},
       {0.0, 1.0, 2.0, 3.0},
       true, true},
      {"exp2_empty_tensor_success",
       ACL_FLOAT, ACL_FLOAT,
       {0, 4},
       {},
       true, true},
      {"exp2_out_dtype_mismatch_expect_fail",
       ACL_FLOAT, ACL_INT32,
       {2, 2},
       {0.0, 1.0, 2.0, 3.0},
       false, true}};

  int overallRet = 0;
  for (const auto& testCase : tensorScalarCases) {
    ret = RunPowTensorScalarCase(testCase, stream);
    UpdateCaseResult(testCase.name, ret, testCase.expectSuccess, testCase.strictCheck, overallRet);
  }
  for (const auto& testCase : scalarTensorCases) {
    ret = RunPowScalarTensorCase(testCase, stream);
    UpdateCaseResult(testCase.name, ret, testCase.expectSuccess, testCase.strictCheck, overallRet);
  }
  for (const auto& testCase : tensorTensorCases) {
    ret = RunPowTensorTensorCase(testCase, stream);
    UpdateCaseResult(testCase.name, ret, testCase.expectSuccess, testCase.strictCheck, overallRet);
  }
  for (const auto& testCase : exp2Cases) {
    ret = RunExp2Case(testCase, stream);
    UpdateCaseResult(testCase.name, ret, testCase.expectSuccess, testCase.strictCheck, overallRet);
  }

  ret = RunApiValidationTests();
  if (ret != 0) {
    overallRet = 1;
  }
  ret = RunPowTensorTensorApiValidationTests();
  if (ret != 0) {
    overallRet = 1;
  }
  ret = RunExp2ApiValidationTests();
  if (ret != 0) {
    overallRet = 1;
  }
  ret = RunL0PowBroadcastFailCoverageCaseFromPowExample();
  if (ret != 0) {
    overallRet = 1;
  }

  ret = aclrtDestroyStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtDestroyStream failed. ERROR: %d\n", ret);
    overallRet = 1;
  }
  ret = aclrtResetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtResetDevice failed. ERROR: %d\n", ret);
    overallRet = 1;
  }
  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclFinalize failed. ERROR: %d\n", ret);
    overallRet = 1;
  }
  return overallRet;
}
