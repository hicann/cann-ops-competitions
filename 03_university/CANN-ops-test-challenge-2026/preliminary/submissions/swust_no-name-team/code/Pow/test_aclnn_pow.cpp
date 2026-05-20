#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdint>
#include <cstring>
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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

static int pass_count = 0;
static int fail_count = 0;

void ReportCase(const std::string& caseName, bool pass) {
    if (pass) {
        LOG_PRINT("[PASS] %s\n", caseName.c_str());
        pass_count++;
    } else {
        LOG_PRINT("[FAIL] %s\n", caseName.c_str());
        fail_count++;
    }
}

bool CompareFloat(const std::vector<float>& actual, const std::vector<float>& expected,
                                    double atol = 1e-4, double rtol = 1e-4) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        double a = static_cast<double>(actual[i]);
        double e = static_cast<double>(expected[i]);
        if (std::isnan(a) && std::isnan(e)) {
            continue;
        }
        if (std::isinf(a) && std::isinf(e) && ((a > 0) == (e > 0))) {
            continue;
        }
        if (std::abs(a - e) > atol + rtol * std::abs(e)) {
            LOG_PRINT("Mismatch idx=%zu, actual=%f, expected=%f\n", i, a, e);
            return false;
        }
    }
    return true;
}

template <typename T>
bool CompareExact(const std::vector<T>& actual, const std::vector<T>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            LOG_PRINT("Mismatch idx=%zu, actual=%lld, expected=%lld\n",
                                i,
                                static_cast<long long>(actual[i]),
                                static_cast<long long>(expected[i]));
            return false;
        }
    }
    return true;
}

static inline float Bf16ToFloat(uint16_t bits) {
    uint32_t full = static_cast<uint32_t>(bits) << 16;
    float out = 0.0f;
    std::memcpy(&out, &full, sizeof(float));
    return out;
}

static inline uint16_t FloatToBf16(float value) {
    uint32_t full = 0;
    std::memcpy(&full, &value, sizeof(float));
    return static_cast<uint16_t>(full >> 16);
}

static inline float Float16ToFloat(uint16_t value) {
    uint32_t sign = (static_cast<uint32_t>(value) >> 15) & 0x1;
    uint32_t exponent = (static_cast<uint32_t>(value) >> 10) & 0x1f;
    uint32_t mantissa = static_cast<uint32_t>(value) & 0x3ff;

    float result = 0.0f;
    if (exponent == 0) {
        result = static_cast<float>(mantissa) * 0.0000019073486328125f;
    } else if (exponent == 31) {
        result = (mantissa == 0) ? (1.0f / 0.0f) : (0.0f / 0.0f);
    } else {
        result = (1.0f + static_cast<float>(mantissa) * 0.0009765625f) * std::pow(2.0f, static_cast<int>(exponent) - 15);
    }
    return sign ? -result : result;
}

static inline uint16_t FloatToFloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(uint32_t));
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;

    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | static_cast<uint16_t>(mantissa >> 13));
}

bool RunPowTensorScalarFloat(aclrtStream stream, const std::string& name,
                                                         const std::vector<float>& base, float exp) {
    std::vector<int64_t> shape = {static_cast<int64_t>(base.size())};
    std::vector<float> outHostData(base.size(), 0.0f);
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* exponent = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    exponent = aclCreateScalar(&exp, ACL_FLOAT);
    ok = ok && (exponent != nullptr);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<float> actual(base.size(), 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);
        std::vector<float> expected(base.size(), 0.0f);
        for (size_t i = 0; i < base.size(); ++i) {
            expected[i] = static_cast<float>(std::pow(static_cast<double>(base[i]), static_cast<double>(exp)));
        }
        ok = ok && CompareFloat(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (out != nullptr) aclDestroyTensor(out);
    if (exponent != nullptr) aclDestroyScalar(exponent);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunInplacePowTensorScalarFloat(aclrtStream stream, const std::string& name,
                                                                        const std::vector<float>& base, float exp) {
    std::vector<int64_t> shape = {static_cast<int64_t>(base.size())};
    void* selfDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* exponent = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    exponent = aclCreateScalar(&exp, ACL_FLOAT);
    ok = ok && (exponent != nullptr);

    if (ok) {
        ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<float> actual(base.size(), 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), selfDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);
        std::vector<float> expected(base.size(), 0.0f);
        for (size_t i = 0; i < base.size(); ++i) {
            expected[i] = static_cast<float>(std::pow(static_cast<double>(base[i]), static_cast<double>(exp)));
        }
        ok = ok && CompareFloat(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyScalar(exponent);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunPowScalarTensorFloat(aclrtStream stream, const std::string& name,
                                                         float base, const std::vector<float>& expTensor) {
    std::vector<int64_t> shape = {static_cast<int64_t>(expTensor.size())};
    std::vector<float> outHostData(expTensor.size(), 0.0f);
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclScalar* self = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    self = aclCreateScalar(&base, ACL_FLOAT);
    ok = ok && (self != nullptr);
    auto ret = CreateAclTensor(expTensor, shape, &expDeviceAddr, ACL_FLOAT, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<float> actual(expTensor.size(), 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);
        std::vector<float> expected(expTensor.size(), 0.0f);
        for (size_t i = 0; i < expTensor.size(); ++i) {
            expected[i] = static_cast<float>(std::pow(static_cast<double>(base), static_cast<double>(expTensor[i])));
        }
        ok = ok && CompareFloat(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyScalar(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

template <typename T>
bool RunPowTensorTensorSameShape(const std::string& name, aclrtStream stream,
                                                                 const std::vector<T>& base, const std::vector<T>& exp,
                                                                 aclDataType dtype) {
    std::vector<int64_t> shape = {static_cast<int64_t>(base.size())};
    std::vector<T> outHostData(base.size(), static_cast<T>(0));
    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, dtype, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(exp, shape, &expDeviceAddr, dtype, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<T> actual(base.size(), static_cast<T>(0));
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(T), outDeviceAddr,
                                            actual.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);

        std::vector<T> expected(base.size(), static_cast<T>(0));
        for (size_t i = 0; i < base.size(); ++i) {
            expected[i] = static_cast<T>(std::pow(static_cast<double>(base[i]), static_cast<double>(exp[i])));
        }
        ok = ok && CompareExact(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunPowTensorTensorDouble(const std::string& name, aclrtStream stream,
                              const std::vector<double>& base, const std::vector<double>& exp) {
    std::vector<int64_t> shape = {static_cast<int64_t>(base.size())};
    std::vector<double> outHostData(base.size(), 0.0);
    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, ACL_DOUBLE, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(exp, shape, &expDeviceAddr, ACL_DOUBLE, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_DOUBLE, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<double> actual(base.size(), 0.0);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(double), outDeviceAddr,
                          actual.size() * sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);

        std::vector<double> expected(base.size(), 0.0);
        for (size_t i = 0; i < base.size(); ++i) {
            expected[i] = std::pow(base[i], exp[i]);
        }

        for (size_t i = 0; ok && i < actual.size(); ++i) {
            double a = actual[i];
            double e = expected[i];
            if (std::abs(a - e) > 1e-8 + 1e-8 * std::abs(e)) {
                ok = false;
                LOG_PRINT("Mismatch idx=%zu, actual=%lf, expected=%lf\n", i, a, e);
            }
        }
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunPowTensorTensorFloat(const std::string& name, aclrtStream stream,
                                                         const std::vector<float>& base, const std::vector<float>& exp,
                                                         double atol = 1e-4, double rtol = 1e-4) {
    std::vector<int64_t> shape = {static_cast<int64_t>(base.size())};
    std::vector<float> outHostData(base.size(), 0.0f);
    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(exp, shape, &expDeviceAddr, ACL_FLOAT, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<float> actual(base.size(), 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);
        std::vector<float> expected(base.size(), 0.0f);
        for (size_t i = 0; i < base.size(); ++i) {
            expected[i] = static_cast<float>(std::pow(static_cast<double>(base[i]), static_cast<double>(exp[i])));
        }
        ok = ok && CompareFloat(actual, expected, atol, rtol);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunPowTensorTensorBroadcastFloat(aclrtStream stream, const std::string& name) {
    std::vector<float> base = {2.0f, 3.0f};
    std::vector<float> exp = {2.0f, 3.0f};
    std::vector<int64_t> baseShape = {2, 1};
    std::vector<int64_t> expShape = {1, 2};
    std::vector<int64_t> outShape = {2, 2};
    std::vector<float> outHostData(4, 0.0f);

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, baseShape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(exp, expShape, &expDeviceAddr, ACL_FLOAT, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<float> actual(4, 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);

        std::vector<float> expected = {
            std::pow(base[0], exp[0]), std::pow(base[0], exp[1]),
            std::pow(base[1], exp[0]), std::pow(base[1], exp[1])
        };
        ok = ok && CompareFloat(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunPowTensorTensorFP16(aclrtStream stream, const std::string& name,
                                                        const std::vector<float>& baseFloat,
                                                        const std::vector<float>& expFloat) {
    std::vector<int64_t> shape = {static_cast<int64_t>(baseFloat.size())};
    std::vector<uint16_t> base(baseFloat.size(), 0);
    std::vector<uint16_t> exp(expFloat.size(), 0);
    std::vector<uint16_t> outHostData(baseFloat.size(), 0);
    for (size_t i = 0; i < baseFloat.size(); ++i) {
        base[i] = FloatToFloat16(baseFloat[i]);
        exp[i] = FloatToFloat16(expFloat[i]);
    }

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, ACL_FLOAT16, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(exp, shape, &expDeviceAddr, ACL_FLOAT16, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT16, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<uint16_t> actualBits(baseFloat.size(), 0);
        ret = aclrtMemcpy(actualBits.data(), actualBits.size() * sizeof(uint16_t), outDeviceAddr,
                                            actualBits.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);

        std::vector<float> actual(baseFloat.size(), 0.0f);
        std::vector<float> expected(baseFloat.size(), 0.0f);
        for (size_t i = 0; i < baseFloat.size(); ++i) {
            actual[i] = Float16ToFloat(actualBits[i]);
            expected[i] = static_cast<float>(std::pow(static_cast<double>(baseFloat[i]), static_cast<double>(expFloat[i])));
        }
        ok = ok && CompareFloat(actual, expected, 3e-2, 3e-2);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunPowTensorTensorBF16(aclrtStream stream, const std::string& name,
                                                        const std::vector<float>& baseFloat,
                                                        const std::vector<float>& expFloat) {
    std::vector<int64_t> shape = {static_cast<int64_t>(baseFloat.size())};
    std::vector<uint16_t> base(baseFloat.size(), 0);
    std::vector<uint16_t> exp(expFloat.size(), 0);
    std::vector<uint16_t> outHostData(baseFloat.size(), 0);
    for (size_t i = 0; i < baseFloat.size(); ++i) {
        base[i] = FloatToBf16(baseFloat[i]);
        exp[i] = FloatToBf16(expFloat[i]);
    }

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, ACL_BF16, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(exp, shape, &expDeviceAddr, ACL_BF16, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_BF16, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<uint16_t> actualBits(baseFloat.size(), 0);
        ret = aclrtMemcpy(actualBits.data(), actualBits.size() * sizeof(uint16_t), outDeviceAddr,
                                            actualBits.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);

        std::vector<float> actual(baseFloat.size(), 0.0f);
        std::vector<float> expected(baseFloat.size(), 0.0f);
        for (size_t i = 0; i < baseFloat.size(); ++i) {
            actual[i] = Bf16ToFloat(actualBits[i]);
            expected[i] = static_cast<float>(std::pow(static_cast<double>(baseFloat[i]), static_cast<double>(expFloat[i])));
        }
        ok = ok && CompareFloat(actual, expected, 2e-2, 2e-2);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunInplacePowTensorTensorFloat(aclrtStream stream, const std::string& name,
                                                                        const std::vector<float>& base, const std::vector<float>& exp) {
    std::vector<int64_t> shape = {static_cast<int64_t>(base.size())};
    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(base, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(exp, shape, &expDeviceAddr, ACL_FLOAT, &exponent);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }

    if (ok) {
        std::vector<float> actual(base.size(), 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), selfDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);
        std::vector<float> expected(base.size(), 0.0f);
        for (size_t i = 0; i < base.size(); ++i) {
            expected[i] = static_cast<float>(std::pow(static_cast<double>(base[i]), static_cast<double>(exp[i])));
        }
        ok = ok && CompareFloat(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunExp2Float(aclrtStream stream, const std::string& name, const std::vector<float>& selfData) {
    std::vector<int64_t> shape = {static_cast<int64_t>(selfData.size())};
    std::vector<float> outHostData(selfData.size(), 0.0f);
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(selfData, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        std::vector<float> actual(selfData.size(), 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);
        std::vector<float> expected(selfData.size(), 0.0f);
        for (size_t i = 0; i < selfData.size(); ++i) {
            expected[i] = static_cast<float>(std::pow(2.0, static_cast<double>(selfData[i])));
        }
        ok = ok && CompareFloat(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunInplaceExp2Float(aclrtStream stream, const std::string& name, const std::vector<float>& selfData) {
    std::vector<int64_t> shape = {static_cast<int64_t>(selfData.size())};
    void* selfDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(selfData, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnInplaceExp2GetWorkspaceSize(self, &workspaceSize, &executor);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok && workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        ret = aclrtSynchronizeStream(stream);
        ok = ok && (ret == ACL_SUCCESS);
    }
    if (ok) {
        std::vector<float> actual(selfData.size(), 0.0f);
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), selfDeviceAddr,
                                            actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && (ret == ACL_SUCCESS);
        std::vector<float> expected(selfData.size(), 0.0f);
        for (size_t i = 0; i < selfData.size(); ++i) {
            expected[i] = static_cast<float>(std::pow(2.0, static_cast<double>(selfData[i])));
        }
        ok = ok && CompareFloat(actual, expected);
    }

    ReportCase(name, ok);

    if (self != nullptr) aclDestroyTensor(self);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return ok;
}

bool RunPowTensorScalarShapeError(aclrtStream stream, const std::string& name) {
    (void)stream;
    std::vector<float> selfHost = {1.0f, 2.0f};
    std::vector<float> outHost = {0.0f, 0.0f, 0.0f};
    std::vector<int64_t> selfShape = {2};
    std::vector<int64_t> outShape = {3};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* exponent = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(selfHost, selfShape, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHost, outShape, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);
    float exp = 2.0f;
    exponent = aclCreateScalar(&exp, ACL_FLOAT);
    ok = ok && (exponent != nullptr);

    if (ok) {
        ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = (ret != ACL_SUCCESS);
    }

    ReportCase(name, ok);
    if (self != nullptr) aclDestroyTensor(self);
    if (out != nullptr) aclDestroyTensor(out);
    if (exponent != nullptr) aclDestroyScalar(exponent);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}

bool RunPowTensorTensorShapeError(aclrtStream stream, const std::string& name) {
    (void)stream;
    std::vector<float> selfHost = {1.0f, 2.0f};
    std::vector<float> expHost = {2.0f, 3.0f};
    std::vector<float> outHost = {0.0f};
    std::vector<int64_t> shape2 = {2};
    std::vector<int64_t> shape1 = {1};
    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    auto ret = CreateAclTensor(selfHost, shape2, &selfDeviceAddr, ACL_FLOAT, &self);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(expHost, shape2, &expDeviceAddr, ACL_FLOAT, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHost, shape1, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = (ret != ACL_SUCCESS);
    }

    ReportCase(name, ok);
    if (self != nullptr) aclDestroyTensor(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}

bool RunPowScalarTensorShapeError(aclrtStream stream, const std::string& name) {
    (void)stream;
    std::vector<float> expHost = {1.0f, 2.0f};
    std::vector<float> outHost = {0.0f, 0.0f, 0.0f};
    std::vector<int64_t> expShape = {2};
    std::vector<int64_t> outShape = {3};
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;
    aclScalar* self = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool ok = true;

    float base = 2.0f;
    self = aclCreateScalar(&base, ACL_FLOAT);
    ok = ok && (self != nullptr);
    auto ret = CreateAclTensor(expHost, expShape, &expDeviceAddr, ACL_FLOAT, &exponent);
    ok = ok && (ret == ACL_SUCCESS);
    ret = CreateAclTensor(outHost, outShape, &outDeviceAddr, ACL_FLOAT, &out);
    ok = ok && (ret == ACL_SUCCESS);

    if (ok) {
        ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
        ok = (ret != ACL_SUCCESS);
    }

    ReportCase(name, ok);
    if (self != nullptr) aclDestroyScalar(self);
    if (exponent != nullptr) aclDestroyTensor(exponent);
    if (out != nullptr) aclDestroyTensor(out);
    if (expDeviceAddr != nullptr) aclrtFree(expDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}

bool RunNullParamChecks(const std::string& name) {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    bool ok = true;

    ok = ok && (aclnnPowTensorScalarGetWorkspaceSize(nullptr, nullptr, nullptr, &ws, &exec) != ACL_SUCCESS);
    ok = ok && (aclnnPowScalarTensorGetWorkspaceSize(nullptr, nullptr, nullptr, &ws, &exec) != ACL_SUCCESS);
    ok = ok && (aclnnPowTensorTensorGetWorkspaceSize(nullptr, nullptr, nullptr, &ws, &exec) != ACL_SUCCESS);
    ok = ok && (aclnnInplacePowTensorScalarGetWorkspaceSize(nullptr, nullptr, &ws, &exec) != ACL_SUCCESS);
    ok = ok && (aclnnInplacePowTensorTensorGetWorkspaceSize(nullptr, nullptr, &ws, &exec) != ACL_SUCCESS);
    ok = ok && (aclnnExp2GetWorkspaceSize(nullptr, nullptr, &ws, &exec) != ACL_SUCCESS);
    ok = ok && (aclnnInplaceExp2GetWorkspaceSize(nullptr, &ws, &exec) != ACL_SUCCESS);

    ReportCase(name, ok);
    return ok;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Null-pointer parameter checks for all 7 APIs.
    RunNullParamChecks("NullParamChecks");

    // TensorScalar / InplaceTensorScalar, including special exponents.
    RunPowTensorScalarFloat(stream, "TensorScalar_exp_0", {0.0f, 2.0f, 4.0f}, 0.0f);
    RunPowTensorScalarFloat(stream, "TensorScalar_exp_1", {1.0f, 2.0f, 3.0f}, 1.0f);
    RunPowTensorScalarFloat(stream, "TensorScalar_exp_0_5", {1.0f, 4.0f, 9.0f}, 0.5f);
    RunPowTensorScalarFloat(stream, "TensorScalar_exp_2", {1.0f, 2.0f, 3.0f}, 2.0f);
    RunPowTensorScalarFloat(stream, "TensorScalar_exp_3", {1.0f, 2.0f, 3.0f}, 3.0f);
    RunPowTensorScalarFloat(stream, "TensorScalar_exp_neg1", {1.0f, 2.0f, 4.0f}, -1.0f);

    RunInplacePowTensorScalarFloat(stream, "InplaceTensorScalar_exp_2", {1.0f, 2.0f, 3.0f}, 2.0f);
    RunInplacePowTensorScalarFloat(stream, "InplaceTensorScalar_exp_3", {1.0f, 2.0f, 3.0f}, 3.0f);

    // ScalarTensor.
    RunPowScalarTensorFloat(stream, "ScalarTensor_base_2", 2.0f, {0.0f, 1.0f, 2.0f, 3.0f});
    RunPowScalarTensorFloat(stream, "ScalarTensor_base_1", 1.0f, {1.0f, 2.0f, 3.0f});
    RunPowScalarTensorFloat(stream, "ScalarTensor_base_0", 0.0f, {1.0f, 2.0f, 3.0f});

    // TensorTensor: same-shape + broadcast.
    RunPowTensorTensorFloat("TensorTensor_float", stream, {1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 3.0f, 2.0f});
    RunPowTensorTensorFloat("TensorTensor_zero_pow_zero", stream, {0.0f, 0.0f}, {0.0f, 0.0f});
    RunPowTensorTensorBroadcastFloat(stream, "TensorTensor_broadcast_2x1_1x2");
    RunInplacePowTensorTensorFloat(stream, "InplaceTensorTensor_float", {2.0f, 3.0f, 4.0f}, {2.0f, 3.0f, 2.0f});

    // Exp2 + InplaceExp2.
    RunExp2Float(stream, "Exp2_float", {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    RunInplaceExp2Float(stream, "InplaceExp2_float", {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});

    // Type matrix for tiling OP_KEYs and pow.cpp AiCore/AiCpu dispatch.
    RunPowTensorTensorFP16(stream, "TensorTensor_fp16", {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 2.0f});
    RunPowTensorTensorBF16(stream, "TensorTensor_bf16", {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 2.0f});
    RunPowTensorTensorSameShape<int32_t>("TensorTensor_int32", stream, {1, 2, 3}, {1, 2, 2}, ACL_INT32);
    RunPowTensorTensorSameShape<int16_t>("TensorTensor_int16", stream, {1, 2, 3}, {1, 2, 2}, ACL_INT16);
    RunPowTensorTensorSameShape<int8_t>("TensorTensor_int8", stream, {1, 2, 3}, {1, 2, 2}, ACL_INT8);
    RunPowTensorTensorSameShape<uint8_t>("TensorTensor_uint8", stream, {1, 2, 3}, {1, 2, 2}, ACL_UINT8);

    // Double path tends to route AiCpu in pow.cpp (dtype not in AiCore support list).
    RunPowTensorTensorDouble("TensorTensor_double", stream, {2.0, 3.0}, {2.0, 2.0});

    // Parameter error branches.
    RunPowTensorScalarShapeError(stream, "TensorScalar_shape_error");
    RunPowScalarTensorShapeError(stream, "ScalarTensor_shape_error");
    RunPowTensorTensorShapeError(stream, "TensorTensor_shape_error");

    LOG_PRINT("\nTest Summary: %d Passed, %d Failed\n", pass_count, fail_count);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (fail_count > 0) ? -1 : 0;
}
