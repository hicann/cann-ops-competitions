#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, return_expr) \
    do {                              \
        if (!(cond)) {                \
            return_expr;              \
        }                             \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));

    auto ret = aclrtMalloc(deviceAddr, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    if (size > 0) {
        ret = aclrtMemcpy(*deviceAddr, static_cast<size_t>(size), hostData.data(), static_cast<size_t>(size),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);

    return ACL_SUCCESS;
}

template <typename T>
void DestroyTensorPair(void* dev, aclTensor* tensor)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (dev != nullptr) {
        aclrtFree(dev);
    }
}

bool AlmostEqual(double expected, double actual, double atol, double rtol)
{
    if (std::isnan(expected) && std::isnan(actual)) {
        return true;
    }
    if (std::isinf(expected) && std::isinf(actual)) {
        return (expected > 0) == (actual > 0);
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

float Fp16BitsToFloat(uint16_t bits)
{
    const uint16_t sign = (bits >> 15) & 0x1;
    const uint16_t exp = (bits >> 10) & 0x1F;
    const uint16_t frac = bits & 0x3FF;

    if (exp == 0) {
        if (frac == 0) {
            return sign ? -0.0f : 0.0f;
        }
        const float m = static_cast<float>(frac) / 1024.0f;
        const float v = std::ldexp(m, -14);
        return sign ? -v : v;
    }

    if (exp == 31) {
        if (frac == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    const float m = 1.0f + static_cast<float>(frac) / 1024.0f;
    const float v = std::ldexp(m, static_cast<int>(exp) - 15);
    return sign ? -v : v;
}

float Bf16BitsToFloat(uint16_t bits)
{
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = static_cast<uint32_t>(bits) << 16;
    return cvt.f;
}

int RunAddFloatCase(const char* name, const std::vector<float>& selfData, const std::vector<float>& otherData,
                    const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                    const std::vector<int64_t>& outShape, float alphaValue, aclrtStream stream)
{
    int64_t outN = GetShapeSize(outShape);

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(selfData, selfShape, &selfDev, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(selfDev, self); return 1);

    std::vector<float> outHost(static_cast<size_t>(outN), 0.0f);
    ret = CreateAclTensor(outHost, outShape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);

    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr,
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other);
              DestroyTensorPair<float>(outDev, out); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other);
              DestroyTensorPair<float>(outDev, out); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other);
                  DestroyTensorPair<float>(outDev, out); return 1);
    }

    ret = aclnnAdd(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other);
              DestroyTensorPair<float>(outDev, out); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other);
              DestroyTensorPair<float>(outDev, out); return 1);

    if (outN > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(outN * sizeof(float)), outDev,
                          static_cast<size_t>(outN * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other);
                  DestroyTensorPair<float>(outDev, out); return 1);
    }

    int failed = 0;
    if (selfShape == otherShape && selfShape == outShape) {
        for (int64_t i = 0; i < outN; ++i) {
            double expected = static_cast<double>(selfData[static_cast<size_t>(i)]) +
                              static_cast<double>(alphaValue) * static_cast<double>(otherData[static_cast<size_t>(i)]);
            if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
                ++failed;
            }
        }
    } else if (selfShape.size() == 2 && otherShape.size() == 1 && selfShape[1] == otherShape[0] && outShape == selfShape) {
        for (int64_t i = 0; i < selfShape[0]; ++i) {
            for (int64_t j = 0; j < selfShape[1]; ++j) {
                double expected = static_cast<double>(selfData[static_cast<size_t>(i * selfShape[1] + j)]) +
                                  static_cast<double>(alphaValue) * static_cast<double>(otherData[static_cast<size_t>(j)]);
                if (!AlmostEqual(expected, outHost[static_cast<size_t>(i * selfShape[1] + j)], 1e-5, 1e-5)) {
                    ++failed;
                }
            }
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(alpha);
    DestroyTensorPair<float>(selfDev, self);
    DestroyTensorPair<float>(otherDev, other);
    DestroyTensorPair<float>(outDev, out);

    return failed == 0 ? 0 : 1;
}

template <typename T>
int RunAddIntegralCase(const char* name, const std::vector<T>& selfData, const std::vector<T>& otherData,
                       const std::vector<int64_t>& shape, aclDataType dtype, int32_t alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfDev, dtype, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(otherData, shape, &otherDev, dtype, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<T>(selfDev, self); return 1);

    std::vector<T> outHost(static_cast<size_t>(n), static_cast<T>(0));
    ret = CreateAclTensor(outHost, shape, &outDev, dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<T>(selfDev, self); DestroyTensorPair<T>(otherDev, other); return 1);

    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
    CHECK_RET(alpha != nullptr,
              DestroyTensorPair<T>(selfDev, self); DestroyTensorPair<T>(otherDev, other);
              DestroyTensorPair<T>(outDev, out); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(alpha);
              DestroyTensorPair<T>(selfDev, self); DestroyTensorPair<T>(otherDev, other); DestroyTensorPair<T>(outDev, out);
              return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<T>(selfDev, self); DestroyTensorPair<T>(otherDev, other); DestroyTensorPair<T>(outDev, out);
                  return 1);
    }

    ret = aclnnAdd(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<T>(selfDev, self); DestroyTensorPair<T>(otherDev, other); DestroyTensorPair<T>(outDev, out);
              return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<T>(selfDev, self); DestroyTensorPair<T>(otherDev, other); DestroyTensorPair<T>(outDev, out);
              return 1);

    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(T)), outDev, static_cast<size_t>(n * sizeof(T)),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<T>(selfDev, self); DestroyTensorPair<T>(otherDev, other); DestroyTensorPair<T>(outDev, out);
                  return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        long long expected = static_cast<long long>(selfData[static_cast<size_t>(i)]) +
                             static_cast<long long>(alphaValue) * static_cast<long long>(otherData[static_cast<size_t>(i)]);
        if (static_cast<long long>(outHost[static_cast<size_t>(i)]) != expected) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(alpha);
    DestroyTensorPair<T>(selfDev, self);
    DestroyTensorPair<T>(otherDev, other);
    DestroyTensorPair<T>(outDev, out);

    return failed == 0 ? 0 : 1;
}

int RunAddRaw16Case(const char* name, const std::vector<uint16_t>& selfData, const std::vector<uint16_t>& otherData,
                    const std::vector<uint16_t>& expected, const std::vector<int64_t>& shape, aclDataType dtype,
                    float alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfDev, dtype, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(otherData, shape, &otherDev, dtype, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<uint16_t>(selfDev, self); return 1);

    std::vector<uint16_t> outHost(static_cast<size_t>(n), 0);
    ret = CreateAclTensor(outHost, shape, &outDev, dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<uint16_t>(otherDev, other); return 1);

    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr,
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<uint16_t>(otherDev, other);
              DestroyTensorPair<uint16_t>(outDev, out); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(alpha);
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<uint16_t>(otherDev, other); DestroyTensorPair<uint16_t>(outDev, out);
              return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<uint16_t>(otherDev, other); DestroyTensorPair<uint16_t>(outDev, out);
                  return 1);
    }

    ret = aclnnAdd(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<uint16_t>(otherDev, other); DestroyTensorPair<uint16_t>(outDev, out);
              return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<uint16_t>(otherDev, other); DestroyTensorPair<uint16_t>(outDev, out);
              return 1);

    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(uint16_t)), outDev,
                          static_cast<size_t>(n * sizeof(uint16_t)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<uint16_t>(otherDev, other); DestroyTensorPair<uint16_t>(outDev, out);
                  return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (outHost[static_cast<size_t>(i)] != expected[static_cast<size_t>(i)]) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(alpha);
    DestroyTensorPair<uint16_t>(selfDev, self);
    DestroyTensorPair<uint16_t>(otherDev, other);
    DestroyTensorPair<uint16_t>(outDev, out);

    return failed == 0 ? 0 : 1;
}

int RunAddMix16FloatCase(const char* name, const std::vector<uint16_t>& self16, const std::vector<float>& otherFloat,
                         const std::vector<int64_t>& shape, aclDataType dtype16, bool isFp16,
                         float alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(self16, shape, &selfDev, dtype16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(otherFloat, shape, &otherDev, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<uint16_t>(selfDev, self); return 1);

    std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);

    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr,
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
              return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(alpha);
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
              return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
                  return 1);
    }

    ret = aclnnAdd(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
              return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
              return 1);

    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), outDev,
                          static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<uint16_t>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
                  return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        float a = isFp16 ? Fp16BitsToFloat(self16[static_cast<size_t>(i)]) : Bf16BitsToFloat(self16[static_cast<size_t>(i)]);
        double expected = static_cast<double>(a) + static_cast<double>(alphaValue) *
                          static_cast<double>(otherFloat[static_cast<size_t>(i)]);
        if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-3, 1e-3)) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(alpha);
    DestroyTensorPair<uint16_t>(selfDev, self);
    DestroyTensorPair<float>(otherDev, other);
    DestroyTensorPair<float>(outDev, out);

    return failed == 0 ? 0 : 1;
}

int RunAddsFloatCase(const char* name, const std::vector<float>& selfData, const std::vector<int64_t>& shape,
                     float otherScalar, float alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);

    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(selfDev, self); return 1);

    aclScalar* other = aclCreateScalar(&otherScalar, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr,
              if (other != nullptr) { aclDestroyScalar(other); }
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(outDev, out); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddsGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(other); aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(outDev, out); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(other); aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(outDev, out); return 1);
    }

    ret = aclnnAdds(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(other); aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(outDev, out); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(other); aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(outDev, out); return 1);

    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), outDev,
                          static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(other); aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(outDev, out); return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        double expected = static_cast<double>(selfData[static_cast<size_t>(i)]) +
                          static_cast<double>(alphaValue) * static_cast<double>(otherScalar);
        if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
    DestroyTensorPair<float>(selfDev, self);
    DestroyTensorPair<float>(outDev, out);

    return failed == 0 ? 0 : 1;
}

int RunAddsBoolSpecialCase(const char* name, aclrtStream stream)
{
    std::vector<uint8_t> selfData = {1, 0, 1, 0};
    std::vector<int64_t> shape = {2, 2};

    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfDev, ACL_BOOL, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    std::vector<int32_t> outHost(4, 0);
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<uint8_t>(selfDev, self); return 1);

    bool otherValue = true;
    bool alphaValue = true;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_BOOL);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_BOOL);

    CHECK_RET(other != nullptr && alpha != nullptr,
              if (other != nullptr) { aclDestroyScalar(other); }
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<uint8_t>(selfDev, self); DestroyTensorPair<int32_t>(outDev, out); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddsGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(other); aclDestroyScalar(alpha);
              DestroyTensorPair<uint8_t>(selfDev, self); DestroyTensorPair<int32_t>(outDev, out); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(other); aclDestroyScalar(alpha);
                  DestroyTensorPair<uint8_t>(selfDev, self); DestroyTensorPair<int32_t>(outDev, out); return 1);
    }

    ret = aclnnAdds(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(other); aclDestroyScalar(alpha);
              DestroyTensorPair<uint8_t>(selfDev, self); DestroyTensorPair<int32_t>(outDev, out); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(other); aclDestroyScalar(alpha);
              DestroyTensorPair<uint8_t>(selfDev, self); DestroyTensorPair<int32_t>(outDev, out); return 1);

    ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(4 * sizeof(int32_t)), outDev, static_cast<size_t>(4 * sizeof(int32_t)),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(other); aclDestroyScalar(alpha);
              DestroyTensorPair<uint8_t>(selfDev, self); DestroyTensorPair<int32_t>(outDev, out); return 1);

    int failed = 0;
    for (size_t i = 0; i < outHost.size(); ++i) {
        int expected = selfData[i] ? 1 : 1;  // bool + true 经特殊路径防止出现2，期望统一为1
        if (outHost[i] != expected) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
    DestroyTensorPair<uint8_t>(selfDev, self);
    DestroyTensorPair<int32_t>(outDev, out);

    return failed == 0 ? 0 : 1;
}

int RunInplaceAddFloatCase(const char* name, const std::vector<float>& selfData, const std::vector<float>& otherData,
                           const std::vector<int64_t>& shape, float alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);
    void *selfDev = nullptr, *otherDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(selfDev, self); return 1);

    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(alpha != nullptr,
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnInplaceAddGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);
    }

    ret = aclnnInplaceAdd(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(alpha);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);

    std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), selfDev,
                          static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        double expected = static_cast<double>(selfData[static_cast<size_t>(i)]) +
                          static_cast<double>(alphaValue) * static_cast<double>(otherData[static_cast<size_t>(i)]);
        if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(alpha);
    DestroyTensorPair<float>(selfDev, self);
    DestroyTensorPair<float>(otherDev, other);

    return failed == 0 ? 0 : 1;
}

int RunInplaceAddsFloatCase(const char* name, const std::vector<float>& selfData, const std::vector<int64_t>& shape,
                            float otherValue, float alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);
    void* selfDev = nullptr;
    aclTensor* self = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(other != nullptr && alpha != nullptr,
              if (other != nullptr) { aclDestroyScalar(other); }
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<float>(selfDev, self); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnInplaceAddsGetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(other); aclDestroyScalar(alpha); DestroyTensorPair<float>(selfDev, self); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(other); aclDestroyScalar(alpha); DestroyTensorPair<float>(selfDev, self); return 1);
    }

    ret = aclnnInplaceAdds(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(other); aclDestroyScalar(alpha); DestroyTensorPair<float>(selfDev, self); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(other); aclDestroyScalar(alpha); DestroyTensorPair<float>(selfDev, self); return 1);

    std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), selfDev,
                          static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(other); aclDestroyScalar(alpha); DestroyTensorPair<float>(selfDev, self); return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        double expected = static_cast<double>(selfData[static_cast<size_t>(i)]) +
                          static_cast<double>(alphaValue) * static_cast<double>(otherValue);
        if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
    DestroyTensorPair<float>(selfDev, self);

    return failed == 0 ? 0 : 1;
}

int RunAddV3FloatCase(const char* name, float selfScalar, const std::vector<float>& otherData,
                      const std::vector<int64_t>& shape, float alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);
    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(otherDev, other); return 1);

    aclScalar* self = aclCreateScalar(&selfScalar, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr,
              if (self != nullptr) { aclDestroyScalar(self); }
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddV3GetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(self); aclDestroyScalar(alpha);
              DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(self); aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out); return 1);
    }

    ret = aclnnAddV3(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(self); aclDestroyScalar(alpha);
              DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(self); aclDestroyScalar(alpha);
              DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out); return 1);

    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), outDev,
                          static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(self); aclDestroyScalar(alpha);
                  DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out); return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        double expected = static_cast<double>(selfScalar) + static_cast<double>(alphaValue) *
                          static_cast<double>(otherData[static_cast<size_t>(i)]);
        if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(self);
    aclDestroyScalar(alpha);
    DestroyTensorPair<float>(otherDev, other);
    DestroyTensorPair<float>(outDev, out);

    return failed == 0 ? 0 : 1;
}

int RunAddV3Int8FallbackCase(const char* name, int8_t selfScalar, const std::vector<int8_t>& otherData,
                             const std::vector<int64_t>& shape, int32_t alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);
    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    std::vector<int8_t> outHost(static_cast<size_t>(n), 0);
    ret = CreateAclTensor(outHost, shape, &outDev, ACL_INT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<int8_t>(otherDev, other); return 1);

    aclScalar* self = aclCreateScalar(&selfScalar, ACL_INT8);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
    CHECK_RET(self != nullptr && alpha != nullptr,
              if (self != nullptr) { aclDestroyScalar(self); }
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<int8_t>(otherDev, other); DestroyTensorPair<int8_t>(outDev, out); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnAddV3GetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(self); aclDestroyScalar(alpha);
              DestroyTensorPair<int8_t>(otherDev, other); DestroyTensorPair<int8_t>(outDev, out); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(self); aclDestroyScalar(alpha);
                  DestroyTensorPair<int8_t>(otherDev, other); DestroyTensorPair<int8_t>(outDev, out); return 1);
    }

    ret = aclnnAddV3(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(self); aclDestroyScalar(alpha);
              DestroyTensorPair<int8_t>(otherDev, other); DestroyTensorPair<int8_t>(outDev, out); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(self); aclDestroyScalar(alpha);
              DestroyTensorPair<int8_t>(otherDev, other); DestroyTensorPair<int8_t>(outDev, out); return 1);

    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(int8_t)), outDev,
                          static_cast<size_t>(n * sizeof(int8_t)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(self); aclDestroyScalar(alpha);
                  DestroyTensorPair<int8_t>(otherDev, other); DestroyTensorPair<int8_t>(outDev, out); return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        int expected = static_cast<int>(selfScalar) + alphaValue * static_cast<int>(otherData[static_cast<size_t>(i)]);
        if (static_cast<int>(outHost[static_cast<size_t>(i)]) != expected) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(self);
    aclDestroyScalar(alpha);
    DestroyTensorPair<int8_t>(otherDev, other);
    DestroyTensorPair<int8_t>(outDev, out);

    return failed == 0 ? 0 : 1;
}

int RunInplaceAddV3Case(const char* name, float selfScalar, const std::vector<float>& otherData,
                        const std::vector<int64_t>& shape, float alphaValue, aclrtStream stream)
{
    int64_t n = GetShapeSize(shape);
    void* otherDev = nullptr;
    aclTensor* other = nullptr;

    auto ret = CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, return 1);

    aclScalar* self = aclCreateScalar(&selfScalar, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    CHECK_RET(self != nullptr && alpha != nullptr,
              if (self != nullptr) { aclDestroyScalar(self); }
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<float>(otherDev, other); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[FAIL] %s: aclnnInplaceAddV3GetWorkspaceSize=%d\n", name, ret);
              aclDestroyScalar(self); aclDestroyScalar(alpha); DestroyTensorPair<float>(otherDev, other); return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  aclDestroyScalar(self); aclDestroyScalar(alpha); DestroyTensorPair<float>(otherDev, other); return 1);
    }

    ret = aclnnInplaceAddV3(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(self); aclDestroyScalar(alpha); DestroyTensorPair<float>(otherDev, other); return 1);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              if (workspace != nullptr) { aclrtFree(workspace); }
              aclDestroyScalar(self); aclDestroyScalar(alpha); DestroyTensorPair<float>(otherDev, other); return 1);

    std::vector<float> outHost(static_cast<size_t>(n), 0.0f);
    if (n > 0) {
        ret = aclrtMemcpy(outHost.data(), static_cast<size_t>(n * sizeof(float)), otherDev,
                          static_cast<size_t>(n * sizeof(float)), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  if (workspace != nullptr) { aclrtFree(workspace); }
                  aclDestroyScalar(self); aclDestroyScalar(alpha); DestroyTensorPair<float>(otherDev, other); return 1);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        double expected = static_cast<double>(selfScalar) +
                          static_cast<double>(alphaValue) * static_cast<double>(otherData[static_cast<size_t>(i)]);
        if (!AlmostEqual(expected, outHost[static_cast<size_t>(i)], 1e-5, 1e-5)) {
            ++failed;
        }
    }

    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyScalar(self);
    aclDestroyScalar(alpha);
    DestroyTensorPair<float>(otherDev, other);

    return failed == 0 ? 0 : 1;
}

int RunExpectedFailureCases(aclrtStream stream)
{
    (void)stream;
    int failed = 0;

    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape2x2 = {2, 2};
    std::vector<int64_t> shape1x4 = {1, 4};

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;

    auto ret = CreateAclTensor(base, shape2x2, &selfDev, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(base, shape2x2, &otherDev, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensorPair<float>(selfDev, self); return 1);
    ret = CreateAclTensor(base, shape2x2, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS,
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); return 1);

    float alphaV = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaV, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // NOTE:
    // In simulator mode, Add APIs may dereference inputs in DFX stage before null checks,
    // so nullptr negative tests can crash the process directly. We keep only safe invalid
    // parameter tests (shape/dtype) here.
    auto st = ACLNN_SUCCESS;

    void* badOutDev = nullptr;
    aclTensor* badOut = nullptr;
    std::vector<float> badOutData(4, 0.0f);
    ret = CreateAclTensor(badOutData, shape1x4, &badOutDev, ACL_FLOAT, &badOut);
    CHECK_RET(ret == ACL_SUCCESS,
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
              return 1);

    st = aclnnAddGetWorkspaceSize(self, other, alpha, badOut, &workspaceSize, &executor);
    if (st == ACL_SUCCESS) { ++failed; LOG_PRINT("[FAIL] add_bad_out_shape\n"); } else { LOG_PRINT("[PASS] add_bad_out_shape\n"); }

    st = aclnnInplaceAddGetWorkspaceSize(self, badOut, alpha, &workspaceSize, &executor);
    if (st == ACL_SUCCESS) { ++failed; LOG_PRINT("[FAIL] inplace_add_bad_shape\n"); } else { LOG_PRINT("[PASS] inplace_add_bad_shape\n"); }

    std::vector<uint16_t> u16 = {1, 2, 3, 4};
    void* u16Dev = nullptr;
    aclTensor* u16T = nullptr;
    ret = CreateAclTensor(u16, shape2x2, &u16Dev, ACL_UINT16, &u16T);
    CHECK_RET(ret == ACL_SUCCESS,
              if (alpha != nullptr) { aclDestroyScalar(alpha); }
              DestroyTensorPair<float>(badOutDev, badOut);
              DestroyTensorPair<float>(selfDev, self); DestroyTensorPair<float>(otherDev, other); DestroyTensorPair<float>(outDev, out);
              return 1);

    st = aclnnAddGetWorkspaceSize(u16T, u16T, alpha, u16T, &workspaceSize, &executor);
    if (st == ACL_SUCCESS) { ++failed; LOG_PRINT("[FAIL] add_unsupported_uint16\n"); } else { LOG_PRINT("[PASS] add_unsupported_uint16\n"); }

    float v3SelfVal = 1.0f;
    aclScalar* v3Self = aclCreateScalar(&v3SelfVal, ACL_FLOAT);
    st = aclnnAddV3GetWorkspaceSize(v3Self, other, alpha, badOut, &workspaceSize, &executor);
    if (st == ACL_SUCCESS) { ++failed; LOG_PRINT("[FAIL] addv3_bad_out_shape\n"); } else { LOG_PRINT("[PASS] addv3_bad_out_shape\n"); }

    if (v3Self != nullptr) {
        aclDestroyScalar(v3Self);
    }
    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensorPair<uint16_t>(u16Dev, u16T);
    DestroyTensorPair<float>(badOutDev, badOut);
    DestroyTensorPair<float>(selfDev, self);
    DestroyTensorPair<float>(otherDev, other);
    DestroyTensorPair<float>(outDev, out);

    return failed;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int totalFailed = 0;

    totalFailed += RunAddFloatCase("add_float_alpha1", {0, 1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 2, 2, 2, 3, 3},
                                   {4, 2}, {4, 2}, {4, 2}, 1.0f, stream);

    totalFailed += RunAddFloatCase("add_float_alpha0", {0, 1, 2, 3}, {1, 2, 3, 4}, {2, 2}, {2, 2}, {2, 2}, 0.0f, stream);

    totalFailed += RunAddFloatCase("add_float_alpha_neg", {1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2}, {2, 2}, {2, 2}, -1.25f, stream);

    totalFailed += RunAddFloatCase("add_float_broadcast", {1, 2, 3, 4, 5, 6}, {10, 20, 30},
                                   {2, 3}, {3}, {2, 3}, 0.5f, stream);

    totalFailed += RunAddFloatCase("add_float_special", {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                                           std::numeric_limits<float>::quiet_NaN(), 1.0f},
                                   {1.0f, 2.0f, 3.0f, std::numeric_limits<float>::infinity()},
                                   {2, 2}, {2, 2}, {2, 2}, 1.0f, stream);

    totalFailed += RunAddIntegralCase<int32_t>("add_int32", {-1, 0, 10, 100}, {2, 5, -1, 3}, {2, 2}, ACL_INT32, 2, stream);
    totalFailed += RunAddIntegralCase<int64_t>("add_int64", {-1, 2, 3, 4}, {6, 7, 8, 9}, {2, 2}, ACL_INT64, -1, stream);
    totalFailed += RunAddIntegralCase<int8_t>("add_int8", {-1, 2, 3, 4}, {1, 1, 1, 1}, {2, 2}, ACL_INT8, 1, stream);
    totalFailed += RunAddIntegralCase<uint8_t>("add_uint8", {1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2}, ACL_UINT8, 1, stream);

    totalFailed += RunAddRaw16Case("add_fp16", {15360, 16384, 16896, 17408}, {16384, 16384, 16384, 16384},
                                   {16896, 17408, 17664, 17920}, {2, 2}, ACL_FLOAT16, 0.5f, stream);

    totalFailed += RunAddRaw16Case("add_bf16", {16256, 16384, 16448, 16512}, {16384, 16384, 16384, 16384},
                                   {16448, 16512, 16544, 16576}, {2, 2}, ACL_BF16, 0.5f, stream);

    totalFailed += RunAddMix16FloatCase("add_mix_fp16_float", {15360, 16384, 16896, 17408}, {1, 2, 3, 4},
                                        {2, 2}, ACL_FLOAT16, true, 1.0f, stream);

    totalFailed += RunAddMix16FloatCase("add_mix_bf16_float", {16256, 16384, 16448, 16512}, {1, 2, 3, 4},
                                        {2, 2}, ACL_BF16, false, 1.0f, stream);

    totalFailed += RunAddsFloatCase("adds_float_alpha1", {1, 2, 3, 4}, {2, 2}, 5.0f, 1.0f, stream);
    totalFailed += RunAddsFloatCase("adds_float_alpha_non1", {1, 2, 3, 4}, {2, 2}, 3.0f, -0.75f, stream);
    totalFailed += RunAddsBoolSpecialCase("adds_bool_special", stream);

    totalFailed += RunInplaceAddFloatCase("inplace_add_float", {1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2}, 0.5f, stream);
    totalFailed += RunInplaceAddsFloatCase("inplace_adds_float", {1, 2, 3, 4}, {2, 2}, 9.0f, 0.5f, stream);

    totalFailed += RunAddV3FloatCase("addv3_float_alpha1", 1.5f, {1, 2, 3, 4}, {2, 2}, 1.0f, stream);
    totalFailed += RunAddV3FloatCase("addv3_float_axpy", 2.0f, {1, 2, 3, 4}, {2, 2}, 0.25f, stream);
    totalFailed += RunAddV3Int8FallbackCase("addv3_int8_fallback", 2, {1, 2, 3, 4}, {2, 2}, 2, stream);
    totalFailed += RunInplaceAddV3Case("inplace_addv3_float", 3.0f, {1, 2, 3, 4}, {2, 2}, 2.0f, stream);

    totalFailed += RunExpectedFailureCases(stream);

    LOG_PRINT("\n=== Summary: %d failed ===\n", totalFailed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return totalFailed;
}
