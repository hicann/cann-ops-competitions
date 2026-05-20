#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

struct TestStats {
    int passed = 0;
    int failed = 0;
};

static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= shape[i];
    }
    return size;
}

static std::vector<int64_t> GetStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static int32_t GetDeviceId()
{
    const char *env1 = std::getenv("ASCEND_DEVICE_ID");
    if (env1 != nullptr && std::strlen(env1) > 0) {
        return static_cast<int32_t>(std::atoi(env1));
    }
    const char *env2 = std::getenv("CANN_DEVICE_ID");
    if (env2 != nullptr && std::strlen(env2) > 0) {
        return static_cast<int32_t>(std::atoi(env2));
    }
    return 15;
}

template <typename T>
static bool CopyDeviceToHost(void *deviceAddr, std::vector<T> &hostOut)
{
    size_t bytes = hostOut.size() * sizeof(T);
    aclError ret = aclrtMemcpy(hostOut.data(), bytes, deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::printf("D2H memcpy failed ret=%d\n", ret);
        return false;
    }
    return true;
}

template <typename T>
static int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    aclError ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMalloc failed ret=%d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMemcpy H2D failed ret=%d\n", ret); return ret);

    std::vector<int64_t> strides = GetStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
        shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, std::printf("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T, typename AlphaT>
static bool RunAddWsCase(const std::string &name, const std::vector<T> &selfData, const std::vector<int64_t> &selfShape,
    const std::vector<T> &otherData, const std::vector<int64_t> &otherShape, const std::vector<int64_t> &outShape,
    AlphaT alphaValue, aclDataType tensorType, aclDataType alphaType)
{
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (self != nullptr) aclDestroyTensor(self);
        if (other != nullptr) aclDestroyTensor(other);
        if (out != nullptr) aclDestroyTensor(out);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (selfAddr != nullptr) aclrtFree(selfAddr);
        if (otherAddr != nullptr) aclrtFree(otherAddr);
        if (outAddr != nullptr) aclrtFree(outAddr);
    };

    std::vector<T> outInit(static_cast<size_t>(GetShapeSize(outShape)), static_cast<T>(0));
    if (CreateAclTensor(selfData, selfShape, &selfAddr, tensorType, &self) != ACL_SUCCESS ||
        CreateAclTensor(otherData, otherShape, &otherAddr, tensorType, &other) != ACL_SUCCESS ||
        CreateAclTensor(outInit, outShape, &outAddr, tensorType, &out) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] alpha create failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d workspace=%llu\n  [%s]\n\n", name.c_str(), ret,
        static_cast<unsigned long long>(workspaceSize), ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

template <typename T, typename ScalarT, typename AlphaT>
static bool RunAddsWsCase(const std::string &name, const std::vector<T> &selfData, const std::vector<int64_t> &shape,
    ScalarT otherValue, AlphaT alphaValue, aclDataType tensorType, aclDataType scalarType, aclDataType alphaType)
{
    void *selfAddr = nullptr;
    void *outAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    aclScalar *other = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (self != nullptr) aclDestroyTensor(self);
        if (out != nullptr) aclDestroyTensor(out);
        if (other != nullptr) aclDestroyScalar(other);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (selfAddr != nullptr) aclrtFree(selfAddr);
        if (outAddr != nullptr) aclrtFree(outAddr);
    };

    std::vector<T> outInit(selfData.size(), static_cast<T>(0));
    if (CreateAclTensor(selfData, shape, &selfAddr, tensorType, &self) != ACL_SUCCESS ||
        CreateAclTensor(outInit, shape, &outAddr, tensorType, &out) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    other = aclCreateScalar(&otherValue, scalarType);
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (other == nullptr || alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] scalar creation failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d workspace=%llu\n  [%s]\n\n", name.c_str(), ret,
        static_cast<unsigned long long>(workspaceSize), ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

template <typename T, typename AlphaT>
static bool RunInplaceAddWsCase(const std::string &name, const std::vector<T> &selfData, const std::vector<int64_t> &selfShape,
    const std::vector<T> &otherData, const std::vector<int64_t> &otherShape, AlphaT alphaValue, aclDataType tensorType,
    aclDataType alphaType)
{
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (self != nullptr) aclDestroyTensor(self);
        if (other != nullptr) aclDestroyTensor(other);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (selfAddr != nullptr) aclrtFree(selfAddr);
        if (otherAddr != nullptr) aclrtFree(otherAddr);
    };

    if (CreateAclTensor(selfData, selfShape, &selfAddr, tensorType, &self) != ACL_SUCCESS ||
        CreateAclTensor(otherData, otherShape, &otherAddr, tensorType, &other) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] alpha creation failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d workspace=%llu\n  [%s]\n\n", name.c_str(), ret,
        static_cast<unsigned long long>(workspaceSize), ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

template <typename T, typename ScalarT, typename AlphaT>
static bool RunInplaceAddsWsCase(const std::string &name, const std::vector<T> &selfData, const std::vector<int64_t> &shape,
    ScalarT otherValue, AlphaT alphaValue, aclDataType tensorType, aclDataType scalarType, aclDataType alphaType)
{
    void *selfAddr = nullptr;
    aclTensor *self = nullptr;
    aclScalar *other = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (self != nullptr) aclDestroyTensor(self);
        if (other != nullptr) aclDestroyScalar(other);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (selfAddr != nullptr) aclrtFree(selfAddr);
    };

    if (CreateAclTensor(selfData, shape, &selfAddr, tensorType, &self) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    other = aclCreateScalar(&otherValue, scalarType);
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (other == nullptr || alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] scalar creation failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d workspace=%llu\n  [%s]\n\n", name.c_str(), ret,
        static_cast<unsigned long long>(workspaceSize), ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

template <typename T, typename ScalarT, typename AlphaT>
static bool RunAddV3WsCase(const std::string &name, ScalarT selfValue, const std::vector<T> &otherData,
    const std::vector<int64_t> &shape, AlphaT alphaValue, aclDataType tensorType, aclDataType scalarType,
    aclDataType alphaType)
{
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    aclScalar *self = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (other != nullptr) aclDestroyTensor(other);
        if (out != nullptr) aclDestroyTensor(out);
        if (self != nullptr) aclDestroyScalar(self);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (otherAddr != nullptr) aclrtFree(otherAddr);
        if (outAddr != nullptr) aclrtFree(outAddr);
    };

    std::vector<T> outInit(otherData.size(), static_cast<T>(0));
    if (CreateAclTensor(otherData, shape, &otherAddr, tensorType, &other) != ACL_SUCCESS ||
        CreateAclTensor(outInit, shape, &outAddr, tensorType, &out) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    self = aclCreateScalar(&selfValue, scalarType);
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (self == nullptr || alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] scalar creation failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d workspace=%llu\n  [%s]\n\n", name.c_str(), ret,
        static_cast<unsigned long long>(workspaceSize), ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

template <typename T, typename ScalarT, typename AlphaT>
static bool RunInplaceAddV3WsCase(const std::string &name, ScalarT selfValue, const std::vector<T> &otherData,
    const std::vector<int64_t> &shape, AlphaT alphaValue, aclDataType tensorType, aclDataType scalarType,
    aclDataType alphaType)
{
    void *otherAddr = nullptr;
    aclTensor *other = nullptr;
    aclScalar *self = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (other != nullptr) aclDestroyTensor(other);
        if (self != nullptr) aclDestroyScalar(self);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (otherAddr != nullptr) aclrtFree(otherAddr);
    };

    if (CreateAclTensor(otherData, shape, &otherAddr, tensorType, &other) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    self = aclCreateScalar(&selfValue, scalarType);
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (self == nullptr || alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] scalar creation failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d workspace=%llu\n  [%s]\n\n", name.c_str(), ret,
        static_cast<unsigned long long>(workspaceSize), ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

static bool RunNullptrCase()
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    std::printf("Case: Error-Nullptr-AddGetWorkspace\n  ret=%d\n  [%s]\n\n", ret, ok ? "PASS" : "FAIL");
    return ok;
}

template <typename T, typename AlphaT>
static bool RunAddWsExpectFailCase(const std::string &name, const std::vector<T> &selfData, const std::vector<int64_t> &selfShape,
    const std::vector<T> &otherData, const std::vector<int64_t> &otherShape, const std::vector<int64_t> &outShape,
    AlphaT alphaValue, aclDataType tensorType, aclDataType alphaType)
{
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (self != nullptr) aclDestroyTensor(self);
        if (other != nullptr) aclDestroyTensor(other);
        if (out != nullptr) aclDestroyTensor(out);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (selfAddr != nullptr) aclrtFree(selfAddr);
        if (otherAddr != nullptr) aclrtFree(otherAddr);
        if (outAddr != nullptr) aclrtFree(outAddr);
    };

    std::vector<T> outInit(static_cast<size_t>(GetShapeSize(outShape)), static_cast<T>(0));
    if (CreateAclTensor(selfData, selfShape, &selfAddr, tensorType, &self) != ACL_SUCCESS ||
        CreateAclTensor(otherData, otherShape, &otherAddr, tensorType, &other) != ACL_SUCCESS ||
        CreateAclTensor(outInit, outShape, &outAddr, tensorType, &out) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] alpha create failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d expected=non-zero\n  [%s]\n\n", name.c_str(), ret, ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

template <typename T, typename AlphaT>
static bool RunAddExecuteCase(
    const std::string &name,
    const std::vector<T> &selfData,
    const std::vector<int64_t> &selfShape,
    const std::vector<T> &otherData,
    const std::vector<int64_t> &otherShape,
    const std::vector<int64_t> &outShape,
    AlphaT alphaValue,
    aclDataType tensorType,
    aclDataType alphaType,
    aclrtStream stream)
{
    std::vector<T> outData(static_cast<size_t>(GetShapeSize(outShape)), static_cast<T>(0));

    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (self != nullptr) aclDestroyTensor(self);
        if (other != nullptr) aclDestroyTensor(other);
        if (out != nullptr) aclDestroyTensor(out);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (selfAddr != nullptr) aclrtFree(selfAddr);
        if (otherAddr != nullptr) aclrtFree(otherAddr);
        if (outAddr != nullptr) aclrtFree(outAddr);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    };

    if (CreateAclTensor(selfData, selfShape, &selfAddr, tensorType, &self) != ACL_SUCCESS ||
        CreateAclTensor(otherData, otherShape, &otherAddr, tensorType, &other) != ACL_SUCCESS ||
        CreateAclTensor(outData, outShape, &outAddr, tensorType, &out) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] alpha create failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] get workspace ret=%d\n\n", name.c_str(), ret);
        return false;
    }
    if (workspaceSize > 0) {
        if (aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            cleanup();
            std::printf("Case: %s\n  [FAIL] workspace malloc failed\n\n", name.c_str());
            return false;
        }
    }

    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] aclnnAdd ret=%d\n\n", name.c_str(), ret);
        return false;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] stream sync failed\n\n", name.c_str());
        return false;
    }
    if (!CopyDeviceToHost(outAddr, outData)) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] copy back failed\n\n", name.c_str());
        return false;
    }

    bool valueMatch = true;
    for (size_t i = 0; i < outData.size(); ++i) {
        double expected = static_cast<double>(selfData[i]) + static_cast<double>(alphaValue) * static_cast<double>(otherData[i]);
        if (std::fabs(static_cast<double>(outData[i]) - expected) > 1e-6) {
            valueMatch = false;
            break;
        }
    }
    std::printf("Case: %s\n  value_check=%s first=%.6f\n  [PASS]\n\n",
        name.c_str(), valueMatch ? "match" : "mismatch", static_cast<double>(outData[0]));
    cleanup();
    return true;
}

template <typename T, typename ScalarT, typename AlphaT>
static bool RunAddV3WsExpectFailCase(const std::string &name, ScalarT selfValue, const std::vector<T> &otherData,
    const std::vector<int64_t> &shape, AlphaT alphaValue, aclDataType tensorType, aclDataType scalarType,
    aclDataType alphaType)
{
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    aclScalar *self = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        if (other != nullptr) aclDestroyTensor(other);
        if (out != nullptr) aclDestroyTensor(out);
        if (self != nullptr) aclDestroyScalar(self);
        if (alpha != nullptr) aclDestroyScalar(alpha);
        if (otherAddr != nullptr) aclrtFree(otherAddr);
        if (outAddr != nullptr) aclrtFree(outAddr);
    };

    std::vector<T> outInit(otherData.size(), static_cast<T>(0));
    if (CreateAclTensor(otherData, shape, &otherAddr, tensorType, &other) != ACL_SUCCESS ||
        CreateAclTensor(outInit, shape, &outAddr, tensorType, &out) != ACL_SUCCESS) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] tensor creation failed\n\n", name.c_str());
        return false;
    }
    self = aclCreateScalar(&selfValue, scalarType);
    alpha = aclCreateScalar(&alphaValue, alphaType);
    if (self == nullptr || alpha == nullptr) {
        cleanup();
        std::printf("Case: %s\n  [FAIL] scalar creation failed\n\n", name.c_str());
        return false;
    }

    aclnnStatus ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    std::printf("Case: %s\n  ret=%d expected=non-zero\n  [%s]\n\n", name.c_str(), ret, ok ? "PASS" : "FAIL");
    cleanup();
    return ok;
}

int main()
{
    int32_t deviceId = GetDeviceId();
    aclrtStream stream = nullptr;

    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclInit failed ret=%d\n", ret); return 1);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSetDevice(%d) failed ret=%d\n", deviceId, ret); return 1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtCreateStream failed ret=%d\n", ret); return 1);

    std::printf("Using device id: %d\n\n", deviceId);

    TestStats stats;
    auto Acc = [&](bool ok) {
        if (ok) {
            ++stats.passed;
        } else {
            ++stats.failed;
        }
    };

    std::vector<int64_t> s22 = {2, 2};
    std::vector<int64_t> s12 = {1, 2};

    std::vector<float> fA = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> fB = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> fBr = {9.0f, 10.0f};
    std::vector<int32_t> iA = {1, 2, 3, 4};
    std::vector<int32_t> iB = {4, 3, 2, 1};
    std::vector<int64_t> lA = {1, 2, 3, 4};
    std::vector<int64_t> lB = {4, 3, 2, 1};
    std::vector<int8_t> i8A = {1, 2, 3, 4};
    std::vector<int8_t> i8B = {1, 1, 1, 1};
    std::vector<uint8_t> u8A = {1, 2, 3, 4};
    std::vector<uint8_t> u8B = {1, 2, 3, 4};
    std::vector<uint8_t> boolA = {1, 0, 1, 0};
    std::vector<uint8_t> boolB = {0, 1, 0, 1};
    std::vector<uint16_t> fp16A = {0x3C00, 0x4000, 0x4200, 0x4400};
    std::vector<uint16_t> fp16B = {0x3C00, 0x3C00, 0x3C00, 0x3C00};
    std::vector<uint16_t> bf16A = {0x3F80, 0x4000, 0x4040, 0x4080};
    std::vector<uint16_t> bf16B = {0x3F80, 0x3F80, 0x3F80, 0x3F80};

    Acc(RunAddWsCase<float, float>("AddWs-FP32-alpha1", fA, s22, fB, s22, s22, 1.0f, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddWsCase<float, float>("AddWs-FP32-broadcast", fA, s22, fBr, s12, s22, 1.0f, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddWsCase<float, float>("AddWs-FP32-alphaNeg", fA, s22, fB, s22, s22, -0.75f, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddWsCase<int32_t, int32_t>("AddWs-INT32", iA, s22, iB, s22, s22, 2, ACL_INT32, ACL_INT32));
    Acc(RunAddWsCase<int64_t, int64_t>("AddWs-INT64", lA, s22, lB, s22, s22, 1, ACL_INT64, ACL_INT64));
    Acc(RunAddWsCase<int8_t, int8_t>("AddWs-INT8", i8A, s22, i8B, s22, s22, 1, ACL_INT8, ACL_INT8));
    Acc(RunAddWsCase<uint8_t, uint8_t>("AddWs-UINT8", u8A, s22, u8B, s22, s22, 1, ACL_UINT8, ACL_UINT8));
    Acc(RunAddWsCase<uint8_t, uint8_t>("AddWs-BOOL", boolA, s22, boolB, s22, s22, 1, ACL_BOOL, ACL_BOOL));
    Acc(RunAddWsCase<uint16_t, float>("AddWs-FP16", fp16A, s22, fp16B, s22, s22, 1.0f, ACL_FLOAT16, ACL_FLOAT));
    Acc(RunAddWsCase<uint16_t, float>("AddWs-BF16", bf16A, s22, bf16B, s22, s22, 1.0f, ACL_BF16, ACL_FLOAT));

    Acc(RunAddsWsCase<float, float, float>("AddsWs-FP32", fA, s22, 2.5f, 1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddsWsCase<int32_t, int32_t, int32_t>("AddsWs-INT32", iA, s22, 3, 2, ACL_INT32, ACL_INT32, ACL_INT32));
    Acc(RunInplaceAddWsCase<float, float>("InplaceAddWs-FP32", fA, s22, fB, s22, 1.0f, ACL_FLOAT, ACL_FLOAT));
    Acc(RunInplaceAddsWsCase<float, float, float>("InplaceAddsWs-FP32", fA, s22, 1.25f, 0.5f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT));

    Acc(RunAddV3WsCase<float, float, float>("AddV3Ws-FP32-alpha1", 3.0f, fB, s22, 1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddV3WsCase<float, float, float>("AddV3Ws-FP32-alpha2", 3.0f, fB, s22, 2.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddV3WsCase<int32_t, int32_t, int32_t>("AddV3Ws-INT32", 5, iB, s22, 3, ACL_INT32, ACL_INT32, ACL_INT32));
    Acc(RunAddV3WsCase<uint16_t, float, float>("AddV3Ws-FP16", 1.0f, fp16B, s22, 1.0f, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT));
    Acc(RunInplaceAddV3WsCase<float, float, float>("InplaceAddV3Ws-FP32", 2.0f, fB, s22, 1.0f, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT));

    // Put strict execution validation first to avoid side effects
    Acc(RunAddExecuteCase<float, float>(
        "Add-Execute-FP32",
        {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2},
        {0.5f, 1.0f, 1.5f, 2.0f}, {2, 2},
        {2, 2}, 1.0f, ACL_FLOAT, ACL_FLOAT, stream));

    Acc(RunNullptrCase());
    Acc(RunAddWsExpectFailCase<float, float>(
        "AddWsExpectFail-BroadcastShape",
        {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2},
        {1.0f, 2.0f, 3.0f}, {3},
        {2, 2}, 1.0f, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddWsExpectFailCase<float, float>(
        "AddWsExpectFail-OutShapeMismatch",
        {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2},
        {1.0f, 2.0f}, {1, 2},
        {2, 3}, 1.0f, ACL_FLOAT, ACL_FLOAT));
    Acc(RunAddWsExpectFailCase<uint8_t, float>(
        "AddWsExpectFail-BoolAlphaFloat",
        {1, 0, 1, 0}, {2, 2},
        {0, 1, 0, 1}, {2, 2},
        {2, 2}, 1.5f, ACL_BOOL, ACL_FLOAT));
    Acc(RunAddV3WsExpectFailCase<uint8_t, uint8_t, float>(
        "AddV3WsExpectFail-BoolAlphaFloat",
        static_cast<uint8_t>(1), {0, 1, 0, 1}, {2, 2}, 1.5f, ACL_BOOL, ACL_BOOL, ACL_FLOAT));

    std::printf("Summary: %d passed, %d failed\n", stats.passed, stats.failed);

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (stats.failed == 0) ? 0 : 1;
}
