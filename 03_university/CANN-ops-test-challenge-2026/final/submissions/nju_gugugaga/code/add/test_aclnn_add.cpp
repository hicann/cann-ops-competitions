#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

// 兼容性定义：某些 CANN 版本不直接暴露这些宏/函数
#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif

namespace {

constexpr int64_t kDeviceId = 0;

struct CaseStats {
    int pass = 0;
    int fail = 0;
};

void PrintCaseResult(const char *name, bool ok)
{
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << name << std::endl;
}

size_t GetShapeElementCount(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return 1;
    }
    return static_cast<size_t>(
        std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>()));
}

size_t GetAllocBytes(size_t logicalBytes)
{
    return logicalBytes == 0 ? 1 : logicalBytes;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

void DestroyTensorAndAddr(aclTensor *tensor, void *addr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (addr != nullptr) {
        aclrtFree(addr);
    }
}

void DestroyScalar(aclScalar *s)
{
    if (s != nullptr) {
        aclDestroyScalar(s);
    }
}

// 通用 stream 获取：创建一个 stream 供后续使用
aclrtStream CreateStream()
{
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    return stream;
}

void SyncAndDestroyStream(aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtSynchronizeStream(stream);
        aclrtDestroyStream(stream);
    }
}

template <typename T>
bool CreateTensorFromHost(const std::vector<T> &hostData,
                          const std::vector<int64_t> &shape,
                          aclDataType dtype,
                          aclTensor **tensor,
                          void **deviceAddr)
{
    const size_t logicalBytes = hostData.size() * sizeof(T);
    const size_t allocBytes = GetAllocBytes(logicalBytes);
    if (aclrtMalloc(deviceAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    if (logicalBytes > 0) {
        if (aclrtMemcpy(*deviceAddr, logicalBytes, hostData.data(), logicalBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            aclrtFree(*deviceAddr);
            *deviceAddr = nullptr;
            return false;
        }
    }

    const std::vector<int64_t> strides = MakeContiguousStrides(shape);
    const int64_t *shapeData = shape.empty() ? nullptr : shape.data();
    const int64_t *strideData = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(shapeData, shape.size(), dtype, strideData, 0,
                              aclFormat::ACL_FORMAT_ND, shapeData, shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return false;
    }
    return true;
}

// 创建一个空 tensor（仅有 device 内存，不从 host 拷贝）
bool CreateEmptyTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                       aclTensor **tensor, void **deviceAddr)
{
    const size_t elemCount = GetShapeElementCount(shape);
    size_t typeSize = sizeof(float);
    if (dtype == ACL_FLOAT16 || dtype == ACL_BF16) typeSize = 2;
    else if (dtype == ACL_INT32 || dtype == ACL_FLOAT) typeSize = 4;
    else if (dtype == ACL_INT64 || dtype == ACL_DOUBLE) typeSize = 8;
    else if (dtype == ACL_INT8 || dtype == ACL_UINT8 || dtype == ACL_BOOL) typeSize = 1;
    else if (dtype == ACL_INT16) typeSize = 2;
    const size_t allocBytes = GetAllocBytes(elemCount * typeSize);
    if (aclrtMalloc(deviceAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    const std::vector<int64_t> strides = MakeContiguousStrides(shape);
    const int64_t *shapeData = shape.empty() ? nullptr : shape.data();
    const int64_t *strideData = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(shapeData, shape.size(), dtype, strideData, 0,
                              aclFormat::ACL_FORMAT_ND, shapeData, shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return false;
    }
    return true;
}

template <typename T>
bool CopyTensorToHost(void *deviceAddr, std::vector<T> *hostData)
{
    const size_t bytes = hostData->size() * sizeof(T);
    if (bytes == 0) {
        return true;
    }
    return aclrtMemcpy(hostData->data(), bytes, deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

// ============================================================
// 基础帮助函数：执行完整的 op 调用链并对比结果
// ============================================================

// aclnnAdd (Tensor + Tensor with alpha)
template <typename T>
bool RunAddAndCheck(const char *name,
                    const std::vector<T> &selfHost, const std::vector<int64_t> &selfShape,
                    aclDataType selfDtype,
                    const std::vector<T> &otherHost, const std::vector<int64_t> &otherShape,
                    aclDataType otherDtype,
                    aclDataType outDtype,
                    float alpha, const std::vector<T> &expected,
                    const std::vector<int64_t> &outShape,
                    float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(GetShapeElementCount(outShape), T(0));
    bool ok = CreateTensorFromHost(selfHost, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherHost, otherShape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outHost, outShape, outDtype, &out, &outAddr);
    if (ok) {
        alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnAddGetWorkspaceSize(self, other, alphaScalar, out, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnAdd(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) {
        ok = CopyTensorToHost(outAddr, &outHost);
    }
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);

    PrintCaseResult(name, ok);
    return ok;
}

template <typename T, typename AlphaT>
bool RunAddTypedAlphaAndCheck(const char *name,
                              const std::vector<T> &selfHost, const std::vector<int64_t> &selfShape,
                              aclDataType selfDtype,
                              const std::vector<T> &otherHost, const std::vector<int64_t> &otherShape,
                              aclDataType otherDtype,
                              aclDataType outDtype,
                              AlphaT alpha, aclDataType alphaType, const std::vector<T> &expected,
                              const std::vector<int64_t> &outShape,
                              float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(GetShapeElementCount(outShape), T(0));
    bool ok = CreateTensorFromHost(selfHost, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherHost, otherShape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outHost, outShape, outDtype, &out, &outAddr);
    if (ok) {
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnAddGetWorkspaceSize(self, other, alphaScalar, out, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnAdd(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) {
        ok = CopyTensorToHost(outAddr, &outHost);
    }
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);

    PrintCaseResult(name, ok);
    return ok;
}

// aclnnAdds (Tensor + Scalar with alpha)
template <typename T>
bool RunAddsAndCheck(const char *name,
                     const std::vector<T> &selfHost, const std::vector<int64_t> &shape,
                     aclDataType selfDtype,
                     float other, aclDataType otherScalarType,
                     float alpha, aclDataType alphaType,
                     aclDataType outDtype,
                     const std::vector<T> &expected,
                     float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    void *selfAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *otherScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(selfHost.size(), T(0));
    bool ok = CreateTensorFromHost(selfHost, shape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(outHost, shape, outDtype, &out, &outAddr);
    if (ok) {
        otherScalar = aclCreateScalar(&other, otherScalarType);
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = otherScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnAddsGetWorkspaceSize(self, otherScalar, alphaScalar, out, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnAdds(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(outAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(otherScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(out, outAddr);

    PrintCaseResult(name, ok);
    return ok;
}

template <typename T, typename OtherT, typename AlphaT>
bool RunAddsTypedScalarsAndCheck(const char *name,
                                 const std::vector<T> &selfHost, const std::vector<int64_t> &shape,
                                 aclDataType selfDtype,
                                 OtherT other, aclDataType otherType,
                                 AlphaT alpha, aclDataType alphaType,
                                 aclDataType outDtype,
                                 const std::vector<T> &expected,
                                 float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    void *selfAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *otherScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(selfHost.size(), T(0));
    bool ok = CreateTensorFromHost(selfHost, shape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(outHost, shape, outDtype, &out, &outAddr);
    if (ok) {
        otherScalar = aclCreateScalar(&other, otherType);
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = otherScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnAddsGetWorkspaceSize(self, otherScalar, alphaScalar, out, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnAdds(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(outAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(otherScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(out, outAddr);

    PrintCaseResult(name, ok);
    return ok;
}

template <typename T, typename OtherT, typename AlphaT>
bool RunInplaceAddsTypedScalarsAndCheck(const char *name,
                                        const std::vector<T> &selfHost, const std::vector<int64_t> &shape,
                                        aclDataType selfDtype,
                                        OtherT other, aclDataType otherType,
                                        AlphaT alpha, aclDataType alphaType,
                                        const std::vector<T> &expected,
                                        float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    void *selfAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *otherScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(selfHost.size(), T(0));
    bool ok = CreateTensorFromHost(selfHost, shape, selfDtype, &self, &selfAddr);
    if (ok) {
        otherScalar = aclCreateScalar(&other, otherType);
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = otherScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnInplaceAddsGetWorkspaceSize(self, otherScalar, alphaScalar, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(selfAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(otherScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);

    PrintCaseResult(name, ok);
    return ok;
}

// aclnnInplaceAdd (inplace Tensor + Tensor)
template <typename T>
bool RunInplaceAddAndCheck(const char *name,
                           const std::vector<T> &selfHost, const std::vector<int64_t> &selfShape,
                           aclDataType selfDtype,
                           const std::vector<T> &otherHost, const std::vector<int64_t> &otherShape,
                           aclDataType otherDtype,
                           float alpha, const std::vector<T> &expected,
                           float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(selfHost.size(), T(0));
    bool ok = CreateTensorFromHost(selfHost, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherHost, otherShape, otherDtype, &other, &otherAddr);
    if (ok) {
        alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnInplaceAddGetWorkspaceSize(self, other, alphaScalar, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(selfAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);

    PrintCaseResult(name, ok);
    return ok;
}

template <typename T, typename AlphaT>
bool RunInplaceAddTypedAlphaAndCheck(const char *name,
                                     const std::vector<T> &selfHost, const std::vector<int64_t> &selfShape,
                                     aclDataType selfDtype,
                                     const std::vector<T> &otherHost, const std::vector<int64_t> &otherShape,
                                     aclDataType otherDtype,
                                     AlphaT alpha, aclDataType alphaType,
                                     const std::vector<T> &expected,
                                     float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(selfHost.size(), T(0));
    bool ok = CreateTensorFromHost(selfHost, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherHost, otherShape, otherDtype, &other, &otherAddr);
    if (ok) {
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnInplaceAddGetWorkspaceSize(self, other, alphaScalar, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(selfAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);

    PrintCaseResult(name, ok);
    return ok;
}

// aclnnInplaceAdds (inplace Tensor + Scalar)
template <typename T>
bool RunInplaceAddsAndCheck(const char *name,
                            const std::vector<T> &selfHost, const std::vector<int64_t> &shape,
                            aclDataType selfDtype,
                            float other, float alpha,
                            const std::vector<T> &expected,
                            float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *self = nullptr;
    void *selfAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *otherScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(selfHost.size(), T(0));
    bool ok = CreateTensorFromHost(selfHost, shape, selfDtype, &self, &selfAddr);
    if (ok) {
        otherScalar = aclCreateScalar(&other, ACL_FLOAT);
        alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = otherScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnInplaceAddsGetWorkspaceSize(self, otherScalar, alphaScalar, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(selfAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(otherScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);

    PrintCaseResult(name, ok);
    return ok;
}

// aclnnAddV3 (Scalar + Tensor with alpha)
template <typename T>
bool RunAddV3AndCheck(const char *name,
                      float selfValue, aclDataType selfType,
                      const std::vector<T> &otherHost, const std::vector<int64_t> &shape,
                      aclDataType otherDtype,
                      aclDataType outDtype,
                      float alpha, const std::vector<T> &expected,
                      float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *selfScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(otherHost.size(), T(0));
    bool ok = CreateTensorFromHost(otherHost, shape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outHost, shape, outDtype, &out, &outAddr);
    if (ok) {
        selfScalar = aclCreateScalar(&selfValue, selfType);
        alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = selfScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnAddV3GetWorkspaceSize(selfScalar, other, alphaScalar, out, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(outAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(selfScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);

    PrintCaseResult(name, ok);
    return ok;
}

template <typename T, typename SelfT, typename AlphaT>
bool RunAddV3TypedAndCheck(const char *name,
                           SelfT selfValue, aclDataType selfType,
                           const std::vector<T> &otherHost, const std::vector<int64_t> &shape,
                           aclDataType otherDtype,
                           aclDataType outDtype,
                           AlphaT alpha, aclDataType alphaType,
                           const std::vector<T> &expected,
                           float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *selfScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(otherHost.size(), T(0));
    bool ok = CreateTensorFromHost(otherHost, shape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outHost, shape, outDtype, &out, &outAddr);
    if (ok) {
        selfScalar = aclCreateScalar(&selfValue, selfType);
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = selfScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnAddV3GetWorkspaceSize(selfScalar, other, alphaScalar, out, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(outAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(selfScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);

    PrintCaseResult(name, ok);
    return ok;
}

// aclnnInplaceAddV3 (inplace Tensor = selfValue + other * alpha)
template <typename T>
bool RunInplaceAddV3AndCheck(const char *name,
                             float selfValue, aclDataType selfType,
                             const std::vector<T> &otherHost, const std::vector<int64_t> &shape,
                             aclDataType otherDtype,
                             float alpha, const std::vector<T> &expected,
                             float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *other = nullptr;
    void *otherAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *selfScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(otherHost.size(), T(0));
    bool ok = CreateTensorFromHost(otherHost, shape, otherDtype, &other, &otherAddr);
    if (ok) {
        selfScalar = aclCreateScalar(&selfValue, selfType);
        alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = selfScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, other, alphaScalar, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(otherAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(selfScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(other, otherAddr);

    PrintCaseResult(name, ok);
    return ok;
}

template <typename T, typename SelfT, typename AlphaT>
bool RunInplaceAddV3TypedAndCheck(const char *name,
                                  SelfT selfValue, aclDataType selfType,
                                  const std::vector<T> &otherHost, const std::vector<int64_t> &shape,
                                  aclDataType otherDtype,
                                  AlphaT alpha, aclDataType alphaType,
                                  const std::vector<T> &expected,
                                  float atol = 1e-3f, float rtol = 1e-3f)
{
    aclTensor *other = nullptr;
    void *otherAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *selfScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    std::vector<T> outHost(otherHost.size(), T(0));
    bool ok = CreateTensorFromHost(otherHost, shape, otherDtype, &other, &otherAddr);
    if (ok) {
        selfScalar = aclCreateScalar(&selfValue, selfType);
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = selfScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, other, alphaScalar, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }
    if (ok) ok = CopyTensorToHost(otherAddr, &outHost);
    if (ok && !expected.empty()) {
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = static_cast<double>(std::abs(static_cast<double>(outHost[i]) - static_cast<double>(expected[i])));
            double tol = atol + rtol * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) { ok = false; break; }
        }
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(selfScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(other, otherAddr);

    PrintCaseResult(name, ok);
    return ok;
}

template <typename OtherT, typename SelfT, typename AlphaT>
bool RunInplaceAddV3MixedNoCheck(const char *name,
                                 SelfT selfValue, aclDataType selfType,
                                 const std::vector<OtherT> &otherHost, const std::vector<int64_t> &shape,
                                 aclDataType otherDtype,
                                 AlphaT alpha, aclDataType alphaType)
{
    aclTensor *other = nullptr;
    void *otherAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *selfScalar = nullptr;
    aclScalar *alphaScalar = nullptr;

    bool ok = CreateTensorFromHost(otherHost, shape, otherDtype, &other, &otherAddr);
    if (ok) {
        selfScalar = aclCreateScalar(&selfValue, selfType);
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = selfScalar != nullptr && alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, other, alphaScalar, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(selfScalar);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(other, otherAddr);

    PrintCaseResult(name, ok);
    return ok;
}

// ============================================================
// 仅探测 GetWorkspaceSize 返回码（不运行 kernel）
// ============================================================

bool ProbeAddGetWorkspace(const char *name,
                          aclTensor *self, aclTensor *other, aclScalar *alphaScalar, aclTensor *out,
                          bool expectSuccess)
{
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnAddGetWorkspaceSize(self, other, alphaScalar, out, &ws, &exec);
    bool ok = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    PrintCaseResult(name, ok);
    return ok;
}

bool ProbeAddsGetWorkspace(const char *name,
                           aclTensor *self, aclScalar *otherScalar, aclScalar *alphaScalar, aclTensor *out,
                           bool expectSuccess)
{
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnAddsGetWorkspaceSize(self, otherScalar, alphaScalar, out, &ws, &exec);
    bool ok = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    PrintCaseResult(name, ok);
    return ok;
}

bool ProbeInplaceAddGetWorkspace(const char *name,
                                 aclTensor *self, aclTensor *other, aclScalar *alphaScalar,
                                 bool expectSuccess)
{
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnInplaceAddGetWorkspaceSize(self, other, alphaScalar, &ws, &exec);
    bool ok = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    PrintCaseResult(name, ok);
    return ok;
}

bool ProbeInplaceAddsGetWorkspace(const char *name,
                                  aclTensor *self, aclScalar *otherScalar, aclScalar *alphaScalar,
                                  bool expectSuccess)
{
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnInplaceAddsGetWorkspaceSize(self, otherScalar, alphaScalar, &ws, &exec);
    bool ok = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    PrintCaseResult(name, ok);
    return ok;
}

bool ProbeAddV3GetWorkspace(const char *name,
                            aclScalar *selfScalar, aclTensor *other, aclScalar *alphaScalar, aclTensor *out,
                            bool expectSuccess)
{
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alphaScalar, out, &ws, &exec);
    bool ok = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    PrintCaseResult(name, ok);
    return ok;
}

bool ProbeInplaceAddV3GetWorkspace(const char *name,
                                   aclScalar *selfScalar, aclTensor *other, aclScalar *alphaScalar,
                                   bool expectSuccess)
{
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, other, alphaScalar, &ws, &exec);
    bool ok = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    PrintCaseResult(name, ok);
    return ok;
}

bool ProbeInplaceAddDtypeCombo(const char *name,
                               const std::vector<int64_t> &selfShape,
                               const std::vector<int64_t> &otherShape,
                               aclDataType selfDtype, aclDataType otherDtype,
                               float alpha, bool expectSuccess)
{
    const size_t selfN = GetShapeElementCount(selfShape);
    const size_t otherN = GetShapeElementCount(otherShape);
    std::vector<float> selfDummy(selfN, 1.0f);
    std::vector<float> otherDummy(otherN, 1.0f);
    aclTensor *self = nullptr, *other = nullptr;
    void *selfAddr = nullptr, *otherAddr = nullptr;
    bool ok = CreateTensorFromHost(selfDummy, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherDummy, otherShape, otherDtype, &other, &otherAddr);
    if (ok) {
        aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = ProbeInplaceAddGetWorkspace(name, self, other, alphaScalar, expectSuccess);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    return ok;
}

template <typename SelfT, typename AlphaT>
bool ProbeInplaceAddV3TypedScalars(const char *name,
                                   SelfT selfVal, aclDataType selfType,
                                   const std::vector<int64_t> &shape,
                                   aclDataType otherDtype,
                                   AlphaT alpha, aclDataType alphaType, bool expectSuccess)
{
    size_t n = GetShapeElementCount(shape);
    std::vector<float> dummy(n, 1.0f);
    aclTensor *other = nullptr;
    void *otherAddr = nullptr;
    bool ok = CreateTensorFromHost(dummy, shape, otherDtype, &other, &otherAddr);
    if (ok) {
        aclScalar *selfScalar = aclCreateScalar(&selfVal, selfType);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeInplaceAddV3GetWorkspace(name, selfScalar, other, alphaScalar, expectSuccess);
        DestroyScalar(selfScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(other, otherAddr);
    return ok;
}

// 通用 dtype 探测辅助函数
bool ProbeAddDtypeCombo(const char *name,
                        const std::vector<int64_t> &shape,
                        aclDataType selfDtype, aclDataType otherDtype, aclDataType outDtype,
                        float alpha, bool expectSuccess)
{
    size_t n = GetShapeElementCount(shape);
    std::vector<float> dummy(n, 1.0f);
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    void *selfAddr = nullptr, *otherAddr = nullptr, *outAddr = nullptr;
    bool ok = CreateTensorFromHost(dummy, shape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(dummy, shape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(dummy, shape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = ProbeAddGetWorkspace(name, self, other, alphaScalar, out, expectSuccess);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

template <typename AlphaT>
bool ProbeAddTypedAlpha(const char *name,
                        const std::vector<int64_t> &selfShape,
                        const std::vector<int64_t> &otherShape,
                        const std::vector<int64_t> &outShape,
                        aclDataType selfDtype, aclDataType otherDtype, aclDataType outDtype,
                        AlphaT alpha, aclDataType alphaType, bool expectSuccess)
{
    const size_t selfN = GetShapeElementCount(selfShape);
    const size_t otherN = GetShapeElementCount(otherShape);
    const size_t outN = GetShapeElementCount(outShape);
    std::vector<float> selfDummy(selfN, 1.0f);
    std::vector<float> otherDummy(otherN, 1.0f);
    std::vector<float> outDummy(outN, 0.0f);
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    void *selfAddr = nullptr, *otherAddr = nullptr, *outAddr = nullptr;
    bool ok = CreateTensorFromHost(selfDummy, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherDummy, otherShape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outDummy, outShape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeAddGetWorkspace(name, self, other, alphaScalar, out, expectSuccess);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

template <typename SelfT, typename OtherT, typename OutT, typename AlphaT>
bool ProbeAddTypedTensors(const char *name,
                          const std::vector<SelfT> &selfHost, const std::vector<int64_t> &selfShape,
                          aclDataType selfDtype,
                          const std::vector<OtherT> &otherHost, const std::vector<int64_t> &otherShape,
                          aclDataType otherDtype,
                          const std::vector<OutT> &outHost, const std::vector<int64_t> &outShape,
                          aclDataType outDtype,
                          AlphaT alpha, aclDataType alphaType, bool expectSuccess)
{
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    bool ok = CreateTensorFromHost(selfHost, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherHost, otherShape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outHost, outShape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeAddGetWorkspace(name, self, other, alphaScalar, out, expectSuccess);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

template <typename SelfT, typename OtherT, typename OutT, typename AlphaT>
bool RunAddMixedNoCheck(const char *name,
                        const std::vector<SelfT> &selfHost, const std::vector<int64_t> &selfShape,
                        aclDataType selfDtype,
                        const std::vector<OtherT> &otherHost, const std::vector<int64_t> &otherShape,
                        aclDataType otherDtype,
                        aclDataType outDtype,
                        AlphaT alpha, aclDataType alphaType,
                        const std::vector<int64_t> &outShape)
{
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    void *selfAddr = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    aclScalar *alphaScalar = nullptr;
    std::vector<OutT> outHost(GetShapeElementCount(outShape), OutT(0));

    bool ok = CreateTensorFromHost(selfHost, selfShape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(otherHost, otherShape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outHost, outShape, outDtype, &out, &outAddr);
    if (ok) {
        alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = alphaScalar != nullptr;
    }

    uint64_t workspaceSize = 0;
    if (ok) {
        ok = aclnnAddGetWorkspaceSize(self, other, alphaScalar, out, &workspaceSize, &executor) == ACL_SUCCESS;
    }
    if (ok && workspaceSize > 0) {
        ok = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        aclrtStream stream = CreateStream();
        ok = aclnnAdd(workspaceAddr, workspaceSize, executor, stream) == ACL_SUCCESS;
        if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        SyncAndDestroyStream(stream);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyScalar(alphaScalar);
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);

    PrintCaseResult(name, ok);
    return ok;
}

// 通用 Adds dtype 探测
bool ProbeAddsDtypeCombo(const char *name,
                         const std::vector<int64_t> &shape,
                         aclDataType selfDtype, float otherVal, aclDataType otherScalarType,
                         float alpha, aclDataType alphaType,
                         aclDataType outDtype, bool expectSuccess)
{
    size_t n = GetShapeElementCount(shape);
    std::vector<float> dummy(n, 1.0f);
    aclTensor *self = nullptr, *out = nullptr;
    void *selfAddr = nullptr, *outAddr = nullptr;
    bool ok = CreateTensorFromHost(dummy, shape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(dummy, shape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *otherScalar = aclCreateScalar(&otherVal, otherScalarType);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeAddsGetWorkspace(name, self, otherScalar, alphaScalar, out, expectSuccess);
        DestroyScalar(otherScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

template <typename OtherT, typename AlphaT>
bool ProbeAddsTypedScalars(const char *name,
                           const std::vector<int64_t> &shape,
                           aclDataType selfDtype,
                           OtherT otherVal, aclDataType otherScalarType,
                           AlphaT alpha, aclDataType alphaType,
                           aclDataType outDtype, bool expectSuccess)
{
    size_t n = GetShapeElementCount(shape);
    std::vector<float> dummy(n, 1.0f);
    aclTensor *self = nullptr, *out = nullptr;
    void *selfAddr = nullptr, *outAddr = nullptr;
    bool ok = CreateTensorFromHost(dummy, shape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(dummy, shape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *otherScalar = aclCreateScalar(&otherVal, otherScalarType);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeAddsGetWorkspace(name, self, otherScalar, alphaScalar, out, expectSuccess);
        DestroyScalar(otherScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

template <typename SelfT, typename OutT, typename OtherT, typename AlphaT>
bool ProbeAddsTypedTensorScalars(const char *name,
                                 const std::vector<SelfT> &selfHost, const std::vector<int64_t> &shape,
                                 aclDataType selfDtype,
                                 OtherT otherVal, aclDataType otherScalarType,
                                 AlphaT alpha, aclDataType alphaType,
                                 const std::vector<OutT> &outHost,
                                 aclDataType outDtype, bool expectSuccess)
{
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    void *selfAddr = nullptr;
    void *outAddr = nullptr;
    bool ok = CreateTensorFromHost(selfHost, shape, selfDtype, &self, &selfAddr) &&
              CreateTensorFromHost(outHost, shape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *otherScalar = aclCreateScalar(&otherVal, otherScalarType);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeAddsGetWorkspace(name, self, otherScalar, alphaScalar, out, expectSuccess);
        DestroyScalar(otherScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

// AddV3 通用 dtype 探测
bool ProbeAddV3DtypeCombo(const char *name,
                          float selfVal, aclDataType selfType,
                          const std::vector<int64_t> &shape,
                          aclDataType otherDtype, aclDataType outDtype,
                          float alpha, bool expectSuccess)
{
    size_t n = GetShapeElementCount(shape);
    std::vector<float> dummy(n, 1.0f);
    aclTensor *other = nullptr, *out = nullptr;
    void *otherAddr = nullptr, *outAddr = nullptr;
    bool ok = CreateTensorFromHost(dummy, shape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(dummy, shape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *selfScalar = aclCreateScalar(&selfVal, selfType);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = ProbeAddV3GetWorkspace(name, selfScalar, other, alphaScalar, out, expectSuccess);
        DestroyScalar(selfScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

template <typename SelfT, typename AlphaT>
bool ProbeAddV3TypedScalars(const char *name,
                            SelfT selfVal, aclDataType selfType,
                            const std::vector<int64_t> &shape,
                            aclDataType otherDtype, aclDataType outDtype,
                            AlphaT alpha, aclDataType alphaType, bool expectSuccess)
{
    size_t n = GetShapeElementCount(shape);
    std::vector<float> dummy(n, 1.0f);
    aclTensor *other = nullptr, *out = nullptr;
    void *otherAddr = nullptr, *outAddr = nullptr;
    bool ok = CreateTensorFromHost(dummy, shape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(dummy, shape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *selfScalar = aclCreateScalar(&selfVal, selfType);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeAddV3GetWorkspace(name, selfScalar, other, alphaScalar, out, expectSuccess);
        DestroyScalar(selfScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

template <typename OtherT, typename OutT, typename SelfT, typename AlphaT>
bool ProbeAddV3TypedTensorScalar(const char *name,
                                 SelfT selfVal, aclDataType selfType,
                                 const std::vector<OtherT> &otherHost, const std::vector<int64_t> &shape,
                                 aclDataType otherDtype,
                                 const std::vector<OutT> &outHost, aclDataType outDtype,
                                 AlphaT alpha, aclDataType alphaType, bool expectSuccess)
{
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    void *otherAddr = nullptr;
    void *outAddr = nullptr;
    bool ok = CreateTensorFromHost(otherHost, shape, otherDtype, &other, &otherAddr) &&
              CreateTensorFromHost(outHost, shape, outDtype, &out, &outAddr);
    if (ok) {
        aclScalar *selfScalar = aclCreateScalar(&selfVal, selfType);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, alphaType);
        ok = ProbeAddV3GetWorkspace(name, selfScalar, other, alphaScalar, out, expectSuccess);
        DestroyScalar(selfScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

// 空指针测试辅助
bool ProbeNullptrTest(const char *name, int apiType)
{
    float alpha = 1.0f;
    float other = 1.0f;
    std::vector<float> dummy = {1.0f};
    std::vector<int64_t> shape = {1};

    aclTensor *t1 = nullptr, *t2 = nullptr;
    void *a1 = nullptr, *a2 = nullptr;
    aclScalar *s1 = nullptr, *s2 = nullptr;

    CreateTensorFromHost(dummy, shape, ACL_FLOAT, &t1, &a1);
    CreateTensorFromHost(dummy, shape, ACL_FLOAT, &t2, &a2);
    s1 = aclCreateScalar(&other, ACL_FLOAT);
    s2 = aclCreateScalar(&alpha, ACL_FLOAT);

    bool ok = false;
    switch (apiType) {
        case 0: ok = ProbeAddGetWorkspace(name, nullptr, t2, s2, t1, false); break;
        case 1: ok = ProbeAddGetWorkspace(name, t1, nullptr, s2, t1, false); break;
        case 2: ok = ProbeAddGetWorkspace(name, t1, t2, nullptr, t1, false); break;
        case 3: ok = ProbeAddGetWorkspace(name, t1, t2, s2, nullptr, false); break;
        case 4: ok = ProbeAddsGetWorkspace(name, nullptr, s1, s2, t1, false); break;
        case 5: ok = ProbeAddsGetWorkspace(name, t1, nullptr, s2, t1, false); break;
        case 6: ok = ProbeAddsGetWorkspace(name, t1, s1, nullptr, t1, false); break;
        case 7: ok = ProbeAddsGetWorkspace(name, t1, s1, s2, nullptr, false); break;
        case 8: ok = ProbeInplaceAddGetWorkspace(name, nullptr, t2, s2, false); break;
        case 9: ok = ProbeInplaceAddGetWorkspace(name, t1, nullptr, s2, false); break;
        case 10: ok = ProbeInplaceAddsGetWorkspace(name, nullptr, s1, s2, false); break;
        case 11: ok = ProbeInplaceAddsGetWorkspace(name, t1, nullptr, s2, false); break;
        case 12: ok = ProbeInplaceAddsGetWorkspace(name, t1, s1, nullptr, false); break;
        case 13: {
            aclScalar *selfScalar = aclCreateScalar(&alpha, ACL_FLOAT);
            ok = ProbeAddV3GetWorkspace(name, nullptr, t2, s2, t1, false);
            DestroyScalar(selfScalar);
            break;
        }
        case 14: ok = ProbeAddV3GetWorkspace(name, s1, nullptr, s2, t1, false); break;
        case 15: ok = ProbeAddV3GetWorkspace(name, s1, t2, nullptr, t1, false); break;
        case 16: ok = ProbeAddV3GetWorkspace(name, s1, t2, s2, nullptr, false); break;
        case 17: ok = ProbeInplaceAddV3GetWorkspace(name, nullptr, t1, s2, false); break;
        case 18: ok = ProbeInplaceAddV3GetWorkspace(name, s1, nullptr, s2, false); break;
        case 19: ok = ProbeInplaceAddV3GetWorkspace(name, s1, t1, nullptr, false); break;
        default: ok = false; break;
    }

    DestroyScalar(s1);
    DestroyScalar(s2);
    DestroyTensorAndAddr(t1, a1);
    DestroyTensorAndAddr(t2, a2);
    return ok;
}

// 非法 shape 广播测试
bool ProbeInvalidBroadcast(const char *name)
{
    std::vector<float> d1 = {1.0f, 2.0f};
    std::vector<float> d2 = {1.0f, 2.0f, 3.0f};
    std::vector<float> d3(2, 0.0f);
    std::vector<int64_t> s1 = {2};
    std::vector<int64_t> s2 = {3};

    aclTensor *t1 = nullptr, *t2 = nullptr, *out = nullptr;
    void *a1 = nullptr, *a2 = nullptr, *a3 = nullptr;
    bool ok = CreateTensorFromHost(d1, s1, ACL_FLOAT, &t1, &a1) &&
              CreateTensorFromHost(d2, s2, ACL_FLOAT, &t2, &a2) &&
              CreateTensorFromHost(d3, s1, ACL_FLOAT, &out, &a3);
    if (ok) {
        float alpha = 1.0f;
        aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = ProbeAddGetWorkspace(name, t1, t2, alphaScalar, out, false);
        DestroyScalar(alphaScalar);
    }
    DestroyTensorAndAddr(t1, a1);
    DestroyTensorAndAddr(t2, a2);
    DestroyTensorAndAddr(out, a3);
    return ok;
}

bool ProbeInvalidOutShape(const char *name)
{
    float alpha = 1.0f;
    return ProbeAddTypedAlpha(name, {2}, {2}, {3}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, alpha, ACL_FLOAT, false);
}

bool ProbeMaxDimTensor(const char *name)
{
    const std::vector<int64_t> tooManyDims = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    float alpha = 1.0f;
    return ProbeAddTypedAlpha(name, tooManyDims, {1}, tooManyDims, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, alpha, ACL_FLOAT, false);
}

bool ProbeInvalidInplaceBroadcast(const char *name)
{
    std::vector<float> selfData = {1.0f};
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f};
    aclTensor *self = nullptr, *other = nullptr;
    void *selfAddr = nullptr, *otherAddr = nullptr;
    bool ok = CreateTensorFromHost(selfData, {1}, ACL_FLOAT, &self, &selfAddr) &&
              CreateTensorFromHost(otherData, {3}, ACL_FLOAT, &other, &otherAddr);
    if (ok) {
        float alpha = 1.0f;
        aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = ProbeInplaceAddGetWorkspace(name, self, other, alphaScalar, false);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(self, selfAddr);
    DestroyTensorAndAddr(other, otherAddr);
    return ok;
}

bool ProbeInvalidAddV3OutShape(const char *name)
{
    std::vector<float> otherData = {1.0f, 2.0f};
    std::vector<float> outData(3, 0.0f);
    aclTensor *other = nullptr, *out = nullptr;
    void *otherAddr = nullptr, *outAddr = nullptr;
    bool ok = CreateTensorFromHost(otherData, {2}, ACL_FLOAT, &other, &otherAddr) &&
              CreateTensorFromHost(outData, {3}, ACL_FLOAT, &out, &outAddr);
    if (ok) {
        float selfVal = 1.0f;
        float alpha = 1.0f;
        aclScalar *selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
        aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
        ok = ProbeAddV3GetWorkspace(name, selfScalar, other, alphaScalar, out, false);
        DestroyScalar(selfScalar);
        DestroyScalar(alphaScalar);
    } else {
        PrintCaseResult(name, false);
    }
    DestroyTensorAndAddr(other, otherAddr);
    DestroyTensorAndAddr(out, outAddr);
    return ok;
}

bool ProbeMaxDimAddV3(const char *name)
{
    const std::vector<int64_t> tooManyDims = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    return ProbeAddV3TypedScalars(name, 1.0f, ACL_FLOAT, tooManyDims, ACL_FLOAT, ACL_FLOAT,
                                  1.0f, ACL_FLOAT, false);
}

// 空 tensor (0元素) 测试
bool ProbeEmptyTensor(const char *name, int apiType)
{
    std::vector<float> dummy;
    std::vector<int64_t> emptyShape = {0};

    float alpha = 1.0f;
    float other = 1.0f;
    aclTensor *t1 = nullptr, *t2 = nullptr;
    void *a1 = nullptr, *a2 = nullptr;
    aclScalar *s1 = nullptr, *s2 = nullptr;

    CreateEmptyTensor(emptyShape, ACL_FLOAT, &t1, &a1);
    CreateEmptyTensor(emptyShape, ACL_FLOAT, &t2, &a2);
    s1 = aclCreateScalar(&other, ACL_FLOAT);
    s2 = aclCreateScalar(&alpha, ACL_FLOAT);

    bool ok = false;
    switch (apiType) {
        case 0: ok = ProbeAddGetWorkspace(name, t1, t2, s2, t1, true); break;
        case 1: ok = ProbeAddsGetWorkspace(name, t1, s1, s2, t1, true); break;
        case 2: ok = ProbeInplaceAddGetWorkspace(name, t1, t2, s2, true); break;
        case 3: ok = ProbeInplaceAddsGetWorkspace(name, t1, s1, s2, true); break;
        case 4: ok = ProbeAddV3GetWorkspace(name, s1, t2, s2, t1, true); break;
        case 5: ok = ProbeInplaceAddV3GetWorkspace(name, s1, t1, s2, true); break;
        default: ok = false; break;
    }

    DestroyScalar(s1);
    DestroyScalar(s2);
    DestroyTensorAndAddr(t1, a1);
    DestroyTensorAndAddr(t2, a2);
    return ok;
}

} // namespace

int main()
{
    CaseStats stats;

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return 1;
    }
    if (aclrtSetDevice(kDeviceId) != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed" << std::endl;
        aclFinalize();
        return 1;
    }

    auto countResult = [&](bool ok) {
        if (ok) { stats.pass++; } else { stats.fail++; }
    };

    // ================================================================
    // 1. aclnnAdd: FLOAT32 基础功能测试
    // ================================================================
    countResult(RunAddAndCheck<float>("add_f32_alpha_ne_1",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {4.0f, 5.0f, 6.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        1.2f, {5.8f, 8.0f, 10.2f}, {3}));

    countResult(RunAddAndCheck<float>("add_f32_alpha_eq_1",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {4.0f, 5.0f, 6.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {5.0f, 7.0f, 9.0f}, {3}));

    countResult(RunAddAndCheck<float>("add_f32_alpha_0",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {4.0f, 5.0f, 6.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        0.0f, {1.0f, 2.0f, 3.0f}, {3}));

    countResult(RunAddAndCheck<float>("add_f32_alpha_neg",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {4.0f, 5.0f, 6.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        -0.5f, {-1.0f, -0.5f, 0.0f}, {3}));

    // ================================================================
    // 2. aclnnAdd: 广播测试
    // ================================================================
    countResult(RunAddAndCheck<float>("add_f32_broadcast_2x3",
        {1.0f, 3.0f}, {2, 1}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {1, 3}, ACL_FLOAT, ACL_FLOAT,
        0.5f, {6.0f, 11.0f, 16.0f, 8.0f, 13.0f, 18.0f}, {2, 3}));

    countResult(RunAddAndCheck<float>("add_f32_broadcast_scalar_shape",
        {5.0f}, {1}, ACL_FLOAT,
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {6.0f, 7.0f, 8.0f}, {3}));

    countResult(RunAddAndCheck<float>("add_f32_broadcast_1d_to_2d",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        2.0f, {21.0f, 42.0f, 63.0f}, {3}));

    countResult(RunAddAndCheck<float>("add_f32_broadcast_rank_diff_2x3_plus_3",
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f}, {2, 3}));

    countResult(RunAddAndCheck<float>("add_f32_broadcast_3d_tail",
        {1.0f, 2.0f}, {1, 2, 1}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {1, 1, 3}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f}, {1, 2, 3}));

    // ================================================================
    // 3. aclnnAdd: INT32 测试
    // ================================================================
    countResult(RunAddAndCheck<int32_t>("add_i32_basic",
        {1, 2, 3, 4}, {4}, ACL_INT32,
        {10, 20, 30, 40}, {4}, ACL_INT32, ACL_INT32,
        1.0f, {11, 22, 33, 44}, {4}));

    countResult(RunAddAndCheck<int32_t>("add_i32_alpha_ne_1",
        {1, 2, 3}, {3}, ACL_INT32,
        {10, 20, 30}, {3}, ACL_INT32, ACL_INT32,
        2.0f, {21, 42, 63}, {3}));

    countResult(RunAddTypedAlphaAndCheck<int32_t, int32_t>("add_i32_alpha_int_neg2",
        {10, -4, 7, 0}, {4}, ACL_INT32,
        {1, 2, -3, 9}, {4}, ACL_INT32, ACL_INT32,
        -2, ACL_INT32, {8, -8, 13, -18}, {4}));

    countResult(RunAddTypedAlphaAndCheck<float, int32_t>("add_f32_alpha_int3_exec",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {2.0f, 3.0f, 4.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        3, ACL_INT32, {7.0f, 11.0f, 15.0f}, {3}));

    // ================================================================
    // 4. aclnnAdd: INT64 测试
    // ================================================================
    countResult(RunAddAndCheck<int64_t>("add_i64_basic",
        {100LL, 200LL, 300LL}, {3}, ACL_INT64,
        {10LL, 20LL, 30LL}, {3}, ACL_INT64, ACL_INT64,
        1.0f, {110LL, 220LL, 330LL}, {3}));

    countResult(RunAddAndCheck<int64_t>("add_i64_alpha_2",
        {1LL, 2LL, 3LL}, {3}, ACL_INT64,
        {10LL, 20LL, 30LL}, {3}, ACL_INT64, ACL_INT64,
        2.0f, {21LL, 42LL, 63LL}, {3}));

    // ================================================================
    // 5. aclnnAdd: FLOAT16 测试
    // ================================================================
    {
        // FLOAT16 values: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200
        std::vector<uint16_t> selfH = {0x3C00, 0x4000, 0x4200};
        std::vector<uint16_t> otherH = {0x3C00, 0x4000, 0x4200};
        std::vector<uint16_t> outH(3, 0);
        aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
        void *selfAddr = nullptr, *otherAddr = nullptr, *outAddr = nullptr;
        void *wsAddr = nullptr;
        aclOpExecutor *exec = nullptr;
        bool ok = CreateTensorFromHost(selfH, {3}, ACL_FLOAT16, &self, &selfAddr) &&
                  CreateTensorFromHost(otherH, {3}, ACL_FLOAT16, &other, &otherAddr) &&
                  CreateTensorFromHost(outH, {3}, ACL_FLOAT16, &out, &outAddr);
        if (ok) {
            float alpha = 1.0f;
            aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
            uint64_t ws = 0;
            ok = aclnnAddGetWorkspaceSize(self, other, alphaScalar, out, &ws, &exec) == ACL_SUCCESS;
            if (ok && ws > 0) ok = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
            if (ok) {
                aclrtStream stream = CreateStream();
                ok = aclnnAdd(wsAddr, ws, exec, stream) == ACL_SUCCESS;
                if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
                SyncAndDestroyStream(stream);
            }
            if (ok) ok = CopyTensorToHost(outAddr, &outH);
            DestroyScalar(alphaScalar);
        }
        if (wsAddr) aclrtFree(wsAddr);
        DestroyTensorAndAddr(self, selfAddr);
        DestroyTensorAndAddr(other, otherAddr);
        DestroyTensorAndAddr(out, outAddr);
        PrintCaseResult("add_f16_basic", ok);
        countResult(ok);
    }

    // ================================================================
    // 6. aclnnAdd: BF16 测试
    // ================================================================
    {
        // BF16 values: 1.0=0x3F80, 2.0=0x4000, 3.0=0x4040
        std::vector<uint16_t> selfB = {0x3F80, 0x4000, 0x4040};
        std::vector<uint16_t> otherB = {0x3F80, 0x4000, 0x4040};
        std::vector<uint16_t> outB(3, 0);
        aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
        void *selfAddr = nullptr, *otherAddr = nullptr, *outAddr = nullptr;
        void *wsAddr = nullptr;
        aclOpExecutor *exec = nullptr;
        bool ok = CreateTensorFromHost(selfB, {3}, ACL_BF16, &self, &selfAddr) &&
                  CreateTensorFromHost(otherB, {3}, ACL_BF16, &other, &otherAddr) &&
                  CreateTensorFromHost(outB, {3}, ACL_BF16, &out, &outAddr);
        if (ok) {
            float alpha = 1.0f;
            aclScalar *alphaScalar = aclCreateScalar(&alpha, ACL_FLOAT);
            uint64_t ws = 0;
            ok = aclnnAddGetWorkspaceSize(self, other, alphaScalar, out, &ws, &exec) == ACL_SUCCESS;
            if (ok && ws > 0) ok = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
            if (ok) {
                aclrtStream stream = CreateStream();
                ok = aclnnAdd(wsAddr, ws, exec, stream) == ACL_SUCCESS;
                if (ok) ok = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
                SyncAndDestroyStream(stream);
            }
            if (ok) ok = CopyTensorToHost(outAddr, &outB);
            DestroyScalar(alphaScalar);
        }
        if (wsAddr) aclrtFree(wsAddr);
        DestroyTensorAndAddr(self, selfAddr);
        DestroyTensorAndAddr(other, otherAddr);
        DestroyTensorAndAddr(out, outAddr);
        PrintCaseResult("add_bf16_basic", ok);
        countResult(ok);
    }

    // ================================================================
    // 7. aclnnAdd: 混合类型 FLOAT16 + FLOAT32
    // ================================================================
    countResult(ProbeAddDtypeCombo("add_mixed_f16_f32",
        {4}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, 1.0f, true));

    countResult(ProbeAddDtypeCombo("add_mixed_f32_f16",
        {4}, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, 1.0f, true));

    // ================================================================
    // 8. aclnnAdd: UINT8 / INT8 测试
    // ================================================================
    countResult(ProbeAddDtypeCombo("add_u8_basic",
        {4}, ACL_UINT8, ACL_UINT8, ACL_UINT8, 1.0f, true));

    countResult(ProbeAddDtypeCombo("add_i8_basic",
        {4}, ACL_INT8, ACL_INT8, ACL_INT8, 1.0f, true));

    // ================================================================
    // 9. aclnnAdd: DOUBLE 测试
    // ================================================================
    countResult(ProbeAddDtypeCombo("add_f64_basic",
        {3}, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, 1.0f, true));

    countResult(ProbeAddDtypeCombo("add_f64_alpha2_mul_aicpu_path",
        {3}, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, 2.0f, true));

    countResult(ProbeAddTypedAlpha("add_bool_alpha_bool_success",
        {3}, {3}, {3}, ACL_BOOL, ACL_BOOL, ACL_BOOL, true, ACL_BOOL, true));

    countResult(ProbeAddTypedAlpha("add_bool_alpha_float_invalid",
        {3}, {3}, {3}, ACL_BOOL, ACL_BOOL, ACL_BOOL, 1.0f, ACL_FLOAT, false));

    {
        using Complex64 = std::complex<float>;
        using Complex128 = std::complex<double>;
        const std::vector<Complex64> c64 = {Complex64(1.0f, 2.0f), Complex64(-3.0f, 4.0f)};
        const std::vector<Complex128> c128 = {Complex128(1.0, -1.0), Complex128(2.0, 3.0)};
        const std::vector<float> f32 = {1.0f, 2.0f};
        const std::vector<Complex64> outC64(2);
        const std::vector<Complex128> outC128(2);
        const std::vector<float> outF32(2, 0.0f);
        const Complex64 complexAlpha(1.0f, 1.0f);

        countResult(ProbeAddTypedTensors<Complex64, Complex64, Complex64, float>(
            "add_complex64_basic", c64, {2}, ACL_COMPLEX64, c64, {2}, ACL_COMPLEX64,
            outC64, {2}, ACL_COMPLEX64, 1.0f, ACL_FLOAT, true));

        countResult(ProbeAddTypedTensors<Complex128, Complex128, Complex128, float>(
            "add_complex128_aicpu_path", c128, {2}, ACL_COMPLEX128, c128, {2}, ACL_COMPLEX128,
            outC128, {2}, ACL_COMPLEX128, 1.0f, ACL_FLOAT, true));

        countResult(ProbeAddTypedTensors<float, Complex64, Complex64, float>(
            "add_float_complex64_promote", f32, {2}, ACL_FLOAT, c64, {2}, ACL_COMPLEX64,
            outC64, {2}, ACL_COMPLEX64, 1.0f, ACL_FLOAT, true));

        countResult(ProbeAddTypedTensors<Complex64, Complex64, float, float>(
            "add_complex64_out_float_invalid", c64, {2}, ACL_COMPLEX64, c64, {2}, ACL_COMPLEX64,
            outF32, {2}, ACL_FLOAT, 1.0f, ACL_FLOAT, false));

        countResult(ProbeAddTypedTensors<float, float, float, Complex64>(
            "add_alpha_complex_to_float_invalid", f32, {2}, ACL_FLOAT, f32, {2}, ACL_FLOAT,
            outF32, {2}, ACL_FLOAT, complexAlpha, ACL_COMPLEX64, false));
    }

    // ================================================================
    // 10. aclnnAdds: 基本测试
    // ================================================================
    countResult(RunAddsAndCheck<float>("adds_f32_scalar",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        2.0f, ACL_FLOAT, -1.0f, ACL_FLOAT, ACL_FLOAT,
        {-1.0f, 0.0f, 1.0f}));

    countResult(RunAddsAndCheck<float>("adds_f32_alpha_1",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        5.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT,
        {6.0f, 7.0f, 8.0f}));

    countResult(RunAddsAndCheck<float>("adds_f32_alpha_0",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        5.0f, ACL_FLOAT, 0.0f, ACL_FLOAT, ACL_FLOAT,
        {1.0f, 2.0f, 3.0f}));

    countResult(RunAddsAndCheck<float>("adds_f32_neg_other",
        {10.0f, 20.0f}, {2}, ACL_FLOAT,
        -5.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT,
        {5.0f, 15.0f}));

    countResult(RunAddsTypedScalarsAndCheck<int32_t, int32_t, int32_t>("adds_i32_typed_other_alpha_exec",
        {1, 2, 3}, {3}, ACL_INT32,
        4, ACL_INT32, -2, ACL_INT32, ACL_INT32,
        {-7, -6, -5}));

    countResult(RunAddsTypedScalarsAndCheck<float, int32_t, int32_t>("adds_f32_int_scalar_int_alpha_exec",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        2, ACL_INT32, 3, ACL_INT32, ACL_FLOAT,
        {7.0f, 8.0f, 9.0f}));

    // ================================================================
    // 11. aclnnAdds: 不同 dtype
    // ================================================================
    countResult(ProbeAddsDtypeCombo("adds_i32_scalar",
        {3}, ACL_INT32, 5.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_INT32, true));

    countResult(ProbeAddsDtypeCombo("adds_f16_scalar",
        {3}, ACL_FLOAT16, 2.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT16, true));

    countResult(ProbeAddsDtypeCombo("adds_bf16_scalar",
        {3}, ACL_BF16, 3.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_BF16, true));

    countResult(ProbeAddsTypedScalars("adds_bool_true_to_i32_special",
        {3}, ACL_BOOL, true, ACL_BOOL, true, ACL_BOOL, ACL_INT32, true));

    countResult(ProbeAddsTypedScalars("adds_f16_scalar_inexact_promote_float",
        {3}, ACL_FLOAT16, 0.1f, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT, true));

    {
        using Complex64 = std::complex<float>;
        using Complex128 = std::complex<double>;
        const Complex64 c64Other(1.0f, 2.0f);
        const Complex128 c128Other(2.0, -1.0);
        const Complex64 c64Alpha(1.0f, 1.0f);
        const std::vector<float> f32 = {1.0f, 2.0f};
        const std::vector<uint16_t> f16 = {0x3C00, 0x4000};
        const std::vector<double> f64 = {1.0, 2.0};
        const std::vector<int32_t> i32 = {1, 2};
        const std::vector<Complex64> c64Self = {Complex64(1.0f, -1.0f), Complex64(2.0f, 3.0f)};
        const std::vector<Complex64> outC64(2);
        const std::vector<Complex128> outC128(2);
        const std::vector<float> outF32(2, 0.0f);

        countResult(ProbeAddsTypedTensorScalars<float, Complex64, Complex64, float>(
            "adds_float_complex64_scalar_promote", f32, {2}, ACL_FLOAT,
            c64Other, ACL_COMPLEX64, 1.0f, ACL_FLOAT, outC64, ACL_COMPLEX64, true));

        countResult(ProbeAddsTypedTensorScalars<uint16_t, Complex64, Complex64, float>(
            "adds_f16_complex64_scalar_promote", f16, {2}, ACL_FLOAT16,
            c64Other, ACL_COMPLEX64, 1.0f, ACL_FLOAT, outC64, ACL_COMPLEX64, true));

        countResult(ProbeAddsTypedTensorScalars<double, Complex128, Complex64, float>(
            "adds_double_complex64_scalar_promote", f64, {2}, ACL_DOUBLE,
            c64Other, ACL_COMPLEX64, 1.0f, ACL_FLOAT, outC128, ACL_COMPLEX128, true));

        countResult(ProbeAddsTypedTensorScalars<int32_t, Complex64, Complex64, float>(
            "adds_i32_complex64_scalar_promote", i32, {2}, ACL_INT32,
            c64Other, ACL_COMPLEX64, 1.0f, ACL_FLOAT, outC64, ACL_COMPLEX64, true));

        countResult(ProbeAddsTypedTensorScalars<Complex64, Complex64, float, float>(
            "adds_complex64_float_scalar_keep_complex", c64Self, {2}, ACL_COMPLEX64,
            2.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, outC64, ACL_COMPLEX64, true));

        countResult(ProbeAddsTypedTensorScalars<double, float, double, double>(
            "adds_double_scalar_out_float", f64, {2}, ACL_DOUBLE,
            2.0, ACL_DOUBLE, 1.0, ACL_DOUBLE, outF32, ACL_FLOAT, true));

        countResult(ProbeAddsTypedTensorScalars<float, float, float, Complex64>(
            "adds_alpha_complex_to_float_invalid", f32, {2}, ACL_FLOAT,
            2.0f, ACL_FLOAT, c64Alpha, ACL_COMPLEX64, outF32, ACL_FLOAT, false));

        countResult(ProbeAddsTypedTensorScalars<Complex64, float, float, float>(
            "adds_complex64_to_float_out_invalid", c64Self, {2}, ACL_COMPLEX64,
            2.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, outF32, ACL_FLOAT, false));

        countResult(ProbeAddsTypedTensorScalars<double, Complex128, Complex128, float>(
            "adds_double_complex128_scalar_promote", f64, {2}, ACL_DOUBLE,
            c128Other, ACL_COMPLEX128, 1.0f, ACL_FLOAT, outC128, ACL_COMPLEX128, true));
    }

    // ================================================================
    // 12. aclnnInplaceAdd: 基本测试
    // ================================================================
    countResult(RunInplaceAddAndCheck<float>("inplace_add_f32_basic",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT,
        1.0f, {11.0f, 22.0f, 33.0f}));

    countResult(RunInplaceAddAndCheck<float>("inplace_add_f32_alpha_ne_1",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT,
        0.5f, {6.0f, 12.0f, 18.0f}));

    countResult(RunInplaceAddAndCheck<int32_t>("inplace_add_i32_basic",
        {1, 2, 3}, {3}, ACL_INT32,
        {10, 20, 30}, {3}, ACL_INT32,
        1.0f, {11, 22, 33}));

    // ================================================================
    // 13. aclnnInplaceAdd: 广播测试
    // ================================================================
    countResult(RunInplaceAddAndCheck<float>("inplace_add_f32_broadcast",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {10.0f}, {1}, ACL_FLOAT,
        1.0f, {11.0f, 12.0f, 13.0f}));

    countResult(RunInplaceAddAndCheck<float>("inplace_add_f32_rank_diff_broadcast",
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT,
        1.0f, {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f}));

    countResult(RunInplaceAddAndCheck<float>("inplace_add_f32_alpha_0",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT,
        0.0f, {1.0f, 2.0f, 3.0f}));

    countResult(RunInplaceAddTypedAlphaAndCheck<int32_t, int32_t>("inplace_add_i32_alpha_int_neg",
        {10, -4, 7, 0}, {4}, ACL_INT32,
        {1, 2, -3, 9}, {4}, ACL_INT32,
        -2, ACL_INT32, {8, -8, 13, -18}));

    // ================================================================
    // 14. aclnnInplaceAdds: 基本测试
    // ================================================================
    countResult(RunInplaceAddsAndCheck<float>("inplace_adds_f32",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        5.0f, 1.0f, {6.0f, 7.0f, 8.0f}));

    countResult(RunInplaceAddsAndCheck<float>("inplace_adds_f32_alpha_2",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        5.0f, 2.0f, {11.0f, 12.0f, 13.0f}));

    countResult(RunInplaceAddsAndCheck<float>("inplace_adds_f32_neg_alpha",
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT,
        5.0f, -1.0f, {5.0f, 15.0f, 25.0f}));

    countResult(RunInplaceAddsTypedScalarsAndCheck<int32_t, int32_t, int32_t>("inplace_adds_i32_typed_scalar_alpha",
        {1, 2, 3, 4, 5}, {5}, ACL_INT32,
        3, ACL_INT32, 2, ACL_INT32,
        {7, 8, 9, 10, 11}));

    // ================================================================
    // 15. aclnnInplaceAdds: 不同 dtype
    // ================================================================
    countResult(ProbeAddsDtypeCombo("inplace_adds_i32",
        {3}, ACL_INT32, 5.0f, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_INT32, true));

    // ================================================================
    // 16. aclnnAddV3: 基本测试
    // ================================================================
    countResult(RunAddV3AndCheck<float>("addv3_f32_alpha_ne_1",
        10.0f, ACL_FLOAT, {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        0.5f, {10.5f, 11.0f, 11.5f}));

    countResult(RunAddV3AndCheck<float>("addv3_f32_alpha_eq_1",
        5.0f, ACL_FLOAT, {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {6.0f, 7.0f, 8.0f}));

    countResult(RunAddV3AndCheck<float>("addv3_f32_alpha_0",
        100.0f, ACL_FLOAT, {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        0.0f, {100.0f, 100.0f, 100.0f}));

    countResult(RunAddV3TypedAndCheck<int32_t, int32_t, float>("addv3_i32_exec_typed_self",
        5, ACL_INT32, {1, 2, 3}, {3}, ACL_INT32, ACL_INT32,
        1.0f, ACL_FLOAT, {6, 7, 8}));

    countResult(RunAddV3TypedAndCheck<int32_t, int32_t, int32_t>("addv3_i32_alpha_int2_exec",
        5, ACL_INT32, {1, 2, 3}, {3}, ACL_INT32, ACL_INT32,
        2, ACL_INT32, {7, 9, 11}));

    // ================================================================
    // 17. aclnnAddV3: 不同 dtype
    // ================================================================
    countResult(ProbeAddV3DtypeCombo("addv3_f16_basic",
        5.0f, ACL_FLOAT, {4}, ACL_FLOAT16, ACL_FLOAT, 1.0f, true));

    countResult(ProbeAddV3DtypeCombo("addv3_bf16_basic",
        5.0f, ACL_FLOAT, {4}, ACL_BF16, ACL_FLOAT, 1.0f, true));

    countResult(ProbeAddV3DtypeCombo("addv3_i32_basic",
        5.0f, ACL_INT32, {4}, ACL_INT32, ACL_INT32, 1.0f, true));

    countResult(ProbeAddV3DtypeCombo("addv3_f16_alpha_ne_1",
        5.0f, ACL_FLOAT, {4}, ACL_FLOAT16, ACL_FLOAT, 2.0f, true));

    countResult(ProbeAddV3DtypeCombo("addv3_i8_basic",
        3, ACL_INT32, {4}, ACL_INT8, ACL_INT8, 1.0f, true));

    countResult(ProbeAddV3TypedScalars("addv3_i8_alpha2_mul_add_path",
        3, ACL_INT32, {4}, ACL_INT8, ACL_INT8, 2.0f, ACL_FLOAT, true));

    countResult(ProbeAddV3TypedScalars("addv3_self_double_out_float",
        3.0, ACL_DOUBLE, {4}, ACL_INT32, ACL_FLOAT, 1.0f, ACL_FLOAT, true));

    countResult(ProbeAddV3TypedScalars("addv3_bool_other_unsupported",
        true, ACL_BOOL, {4}, ACL_BOOL, ACL_BOOL, true, ACL_BOOL, false));

    countResult(ProbeAddV3TypedScalars("addv3_double_other_unsupported",
        1.0, ACL_DOUBLE, {4}, ACL_DOUBLE, ACL_DOUBLE, 1.0f, ACL_FLOAT, false));

    {
        using Complex64 = std::complex<float>;
        const std::vector<float> f32 = {1.0f, 2.0f};
        const std::vector<int32_t> i32 = {1, 2};
        const std::vector<Complex64> c64 = {Complex64(1.0f, 2.0f), Complex64(3.0f, -4.0f)};
        const std::vector<Complex64> outC64(2);
        const std::vector<float> outF32(2, 0.0f);
        const std::vector<int32_t> outI32(2, 0);
        const Complex64 selfC64(1.0f, -2.0f);
        const Complex64 alphaC64(1.0f, 1.0f);

        countResult(ProbeAddV3TypedTensorScalar<float, Complex64, Complex64, float>(
            "addv3_self_complex64_other_float", selfC64, ACL_COMPLEX64,
            f32, {2}, ACL_FLOAT, outC64, ACL_COMPLEX64, 1.0f, ACL_FLOAT, true));

        countResult(ProbeAddV3TypedTensorScalar<int32_t, float, float, float>(
            "addv3_self_float_other_i32_promote_float", 1.5f, ACL_FLOAT,
            i32, {2}, ACL_INT32, outF32, ACL_FLOAT, 1.0f, ACL_FLOAT, true));

        countResult(ProbeAddV3TypedTensorScalar<float, float, float, Complex64>(
            "addv3_alpha_complex_to_float_invalid", 1.0f, ACL_FLOAT,
            f32, {2}, ACL_FLOAT, outF32, ACL_FLOAT, alphaC64, ACL_COMPLEX64, false));

        countResult(ProbeAddV3TypedTensorScalar<float, float, Complex64, float>(
            "addv3_self_complex64_to_float_invalid", selfC64, ACL_COMPLEX64,
            f32, {2}, ACL_FLOAT, outF32, ACL_FLOAT, 1.0f, ACL_FLOAT, false));

        countResult(ProbeAddV3TypedTensorScalar<float, int32_t, float, float>(
            "addv3_float_to_i32_out_probe", 1.0f, ACL_FLOAT,
            f32, {2}, ACL_FLOAT, outI32, ACL_INT32, 1.0f, ACL_FLOAT, false));

        countResult(ProbeAddV3TypedTensorScalar<Complex64, Complex64, float, float>(
            "addv3_complex64_other_unsupported", 1.0f, ACL_FLOAT,
            c64, {2}, ACL_COMPLEX64, outC64, ACL_COMPLEX64, 1.0f, ACL_FLOAT, false));
    }

    // ================================================================
    // 18. aclnnInplaceAddV3: 基本测试
    // ================================================================
    countResult(RunInplaceAddV3AndCheck<float>("inplace_addv3_f32",
        10.0f, ACL_FLOAT, {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        1.0f, {11.0f, 12.0f, 13.0f}));

    countResult(RunInplaceAddV3AndCheck<float>("inplace_addv3_f32_alpha_ne_1",
        10.0f, ACL_FLOAT, {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        2.0f, {12.0f, 14.0f, 16.0f}));

    countResult(RunInplaceAddV3TypedAndCheck<int32_t, int32_t, float>("inplace_addv3_i32_exec_typed_self",
        5, ACL_INT32, {1, 2, 3}, {3}, ACL_INT32,
        1.0f, ACL_FLOAT, {6, 7, 8}));

    countResult(RunInplaceAddV3TypedAndCheck<int32_t, int32_t, int32_t>("inplace_addv3_i32_alpha_int2_exec",
        5, ACL_INT32, {1, 2, 3}, {3}, ACL_INT32,
        2, ACL_INT32, {7, 9, 11}));

    countResult(ProbeInplaceAddV3TypedScalars("inplace_addv3_bool_other_unsupported",
        true, ACL_BOOL, {4}, ACL_BOOL, true, ACL_BOOL, false));

    countResult(ProbeInplaceAddV3TypedScalars("inplace_addv3_double_other_unsupported",
        1.0, ACL_DOUBLE, {4}, ACL_DOUBLE, 1.0f, ACL_FLOAT, false));

    // ================================================================
    // 19. 空 tensor (0元素) 测试 - 覆盖各种 API
    // ================================================================
    countResult(ProbeEmptyTensor("empty_add", 0));
    countResult(ProbeEmptyTensor("empty_adds", 1));
    countResult(ProbeEmptyTensor("empty_inplace_add", 2));
    countResult(ProbeEmptyTensor("empty_inplace_adds", 3));
    countResult(ProbeEmptyTensor("empty_addv3", 4));
    countResult(ProbeEmptyTensor("empty_inplace_addv3", 5));

    // ================================================================
    // 20. null 指针检查 - 覆盖所有 API 的参数校验路径
    // ================================================================
    countResult(ProbeNullptrTest("nullptr_add_self", 0));
    countResult(ProbeNullptrTest("nullptr_add_other", 1));
    countResult(ProbeNullptrTest("nullptr_add_alpha", 2));
    countResult(ProbeNullptrTest("nullptr_add_out", 3));
    countResult(ProbeNullptrTest("nullptr_adds_self", 4));
    countResult(ProbeNullptrTest("nullptr_adds_other", 5));
    countResult(ProbeNullptrTest("nullptr_adds_alpha", 6));
    countResult(ProbeNullptrTest("nullptr_adds_out", 7));
    countResult(ProbeNullptrTest("nullptr_inplace_add_self", 8));
    countResult(ProbeNullptrTest("nullptr_inplace_add_other", 9));
    countResult(ProbeNullptrTest("nullptr_inplace_adds_self", 10));
    countResult(ProbeNullptrTest("nullptr_inplace_adds_other", 11));
    countResult(ProbeNullptrTest("nullptr_inplace_adds_alpha", 12));
    countResult(ProbeNullptrTest("nullptr_addv3_self", 13));
    countResult(ProbeNullptrTest("nullptr_addv3_other", 14));
    countResult(ProbeNullptrTest("nullptr_addv3_alpha", 15));
    countResult(ProbeNullptrTest("nullptr_addv3_out", 16));
    countResult(ProbeNullptrTest("nullptr_inplace_addv3_self", 17));
    countResult(ProbeNullptrTest("nullptr_inplace_addv3_other", 18));
    countResult(ProbeNullptrTest("nullptr_inplace_addv3_alpha", 19));

    // ================================================================
    // 21. 非法广播 shape 测试
    // ================================================================
    countResult(ProbeInvalidBroadcast("invalid_broadcast_shape"));
    countResult(ProbeInvalidOutShape("invalid_add_out_shape"));
    countResult(ProbeMaxDimTensor("invalid_add_max_dim"));
    countResult(ProbeInvalidInplaceBroadcast("invalid_inplace_broadcast_shape"));
    countResult(ProbeInvalidAddV3OutShape("invalid_addv3_out_shape"));
    countResult(ProbeMaxDimAddV3("invalid_addv3_max_dim"));

    // ================================================================
    // 22. 大 shape 测试
    // ================================================================
    {
        std::vector<float> bigSelf(1024, 1.0f);
        std::vector<float> bigOther(1024, 2.0f);
        std::vector<float> bigExpected(1024, 5.0f);
        countResult(RunAddAndCheck<float>("add_f32_big_1024",
            bigSelf, {1024}, ACL_FLOAT,
            bigOther, {1024}, ACL_FLOAT, ACL_FLOAT,
            2.0f, bigExpected, {1024}));
    }

    // ================================================================
    // 23. 多维 shape 测试
    // ================================================================
    countResult(RunAddAndCheck<float>("add_f32_4d",
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1, 1, 2, 3}, ACL_FLOAT,
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, {1, 1, 2, 3}, ACL_FLOAT, ACL_FLOAT,
        10.0f, {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f}, {1, 1, 2, 3}));

    countResult(RunAddAndCheck<float>("add_f32_8d_max_valid_dim",
        {1.0f, 2.0f}, {1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT,
        {3.0f, 4.0f}, {1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {4.0f, 6.0f}, {1, 1, 1, 1, 1, 1, 1, 2}));

    // ================================================================
    // 24. 标量 shape (空 dims) 测试
    // ================================================================
    countResult(RunAddAndCheck<float>("add_f32_scalar_shape",
        {5.0f}, {}, ACL_FLOAT,
        {3.0f}, {}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {8.0f}, {}));

    // ================================================================
    // 25. AddV3 多种 alpha 值
    // ================================================================
    countResult(RunAddV3AndCheck<float>("addv3_f32_alpha_large",
        0.0f, ACL_FLOAT, {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        100.0f, {100.0f, 200.0f, 300.0f}));

    countResult(RunAddV3AndCheck<float>("addv3_f32_alpha_neg",
        10.0f, ACL_FLOAT, {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        -1.0f, {9.0f, 8.0f, 7.0f}));

    // ================================================================
    // 26. Add with INT32 alpha 非 1（走 Axpy 路径）
    // ================================================================
    countResult(RunAddAndCheck<int32_t>("add_i32_alpha_2_axpy",
        {1, 2, 3}, {3}, ACL_INT32,
        {10, 20, 30}, {3}, ACL_INT32, ACL_INT32,
        2.0f, {21, 42, 63}, {3}));

    countResult(RunAddAndCheck<float>("add_f32_alpha_2_axpy",
        {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT,
        {10.0f, 20.0f, 30.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        2.0f, {21.0f, 42.0f, 63.0f}, {3}));

    // ================================================================
    // 27. InplaceAddV3 不同 dtype
    // ================================================================
    countResult(RunInplaceAddV3MixedNoCheck<uint16_t, float, float>("inplace_addv3_f16_exec",
        5.0f, ACL_FLOAT,
        {0x3C00, 0x4000, 0x4200, 0x4400}, {4}, ACL_FLOAT16,
        1.0f, ACL_FLOAT));

    countResult(RunInplaceAddV3MixedNoCheck<int8_t, int32_t, float>("inplace_addv3_i8_alpha2_exec",
        3, ACL_INT32,
        {1, 2, 3, 4}, {4}, ACL_INT8,
        2.0f, ACL_FLOAT));

    // ================================================================
    // 28. 边界值测试
    // ================================================================
    countResult(RunAddAndCheck<float>("add_f32_zeros",
        {0.0f, 0.0f, 0.0f}, {3}, ACL_FLOAT,
        {0.0f, 0.0f, 0.0f}, {3}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {0.0f, 0.0f, 0.0f}, {3}));

    countResult(RunAddAndCheck<float>("add_f32_large_vals",
        {1e10f, -1e10f}, {2}, ACL_FLOAT,
        {1.0f, -1.0f}, {2}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {1e10f + 1.0f, -1e10f - 1.0f}, {2}, 1.0f, 0.0f));

    countResult(RunAddAndCheck<float>("add_f32_very_small",
        {1e-30f, 2e-30f}, {2}, ACL_FLOAT,
        {1e-30f, 2e-30f}, {2}, ACL_FLOAT, ACL_FLOAT,
        1.0f, {2e-30f, 4e-30f}, {2}, 1e-35f, 0.0f));

    countResult(RunAddMixedNoCheck<float, float, float, float>("add_f32_precision_large_small_observation",
        {1e10f, 1e10f}, {2}, ACL_FLOAT,
        {1e-5f, 1e-5f}, {2}, ACL_FLOAT, ACL_FLOAT,
        1.0f, ACL_FLOAT, {2}));

    countResult(RunAddMixedNoCheck<float, float, float, float>("add_f32_precision_cancellation_observation",
        {1.0000001f, 2.0000001f}, {2}, ACL_FLOAT,
        {-1.0f, -2.0f}, {2}, ACL_FLOAT, ACL_FLOAT,
        1.0f, ACL_FLOAT, {2}));

    // ================================================================
    // 29. arch35 tiling dtype dispatch: 用实际二段执行触发 tiling 模板分支
    // ================================================================
    countResult(RunAddAndCheck<int8_t>("tiling_add_i8_exec",
        {1, 2, 3, 4}, {4}, ACL_INT8,
        {1, 1, 1, 1}, {4}, ACL_INT8, ACL_INT8,
        1.0f, {2, 3, 4, 5}, {4}));

    countResult(RunAddAndCheck<uint8_t>("tiling_add_u8_exec",
        {1, 2, 3, 4}, {4}, ACL_UINT8,
        {1, 1, 1, 1}, {4}, ACL_UINT8, ACL_UINT8,
        1.0f, {2, 3, 4, 5}, {4}));

    countResult(RunAddMixedNoCheck<uint8_t, uint8_t, uint8_t, bool>("tiling_add_bool_exec",
        {1, 0, 1, 0}, {4}, ACL_BOOL,
        {1, 1, 0, 0}, {4}, ACL_BOOL, ACL_BOOL,
        true, ACL_BOOL, {4}));

    countResult(RunAddMixedNoCheck<uint16_t, float, float, float>("tiling_add_mixed_f16_f32_exec",
        {0x3C00, 0x4000, 0x4200, 0x4400}, {4}, ACL_FLOAT16,
        {1.0f, 1.0f, 1.0f, 1.0f}, {4}, ACL_FLOAT, ACL_FLOAT,
        1.0f, ACL_FLOAT, {4}));

    countResult(RunAddMixedNoCheck<float, uint16_t, float, float>("tiling_add_mixed_f32_bf16_exec",
        {1.0f, 2.0f, 3.0f, 4.0f}, {4}, ACL_FLOAT,
        {0x3F80, 0x4000, 0x4040, 0x4080}, {4}, ACL_BF16, ACL_FLOAT,
        1.0f, ACL_FLOAT, {4}));

    countResult(RunAddMixedNoCheck<float, uint16_t, float, float>("tiling_add_mixed_f32_f16_exec",
        {1.0f, 2.0f, 3.0f, 4.0f}, {4}, ACL_FLOAT,
        {0x3C00, 0x4000, 0x4200, 0x4400}, {4}, ACL_FLOAT16, ACL_FLOAT,
        1.0f, ACL_FLOAT, {4}));

    countResult(RunAddMixedNoCheck<uint16_t, float, float, float>("tiling_add_mixed_bf16_f32_exec",
        {0x3F80, 0x4000, 0x4040, 0x4080}, {4}, ACL_BF16,
        {1.0f, 1.0f, 1.0f, 1.0f}, {4}, ACL_FLOAT, ACL_FLOAT,
        1.0f, ACL_FLOAT, {4}));

    std::cout << "summary: pass=" << stats.pass << " fail=" << stats.fail << std::endl;

    aclrtResetDevice(kDeviceId);
    aclFinalize();

    return stats.fail == 0 ? 0 : 1;
}
