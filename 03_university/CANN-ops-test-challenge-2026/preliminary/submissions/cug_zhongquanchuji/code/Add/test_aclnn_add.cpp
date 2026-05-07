/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/op_errno.h"

namespace {

inline void ReportOpApiLoadFailure(const char *symbol, const char *detail)
{
    std::fprintf(stderr, "[opapi-loader] %s failed: %s\n", symbol, detail == nullptr ? "unknown error" : detail);
}

inline void *LoadOpApiMathHandle()
{
    static void *handle = []() -> void * {
        std::vector<std::string> candidates;
        const char *customOppPath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
        if (customOppPath != nullptr && customOppPath[0] != '\0') {
            candidates.emplace_back(std::string(customOppPath) + "/libopapi_math.so");
        }
        candidates.emplace_back("./libopapi_math.so");
        candidates.emplace_back("libopapi_math.so");

        for (const auto &path : candidates) {
            void *loaded = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (loaded != nullptr) {
                return loaded;
            }
        }
        return nullptr;
    }();
    return handle;
}

template <typename Fn>
Fn ResolveOpApiMathSymbol(const char *symbol)
{
    void *handle = LoadOpApiMathHandle();
    if (handle == nullptr) {
        ReportOpApiLoadFailure(symbol, dlerror());
        return nullptr;
    }
    dlerror();
    void *resolved = dlsym(handle, symbol);
    const char *error = dlerror();
    if (error != nullptr) {
        ReportOpApiLoadFailure(symbol, error);
        return nullptr;
    }
    return reinterpret_cast<Fn>(resolved);
}

}  // namespace

extern "C" aclnnStatus aclnnAddGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *other,
    const aclScalar *alpha,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    using Fn = aclnnStatus (*)(
        const aclTensor *, const aclTensor *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnAddGetWorkspaceSize");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(self, other, alpha, out, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    using Fn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnAdd");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnAddsGetWorkspaceSize(
    const aclTensor *self,
    const aclScalar *other,
    const aclScalar *alpha,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    using Fn =
        aclnnStatus (*)(const aclTensor *, const aclScalar *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnAddsGetWorkspaceSize");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(self, other, alpha, out, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnAdds(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    using Fn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnAdds");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnInplaceAddGetWorkspaceSize(
    const aclTensor *selfRef, const aclTensor *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    using Fn =
        aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclScalar *, uint64_t *, aclOpExecutor **);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnInplaceAddGetWorkspaceSize");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(selfRef, other, alpha, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnInplaceAdd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    using Fn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnInplaceAdd");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnInplaceAddsGetWorkspaceSize(
    const aclTensor *selfRef, const aclScalar *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    using Fn = aclnnStatus (*)(const aclTensor *, const aclScalar *, const aclScalar *, uint64_t *, aclOpExecutor **);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnInplaceAddsGetWorkspaceSize");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(selfRef, other, alpha, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnInplaceAdds(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    using Fn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnInplaceAdds");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnAddV3GetWorkspaceSize(
    const aclScalar *self,
    const aclTensor *other,
    const aclScalar *alpha,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    using Fn =
        aclnnStatus (*)(const aclScalar *, const aclTensor *, const aclScalar *, aclTensor *, uint64_t *, aclOpExecutor **);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnAddV3GetWorkspaceSize");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(self, other, alpha, out, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnAddV3(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    using Fn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnAddV3");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnInplaceAddV3GetWorkspaceSize(
    const aclScalar *selfRef, const aclTensor *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    using Fn = aclnnStatus (*)(const aclScalar *, const aclTensor *, const aclScalar *, uint64_t *, aclOpExecutor **);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnInplaceAddV3GetWorkspaceSize");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(selfRef, other, alpha, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnInplaceAddV3(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    using Fn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    static Fn fn = ResolveOpApiMathSymbol<Fn>("aclnnInplaceAddV3");
    return fn == nullptr ? ACLNN_ERR_INNER : fn(workspace, workspaceSize, executor, stream);
}

namespace add_example {

int g_totalCases = 0;
int g_failedCases = 0;

struct Fp16 {
    aclFloat16 bits = 0;

    Fp16() = default;

    explicit Fp16(float value) : bits(aclFloatToFloat16(value))
    {
    }

    template <typename T, typename = std::enable_if_t<std::is_integral<T>::value> >
    explicit Fp16(T value) : bits(aclFloatToFloat16(static_cast<float>(value)))
    {
    }

    operator float() const
    {
        return aclFloat16ToFloat(bits);
    }
};

inline uint16_t FloatToBf16Bits(float value)
{
    uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    if (std::isnan(value)) {
        return 0x7FC0U;
    }
    const uint32_t lsb = (raw >> 16U) & 1U;
    raw += 0x7FFFU + lsb;
    return static_cast<uint16_t>(raw >> 16U);
}

inline float Bf16BitsToFloat(uint16_t bits)
{
    const uint32_t raw = static_cast<uint32_t>(bits) << 16U;
    float result = 0.0f;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

struct Bf16 {
    uint16_t bits = 0;

    Bf16() = default;

    explicit Bf16(float value) : bits(FloatToBf16Bits(value))
    {
    }

    template <typename T, typename = std::enable_if_t<std::is_integral<T>::value> >
    explicit Bf16(T value) : bits(FloatToBf16Bits(static_cast<float>(value)))
    {
    }

    operator float() const
    {
        return Bf16BitsToFloat(bits);
    }
};

static_assert(sizeof(Fp16) == sizeof(aclFloat16), "Fp16 wrapper size mismatch");
static_assert(sizeof(Bf16) == sizeof(uint16_t), "Bf16 wrapper size mismatch");

inline void Log(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

inline std::string RecentAclError()
{
    const char *message = aclGetRecentErrMsg();
    return message == nullptr ? std::string() : std::string(message);
}

inline int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}

inline std::vector<int64_t> ComputeContiguousStrides(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

struct RuntimeContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool ready = false;

    bool Init()
    {
        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            Log("aclInit failed. ERROR: %d\n", ret);
            return false;
        }
        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            Log("aclrtSetDevice failed. ERROR: %d\n", ret);
            aclFinalize();
            return false;
        }
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            Log("aclrtCreateStream failed. ERROR: %d\n", ret);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return false;
        }
        ready = true;
        return true;
    }

    ~RuntimeContext()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (ready) {
            aclrtResetDevice(deviceId);
            aclFinalize();
        }
    }
};

struct DeviceBuffer {
    void *addr = nullptr;

    ~DeviceBuffer()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }
};

struct TensorHolder {
    aclTensor *tensor = nullptr;
    DeviceBuffer buffer;

    ~TensorHolder()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
};

struct ScalarHolder {
    aclScalar *scalar = nullptr;

    ~ScalarHolder()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
        }
    }
};

struct ExecutorHolder {
    aclOpExecutor *executor = nullptr;

    ~ExecutorHolder()
    {
        executor = nullptr;
    }
};

struct TensorSpec {
    std::vector<int64_t> viewShape;
    std::vector<int64_t> storageShape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    aclDataType dataType = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
};

inline std::vector<int64_t> ResolvedStorageShape(const TensorSpec &spec)
{
    return spec.storageShape.empty() ? spec.viewShape : spec.storageShape;
}

inline std::vector<int64_t> ResolvedStrides(const TensorSpec &spec)
{
    if (!spec.strides.empty()) {
        return spec.strides;
    }
    const auto storageShape = ResolvedStorageShape(spec);
    return ComputeContiguousStrides(storageShape.empty() ? spec.viewShape : storageShape);
}

inline size_t RequiredStorageElements(const TensorSpec &spec)
{
    if (ShapeSize(spec.viewShape) == 0) {
        return 0;
    }
    const auto strides = ResolvedStrides(spec);
    size_t lastOffset = static_cast<size_t>(spec.offset);
    for (size_t i = 0; i < spec.viewShape.size(); ++i) {
        lastOffset += static_cast<size_t>((spec.viewShape[i] - 1) * strides[i]);
    }
    return lastOffset + 1;
}

template <typename T>
bool CreateTensor(const std::string &name, const std::vector<T> &hostData, const TensorSpec &spec, TensorHolder *holder)
{
    const size_t requiredElements = RequiredStorageElements(spec);
    if (hostData.size() < requiredElements) {
        Log("[%s] host data size is %zu, but tensor view needs at least %zu elements.\n",
            name.c_str(),
            hostData.size(),
            requiredElements);
        return false;
    }

    const size_t hostBytes = hostData.size() * sizeof(T);
    const size_t allocBytes = hostBytes == 0 ? 1U : hostBytes;
    auto ret = aclrtMalloc(&holder->buffer.addr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        Log("[%s] aclrtMalloc failed. ERROR: %d\n", name.c_str(), ret);
        return false;
    }

    if (hostBytes > 0) {
        ret = aclrtMemcpy(holder->buffer.addr, hostBytes, hostData.data(), hostBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            Log("[%s] aclrtMemcpy failed. ERROR: %d\n", name.c_str(), ret);
            return false;
        }
    }

    const auto storageShape = ResolvedStorageShape(spec);
    const auto strides = ResolvedStrides(spec);
    holder->tensor = aclCreateTensor(spec.viewShape.data(),
                                     spec.viewShape.size(),
                                     spec.dataType,
                                     strides.empty() ? nullptr : strides.data(),
                                     spec.offset,
                                     spec.format,
                                     storageShape.data(),
                                     storageShape.size(),
                                     holder->buffer.addr);
    if (holder->tensor == nullptr) {
        Log("[%s] aclCreateTensor failed.\n", name.c_str());
        return false;
    }
    return true;
}

inline const void *ScalarRawPtr(const bool &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const int8_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const uint8_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const int16_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const uint16_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const int32_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const uint32_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const int64_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const uint64_t &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const float &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const double &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const Fp16 &value)
{
    return &value.bits;
}

inline const void *ScalarRawPtr(const Bf16 &value)
{
    return &value.bits;
}

inline const void *ScalarRawPtr(const std::complex<float> &value)
{
    return &value;
}

inline const void *ScalarRawPtr(const std::complex<double> &value)
{
    return &value;
}

template <typename T>
bool CreateScalar(const std::string &name, const T &value, aclDataType dataType, ScalarHolder *holder)
{
    holder->scalar = aclCreateScalar(const_cast<void *>(ScalarRawPtr(value)), dataType);
    if (holder->scalar == nullptr) {
        Log("[%s] aclCreateScalar failed.\n", name.c_str());
        return false;
    }
    return true;
}

template <typename T>
bool CopyTensorToHost(const std::string &name, const TensorHolder &holder, size_t elementCount, std::vector<T> *hostData)
{
    hostData->assign(elementCount, T());
    if (elementCount == 0) {
        hostData->clear();
        return true;
    }

    auto ret = aclrtMemcpy(hostData->data(),
                           elementCount * sizeof(T),
                           holder.buffer.addr,
                           elementCount * sizeof(T),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        Log("[%s] copy result from device to host failed. ERROR: %d\n", name.c_str(), ret);
        return false;
    }
    return true;
}

inline int64_t ViewOffsetFromFlatIndex(size_t flatIndex,
                                       const std::vector<int64_t> &shape,
                                       const std::vector<int64_t> &strides,
                                       int64_t offset)
{
    int64_t viewOffset = offset;
    if (shape.empty()) {
        return viewOffset;
    }
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0; --axis) {
        const int64_t dim = shape[static_cast<size_t>(axis)];
        const int64_t coord = static_cast<int64_t>(flatIndex % static_cast<size_t>(dim));
        flatIndex /= static_cast<size_t>(dim);
        viewOffset += coord * strides[static_cast<size_t>(axis)];
    }
    return viewOffset;
}

template <typename T>
std::vector<T> MaterializeView(const TensorSpec &spec, const std::vector<T> &storageData)
{
    std::vector<T> materialized;
    const size_t total = static_cast<size_t>(ShapeSize(spec.viewShape));
    materialized.reserve(total);
    const auto strides = ResolvedStrides(spec);
    for (size_t flatIndex = 0; flatIndex < total; ++flatIndex) {
        const int64_t offset = ViewOffsetFromFlatIndex(flatIndex, spec.viewShape, strides, spec.offset);
        materialized.push_back(storageData[static_cast<size_t>(offset)]);
    }
    return materialized;
}

inline bool BroadcastShape(const std::vector<int64_t> &shapeA,
                           const std::vector<int64_t> &shapeB,
                           std::vector<int64_t> *outShape)
{
    const size_t rank = std::max(shapeA.size(), shapeB.size());
    outShape->assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dimA = i < rank - shapeA.size() ? 1 : shapeA[i - (rank - shapeA.size())];
        const int64_t dimB = i < rank - shapeB.size() ? 1 : shapeB[i - (rank - shapeB.size())];
        if (dimA != dimB && dimA != 1 && dimB != 1) {
            return false;
        }
        (*outShape)[i] = std::max(dimA, dimB);
    }
    return true;
}

inline size_t BroadcastFlatIndex(size_t outFlatIndex,
                                 const std::vector<int64_t> &outShape,
                                 const std::vector<int64_t> &inShape,
                                 const std::vector<int64_t> &inStrides)
{
    if (inShape.empty()) {
        return 0;
    }
    size_t inFlatIndex = 0;
    for (int64_t outAxis = static_cast<int64_t>(outShape.size()) - 1; outAxis >= 0; --outAxis) {
        const int64_t outDim = outShape[static_cast<size_t>(outAxis)];
        const int64_t coord = static_cast<int64_t>(outFlatIndex % static_cast<size_t>(outDim));
        outFlatIndex /= static_cast<size_t>(outDim);

        const int64_t inAxis = outAxis - static_cast<int64_t>(outShape.size() - inShape.size());
        if (inAxis < 0) {
            continue;
        }
        const int64_t inDim = inShape[static_cast<size_t>(inAxis)];
        const int64_t inCoord = inDim == 1 ? 0 : coord;
        inFlatIndex += static_cast<size_t>(inCoord * inStrides[static_cast<size_t>(inAxis)]);
    }
    return inFlatIndex;
}

template <typename T>
T ZeroValue()
{
    return T();
}

template <>
inline std::complex<float> ZeroValue<std::complex<float> >()
{
    return std::complex<float>(0.0f, 0.0f);
}

template <>
inline std::complex<double> ZeroValue<std::complex<double> >()
{
    return std::complex<double>(0.0, 0.0);
}

template <typename T>
std::vector<T> MakeStorage(const TensorSpec &spec)
{
    return std::vector<T>(RequiredStorageElements(spec), ZeroValue<T>());
}

template <typename T>
std::vector<T> MakeOutputInit(const TensorSpec &spec)
{
    return std::vector<T>(static_cast<size_t>(ShapeSize(spec.viewShape)), ZeroValue<T>());
}

template <typename OutT, typename LeftT, typename RightT, typename Op>
std::vector<OutT> ComputeBroadcastExpected(const TensorSpec &leftSpec,
                                           const std::vector<LeftT> &leftStorage,
                                           const TensorSpec &rightSpec,
                                           const std::vector<RightT> &rightStorage,
                                           Op op)
{
    const auto left = MaterializeView(leftSpec, leftStorage);
    const auto right = MaterializeView(rightSpec, rightStorage);
    std::vector<int64_t> outShape;
    BroadcastShape(leftSpec.viewShape, rightSpec.viewShape, &outShape);
    const auto leftStrides = ComputeContiguousStrides(leftSpec.viewShape);
    const auto rightStrides = ComputeContiguousStrides(rightSpec.viewShape);

    std::vector<OutT> expected;
    expected.reserve(static_cast<size_t>(ShapeSize(outShape)));
    for (size_t i = 0; i < static_cast<size_t>(ShapeSize(outShape)); ++i) {
        const size_t leftIndex = BroadcastFlatIndex(i, outShape, leftSpec.viewShape, leftStrides);
        const size_t rightIndex = BroadcastFlatIndex(i, outShape, rightSpec.viewShape, rightStrides);
        expected.push_back(op(left[leftIndex], right[rightIndex]));
    }
    return expected;
}

template <typename OutT, typename LeftT, typename ScalarT, typename Op>
std::vector<OutT> ComputeScalarExpected(const TensorSpec &leftSpec,
                                        const std::vector<LeftT> &leftStorage,
                                        const ScalarT &scalar,
                                        Op op)
{
    const auto left = MaterializeView(leftSpec, leftStorage);
    std::vector<OutT> expected;
    expected.reserve(left.size());
    for (size_t i = 0; i < left.size(); ++i) {
        expected.push_back(op(left[i], scalar));
    }
    return expected;
}

inline bool CloseFloating(double actual, double expected, double atol, double rtol)
{
    if (std::isnan(actual) && std::isnan(expected)) {
        return true;
    }
    if (std::isinf(actual) || std::isinf(expected)) {
        return actual == expected;
    }
    return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

template <typename T>
std::string ValueToString(const T &value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

template <>
inline std::string ValueToString<int8_t>(const int8_t &value)
{
    return std::to_string(static_cast<int>(value));
}

template <>
inline std::string ValueToString<uint8_t>(const uint8_t &value)
{
    return std::to_string(static_cast<unsigned int>(value));
}

template <>
inline std::string ValueToString<Fp16>(const Fp16 &value)
{
    std::ostringstream stream;
    stream << static_cast<float>(value);
    return stream.str();
}

template <>
inline std::string ValueToString<Bf16>(const Bf16 &value)
{
    std::ostringstream stream;
    stream << static_cast<float>(value);
    return stream.str();
}

template <>
inline std::string ValueToString<std::complex<float> >(const std::complex<float> &value)
{
    std::ostringstream stream;
    stream << "(" << value.real() << ", " << value.imag() << ")";
    return stream.str();
}

template <>
inline std::string ValueToString<std::complex<double> >(const std::complex<double> &value)
{
    std::ostringstream stream;
    stream << "(" << value.real() << ", " << value.imag() << ")";
    return stream.str();
}

template <typename T>
bool ElementsEqual(const T &actual, const T &expected)
{
    return actual == expected;
}

template <>
inline bool ElementsEqual<float>(const float &actual, const float &expected)
{
    return CloseFloating(actual, expected, 1e-6, 1e-6);
}

template <>
inline bool ElementsEqual<double>(const double &actual, const double &expected)
{
    return CloseFloating(actual, expected, 1e-12, 1e-12);
}

template <>
inline bool ElementsEqual<Fp16>(const Fp16 &actual, const Fp16 &expected)
{
    return actual.bits == expected.bits;
}

template <>
inline bool ElementsEqual<Bf16>(const Bf16 &actual, const Bf16 &expected)
{
    return actual.bits == expected.bits;
}

template <>
inline bool ElementsEqual<std::complex<float> >(const std::complex<float> &actual, const std::complex<float> &expected)
{
    return CloseFloating(actual.real(), expected.real(), 1e-5, 1e-5) &&
           CloseFloating(actual.imag(), expected.imag(), 1e-5, 1e-5);
}

template <>
inline bool ElementsEqual<std::complex<double> >(const std::complex<double> &actual, const std::complex<double> &expected)
{
    return CloseFloating(actual.real(), expected.real(), 1e-12, 1e-12) &&
           CloseFloating(actual.imag(), expected.imag(), 1e-12, 1e-12);
}

template <typename T>
bool CompareVectors(const std::string &name, const std::vector<T> &actual, const std::vector<T> &expected)
{
    if (actual.size() != expected.size()) {
        Log("[%s] result size mismatch. actual=%zu expected=%zu\n", name.c_str(), actual.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!ElementsEqual(actual[i], expected[i])) {
            Log("[%s] mismatch at index %zu. actual=%s expected=%s\n",
                name.c_str(),
                i,
                ValueToString(actual[i]).c_str(),
                ValueToString(expected[i]).c_str());
            return false;
        }
    }
    return true;
}

inline bool ExpectStatus(const std::string &name, aclnnStatus actual, aclnnStatus expected)
{
    if (actual != expected) {
        Log("[%s] status mismatch. actual=%d expected=%d\n", name.c_str(), actual, expected);
        const std::string recentError = RecentAclError();
        if (!recentError.empty()) {
            Log("[%s] recent ACL error: %s\n", name.c_str(), recentError.c_str());
        }
        return false;
    }
    return true;
}

inline bool AllocateWorkspace(const std::string &name, uint64_t workspaceSize, DeviceBuffer *workspace)
{
    if (workspaceSize == 0) {
        return true;
    }
    const auto ret = aclrtMalloc(&workspace->addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        Log("[%s] allocate workspace failed. ERROR: %d\n", name.c_str(), ret);
        return false;
    }
    return true;
}

inline bool CheckWorkspaceZero(const std::string &name, uint64_t workspaceSize)
{
    if (workspaceSize != 0) {
        Log("[%s] workspace size should be 0 for empty tensor, but got %lu\n",
            name.c_str(),
            static_cast<unsigned long>(workspaceSize));
        return false;
    }
    return true;
}

inline bool IsKnownKernelGapStatus(aclnnStatus status)
{
    return status == ACLNN_ERR_INNER_NULLPTR || status == ACLNN_ERR_INNER;
}

inline bool HandleWorkspaceStatus(const std::string &name, aclnnStatus status, bool allowKernelGap)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    if (allowKernelGap && IsKnownKernelGapStatus(status)) {
        Log("[%s] skip execution because current simulator package lacks a matching kernel. status=%d\n",
            name.c_str(),
            status);
        const std::string recentError = RecentAclError();
        if (!recentError.empty()) {
            Log("[%s] recent ACL error: %s\n", name.c_str(), recentError.c_str());
        }
        return true;
    }
    return ExpectStatus(name, status, ACL_SUCCESS);
}

inline bool HandleExecutionStatus(const std::string &name, aclnnStatus status, bool allowKernelGap)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    if (allowKernelGap && IsKnownKernelGapStatus(status)) {
        Log("[%s] skip execution because current simulator package lacks a matching kernel. status=%d\n",
            name.c_str(),
            status);
        const std::string recentError = RecentAclError();
        if (!recentError.empty()) {
            Log("[%s] recent ACL error: %s\n", name.c_str(), recentError.c_str());
        }
        return true;
    }
    return ExpectStatus(name, status, ACL_SUCCESS);
}

inline int8_t NarrowingAddInt8(int8_t left, int8_t scaledRight)
{
    const int32_t sum = static_cast<int32_t>(left) + static_cast<int32_t>(scaledRight);
    return static_cast<int8_t>(static_cast<uint8_t>(static_cast<uint32_t>(sum)));
}

inline uint8_t NarrowingAddUInt8(uint8_t left, uint8_t scaledRight)
{
    return static_cast<uint8_t>(static_cast<uint32_t>(left) + static_cast<uint32_t>(scaledRight));
}

inline uint8_t BoolAdd(uint8_t left, uint8_t right)
{
    return static_cast<uint8_t>((left != 0U) || (right != 0U));
}

inline Fp16 Fp16Add(Fp16 left, Fp16 right)
{
    return Fp16(static_cast<float>(left) + static_cast<float>(right));
}

inline Bf16 Bf16Add(Bf16 left, Bf16 right)
{
    return Bf16(static_cast<float>(left) + static_cast<float>(right));
}

inline float Fp16FloatAdd(Fp16 left, float right)
{
    return static_cast<float>(left) + right;
}

inline float FloatFp16Add(float left, Fp16 right)
{
    return left + static_cast<float>(right);
}

inline float Bf16FloatAdd(Bf16 left, float right)
{
    return static_cast<float>(left) + right;
}

inline float FloatBf16Add(float left, Bf16 right)
{
    return left + static_cast<float>(right);
}

template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddExecuteCase(RuntimeContext &runtime,
                       const std::string &name,
                       const TensorSpec &selfSpec,
                       const std::vector<SelfT> &selfData,
                       const TensorSpec &otherSpec,
                       const std::vector<OtherT> &otherData,
                       const AlphaT &alphaValue,
                       aclDataType alphaType,
                       const TensorSpec &outSpec,
                       const std::vector<OutT> &expected,
                       bool allowKernelGap = true)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    if (!CreateTensor(name + "/self", selfData, selfSpec, &self) ||
        !CreateTensor(name + "/other", otherData, otherSpec, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha) ||
        !CreateTensor(name + "/out", expected, outSpec, &out)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret =
        aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor.executor);
    if (!HandleWorkspaceStatus(name + "/GetWorkspaceSize", ret, allowKernelGap)) {
        return false;
    }
    if (ret != ACL_SUCCESS) {
        return true;
    }

    DeviceBuffer workspace;
    if (!AllocateWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    const auto executeRet = aclnnAdd(workspace.addr, workspaceSize, executor.executor, runtime.stream);
    if (!HandleExecutionStatus(name + "/Execute", executeRet, allowKernelGap)) {
        return false;
    }
    if (executeRet != ACL_SUCCESS) {
        return true;
    }
    const auto syncRet = aclrtSynchronizeStream(runtime.stream);
    if (!HandleExecutionStatus(name + "/Synchronize", syncRet, allowKernelGap)) {
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        return true;
    }

    std::vector<OutT> actual;
    if (!CopyTensorToHost(name, out, static_cast<size_t>(ShapeSize(outSpec.viewShape)), &actual)) {
        return false;
    }
    return CompareVectors(name, actual, expected);
}

template <typename SelfT, typename ScalarT, typename AlphaT, typename OutT>
bool RunAddsExecuteCase(RuntimeContext &runtime,
                        const std::string &name,
                        const TensorSpec &selfSpec,
                        const std::vector<SelfT> &selfData,
                        const ScalarT &otherValue,
                        aclDataType otherType,
                        const AlphaT &alphaValue,
                        aclDataType alphaType,
                        const TensorSpec &outSpec,
                        const std::vector<OutT> &expected,
                        bool allowKernelGap = true)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;
    if (!CreateTensor(name + "/self", selfData, selfSpec, &self) ||
        !CreateScalar(name + "/other", otherValue, otherType, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha) ||
        !CreateTensor(name + "/out", expected, outSpec, &out)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret =
        aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor.executor);
    if (!HandleWorkspaceStatus(name + "/GetWorkspaceSize", ret, allowKernelGap)) {
        return false;
    }
    if (ret != ACL_SUCCESS) {
        return true;
    }

    DeviceBuffer workspace;
    if (!AllocateWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    const auto executeRet = aclnnAdds(workspace.addr, workspaceSize, executor.executor, runtime.stream);
    if (!HandleExecutionStatus(name + "/Execute", executeRet, allowKernelGap)) {
        return false;
    }
    if (executeRet != ACL_SUCCESS) {
        return true;
    }
    const auto syncRet = aclrtSynchronizeStream(runtime.stream);
    if (!HandleExecutionStatus(name + "/Synchronize", syncRet, allowKernelGap)) {
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        return true;
    }

    std::vector<OutT> actual;
    if (!CopyTensorToHost(name, out, static_cast<size_t>(ShapeSize(outSpec.viewShape)), &actual)) {
        return false;
    }
    return CompareVectors(name, actual, expected);
}

template <typename SelfT, typename OtherT, typename AlphaT>
bool RunInplaceAddExecuteCase(RuntimeContext &runtime,
                              const std::string &name,
                              const TensorSpec &selfSpec,
                              const std::vector<SelfT> &selfData,
                              const TensorSpec &otherSpec,
                              const std::vector<OtherT> &otherData,
                              const AlphaT &alphaValue,
                              aclDataType alphaType,
                              const std::vector<SelfT> &expected,
                              bool allowKernelGap = true)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    if (!CreateTensor(name + "/self", selfData, selfSpec, &self) ||
        !CreateTensor(name + "/other", otherData, otherSpec, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret =
        aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor.executor);
    if (!HandleWorkspaceStatus(name + "/GetWorkspaceSize", ret, allowKernelGap)) {
        return false;
    }
    if (ret != ACL_SUCCESS) {
        return true;
    }

    DeviceBuffer workspace;
    if (!AllocateWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    const auto executeRet = aclnnInplaceAdd(workspace.addr, workspaceSize, executor.executor, runtime.stream);
    if (!HandleExecutionStatus(name + "/Execute", executeRet, allowKernelGap)) {
        return false;
    }
    if (executeRet != ACL_SUCCESS) {
        return true;
    }
    const auto syncRet = aclrtSynchronizeStream(runtime.stream);
    if (!HandleExecutionStatus(name + "/Synchronize", syncRet, allowKernelGap)) {
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        return true;
    }

    std::vector<SelfT> actual;
    if (!CopyTensorToHost(name, self, static_cast<size_t>(ShapeSize(selfSpec.viewShape)), &actual)) {
        return false;
    }
    return CompareVectors(name, actual, expected);
}

template <typename SelfT, typename ScalarT, typename AlphaT>
bool RunInplaceAddsExecuteCase(RuntimeContext &runtime,
                               const std::string &name,
                               const TensorSpec &selfSpec,
                               const std::vector<SelfT> &selfData,
                               const ScalarT &otherValue,
                               aclDataType otherType,
                               const AlphaT &alphaValue,
                               aclDataType alphaType,
                               const std::vector<SelfT> &expected,
                               bool allowKernelGap = true)
{
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;
    if (!CreateTensor(name + "/self", selfData, selfSpec, &self) ||
        !CreateScalar(name + "/other", otherValue, otherType, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret =
        aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor.executor);
    if (!HandleWorkspaceStatus(name + "/GetWorkspaceSize", ret, allowKernelGap)) {
        return false;
    }
    if (ret != ACL_SUCCESS) {
        return true;
    }

    DeviceBuffer workspace;
    if (!AllocateWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    const auto executeRet = aclnnInplaceAdds(workspace.addr, workspaceSize, executor.executor, runtime.stream);
    if (!HandleExecutionStatus(name + "/Execute", executeRet, allowKernelGap)) {
        return false;
    }
    if (executeRet != ACL_SUCCESS) {
        return true;
    }
    const auto syncRet = aclrtSynchronizeStream(runtime.stream);
    if (!HandleExecutionStatus(name + "/Synchronize", syncRet, allowKernelGap)) {
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        return true;
    }

    std::vector<SelfT> actual;
    if (!CopyTensorToHost(name, self, static_cast<size_t>(ShapeSize(selfSpec.viewShape)), &actual)) {
        return false;
    }
    return CompareVectors(name, actual, expected);
}

template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddV3ExecuteCase(RuntimeContext &runtime,
                         const std::string &name,
                         const SelfScalarT &selfValue,
                         aclDataType selfType,
                         const TensorSpec &otherSpec,
                         const std::vector<OtherT> &otherData,
                         const AlphaT &alphaValue,
                         aclDataType alphaType,
                         const TensorSpec &outSpec,
                         const std::vector<OutT> &expected,
                         bool allowKernelGap = true)
{
    ScalarHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    TensorHolder out;
    if (!CreateScalar(name + "/self", selfValue, selfType, &self) ||
        !CreateTensor(name + "/other", otherData, otherSpec, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha) ||
        !CreateTensor(name + "/out", expected, outSpec, &out)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret =
        aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor.executor);
    if (!HandleWorkspaceStatus(name + "/GetWorkspaceSize", ret, allowKernelGap)) {
        return false;
    }
    if (ret != ACL_SUCCESS) {
        return true;
    }

    DeviceBuffer workspace;
    if (!AllocateWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    const auto executeRet = aclnnAddV3(workspace.addr, workspaceSize, executor.executor, runtime.stream);
    if (!HandleExecutionStatus(name + "/Execute", executeRet, allowKernelGap)) {
        return false;
    }
    if (executeRet != ACL_SUCCESS) {
        return true;
    }
    const auto syncRet = aclrtSynchronizeStream(runtime.stream);
    if (!HandleExecutionStatus(name + "/Synchronize", syncRet, allowKernelGap)) {
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        return true;
    }

    std::vector<OutT> actual;
    if (!CopyTensorToHost(name, out, static_cast<size_t>(ShapeSize(outSpec.viewShape)), &actual)) {
        return false;
    }
    return CompareVectors(name, actual, expected);
}

template <typename SelfScalarT, typename OtherT, typename AlphaT>
bool RunInplaceAddV3ExecuteCase(RuntimeContext &runtime,
                                const std::string &name,
                                const SelfScalarT &selfValue,
                                aclDataType selfType,
                                const TensorSpec &otherSpec,
                                const std::vector<OtherT> &otherData,
                                const AlphaT &alphaValue,
                                aclDataType alphaType,
                                const std::vector<OtherT> &expected,
                                bool allowKernelGap = true)
{
    ScalarHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    if (!CreateScalar(name + "/self", selfValue, selfType, &self) ||
        !CreateTensor(name + "/other", otherData, otherSpec, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret =
        aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor.executor);
    if (!HandleWorkspaceStatus(name + "/GetWorkspaceSize", ret, allowKernelGap)) {
        return false;
    }
    if (ret != ACL_SUCCESS) {
        return true;
    }

    DeviceBuffer workspace;
    if (!AllocateWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    const auto executeRet = aclnnInplaceAddV3(workspace.addr, workspaceSize, executor.executor, runtime.stream);
    if (!HandleExecutionStatus(name + "/Execute", executeRet, allowKernelGap)) {
        return false;
    }
    if (executeRet != ACL_SUCCESS) {
        return true;
    }
    const auto syncRet = aclrtSynchronizeStream(runtime.stream);
    if (!HandleExecutionStatus(name + "/Synchronize", syncRet, allowKernelGap)) {
        return false;
    }
    if (syncRet != ACL_SUCCESS) {
        return true;
    }

    std::vector<OtherT> actual;
    if (!CopyTensorToHost(name, other, static_cast<size_t>(ShapeSize(otherSpec.viewShape)), &actual)) {
        return false;
    }
    return CompareVectors(name, actual, expected);
}

template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddStatusCase(const std::string &name,
                      aclnnStatus expectedStatus,
                      const TensorSpec *selfSpec,
                      const std::vector<SelfT> *selfData,
                      const TensorSpec *otherSpec,
                      const std::vector<OtherT> *otherData,
                      const AlphaT *alphaValue,
                      aclDataType alphaType,
                      const TensorSpec *outSpec,
                      const std::vector<OutT> *outData,
                      bool useNullSelf = false,
                      bool useNullOther = false,
                      bool useNullAlpha = false,
                      bool useNullOut = false)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    bool ok = true;
    if (!useNullSelf && selfSpec != nullptr && selfData != nullptr) {
        ok &= CreateTensor(name + "/self", *selfData, *selfSpec, &self);
    }
    if (!useNullOther && otherSpec != nullptr && otherData != nullptr) {
        ok &= CreateTensor(name + "/other", *otherData, *otherSpec, &other);
    }
    if (!useNullAlpha && alphaValue != nullptr) {
        ok &= CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (!useNullOut && outSpec != nullptr && outData != nullptr) {
        ok &= CreateTensor(name + "/out", *outData, *outSpec, &out);
    }
    if (!ok) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret = aclnnAddGetWorkspaceSize(useNullSelf ? nullptr : self.tensor,
                                              useNullOther ? nullptr : other.tensor,
                                              useNullAlpha ? nullptr : alpha.scalar,
                                              useNullOut ? nullptr : out.tensor,
                                              &workspaceSize,
                                              &executor.executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfT, typename OtherScalarT, typename AlphaT, typename OutT>
bool RunAddsStatusCase(const std::string &name,
                       aclnnStatus expectedStatus,
                       const TensorSpec *selfSpec,
                       const std::vector<SelfT> *selfData,
                       const OtherScalarT *otherValue,
                       aclDataType otherType,
                       const AlphaT *alphaValue,
                       aclDataType alphaType,
                       const TensorSpec *outSpec,
                       const std::vector<OutT> *outData,
                       bool useNullSelf = false,
                       bool useNullOther = false,
                       bool useNullAlpha = false,
                       bool useNullOut = false)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;
    bool ok = true;
    if (!useNullSelf && selfSpec != nullptr && selfData != nullptr) {
        ok &= CreateTensor(name + "/self", *selfData, *selfSpec, &self);
    }
    if (!useNullOther && otherValue != nullptr) {
        ok &= CreateScalar(name + "/other", *otherValue, otherType, &other);
    }
    if (!useNullAlpha && alphaValue != nullptr) {
        ok &= CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (!useNullOut && outSpec != nullptr && outData != nullptr) {
        ok &= CreateTensor(name + "/out", *outData, *outSpec, &out);
    }
    if (!ok) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret = aclnnAddsGetWorkspaceSize(useNullSelf ? nullptr : self.tensor,
                                               useNullOther ? nullptr : other.scalar,
                                               useNullAlpha ? nullptr : alpha.scalar,
                                               useNullOut ? nullptr : out.tensor,
                                               &workspaceSize,
                                               &executor.executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfT, typename OtherT, typename AlphaT>
bool RunInplaceAddStatusCase(const std::string &name,
                             aclnnStatus expectedStatus,
                             const TensorSpec *selfSpec,
                             const std::vector<SelfT> *selfData,
                             const TensorSpec *otherSpec,
                             const std::vector<OtherT> *otherData,
                             const AlphaT *alphaValue,
                             aclDataType alphaType)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    bool ok = true;
    if (selfSpec != nullptr && selfData != nullptr) {
        ok &= CreateTensor(name + "/self", *selfData, *selfSpec, &self);
    }
    if (otherSpec != nullptr && otherData != nullptr) {
        ok &= CreateTensor(name + "/other", *otherData, *otherSpec, &other);
    }
    if (alphaValue != nullptr) {
        ok &= CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (!ok) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret =
        aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor.executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddV3StatusCase(const std::string &name,
                        aclnnStatus expectedStatus,
                        const SelfScalarT *selfValue,
                        aclDataType selfType,
                        const TensorSpec *otherSpec,
                        const std::vector<OtherT> *otherData,
                        const AlphaT *alphaValue,
                        aclDataType alphaType,
                        const TensorSpec *outSpec,
                        const std::vector<OutT> *outData,
                        bool useNullSelf = false,
                        bool useNullOther = false,
                        bool useNullAlpha = false,
                        bool useNullOut = false)
{
    ScalarHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    TensorHolder out;
    bool ok = true;
    if (!useNullSelf && selfValue != nullptr) {
        ok &= CreateScalar(name + "/self", *selfValue, selfType, &self);
    }
    if (!useNullOther && otherSpec != nullptr && otherData != nullptr) {
        ok &= CreateTensor(name + "/other", *otherData, *otherSpec, &other);
    }
    if (!useNullAlpha && alphaValue != nullptr) {
        ok &= CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (!useNullOut && outSpec != nullptr && outData != nullptr) {
        ok &= CreateTensor(name + "/out", *outData, *outSpec, &out);
    }
    if (!ok) {
        return false;
    }

    uint64_t workspaceSize = 0;
    ExecutorHolder executor;
    const auto ret = aclnnAddV3GetWorkspaceSize(useNullSelf ? nullptr : self.scalar,
                                                useNullOther ? nullptr : other.tensor,
                                                useNullAlpha ? nullptr : alpha.scalar,
                                                useNullOut ? nullptr : out.tensor,
                                                &workspaceSize,
                                                &executor.executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

bool ReportCase(const std::string &name, bool ok)
{
    ++g_totalCases;
    if (!ok) {
        ++g_failedCases;
    }
    Log("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    return ok;
}

bool RunAddExecutionCases(RuntimeContext &runtime)
{
    bool ok = true;

    {
        const std::string name = "AddFloatBasicAlphaOne";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        std::vector<float> otherData{2.0f, 3.0f, -4.0f, 0.25f, -2.0f, 8.0f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, float right) { return left + alpha * right; });
        ok &= ReportCase(name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddFloatBroadcastAxpy";
        TensorSpec selfSpec{{2, 1, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{1, 4, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 4, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = -1.5f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 4.0f};
        std::vector<float> otherData{2.0f, -1.0f, 0.5f, 3.0f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, float right) { return left + alpha * right; });
        ok &= ReportCase(name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddFloatNchwAlphaZero";
        TensorSpec selfSpec{{1, 2, 2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_NCHW};
        TensorSpec otherSpec{{1, 2, 2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_NCHW};
        TensorSpec outSpec{{1, 2, 2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_NCHW};
        const float alpha = 0.0f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 5.0f};
        std::vector<float> otherData{0.5f, 2.0f, -1.0f, 0.25f, 8.0f, -2.0f, 1.0f, -1.0f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, float right) { return left + alpha * right; });
        ok &= ReportCase(name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddFloatNonContiguous";
        TensorSpec selfSpec{{2, 3}, {3, 2}, {1, 2}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> otherData{6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, float right) { return left + alpha * right; });
        ok &= ReportCase(name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddFp16AlphaOne";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        const Fp16 alpha(1.0f);
        std::vector<Fp16> selfData{Fp16(1.0f), Fp16(-2.0f), Fp16(0.5f), Fp16(4.0f)};
        std::vector<Fp16> otherData{Fp16(2.0f), Fp16(3.0f), Fp16(-1.0f), Fp16(0.25f)};
        const auto expected = ComputeBroadcastExpected<Fp16>(
            selfSpec, selfData, otherSpec, otherData, [alpha](Fp16 left, Fp16 right) { return Fp16(static_cast<float>(left) + static_cast<float>(alpha) * static_cast<float>(right)); });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT16, outSpec, expected, true));
    }

    {
        const std::string name = "AddBf16AlphaOne";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        const Bf16 alpha(1.0f);
        std::vector<Bf16> selfData{Bf16(1.0f), Bf16(-2.0f), Bf16(0.5f), Bf16(4.0f)};
        std::vector<Bf16> otherData{Bf16(2.0f), Bf16(3.0f), Bf16(-1.0f), Bf16(0.25f)};
        const auto expected = ComputeBroadcastExpected<Bf16>(
            selfSpec, selfData, otherSpec, otherData, [alpha](Bf16 left, Bf16 right) { return Bf16(static_cast<float>(left) + static_cast<float>(alpha) * static_cast<float>(right)); });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_BF16, outSpec, expected, true));
    }

    {
        const std::string name = "AddFp16FloatMix";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<Fp16> selfData{Fp16(1.0f), Fp16(-2.0f), Fp16(0.5f), Fp16(4.0f)};
        std::vector<float> otherData{2.0f, 3.0f, -1.0f, 0.25f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](Fp16 left, float right) { return static_cast<float>(left) + alpha * right; });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddFloatFp16Mix";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{2.0f, -3.0f, 1.5f, 0.25f};
        std::vector<Fp16> otherData{Fp16(1.0f), Fp16(2.0f), Fp16(-2.0f), Fp16(4.0f)};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, Fp16 right) { return left + alpha * static_cast<float>(right); });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddBf16FloatMix";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<Bf16> selfData{Bf16(1.0f), Bf16(-2.0f), Bf16(0.5f), Bf16(4.0f)};
        std::vector<float> otherData{2.0f, 3.0f, -1.0f, 0.25f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](Bf16 left, float right) { return static_cast<float>(left) + alpha * right; });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddFloatBf16Mix";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{2.0f, -3.0f, 1.5f, 0.25f};
        std::vector<Bf16> otherData{Bf16(1.0f), Bf16(2.0f), Bf16(-2.0f), Bf16(4.0f)};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, Bf16 right) { return left + alpha * static_cast<float>(right); });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddInt32AxpyV2";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const int32_t alpha = -3;
        std::vector<int32_t> selfData{100, -200, 3, 17};
        std::vector<int32_t> otherData{2, 3, -4, 5};
        const auto expected = ComputeBroadcastExpected<int32_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](int32_t left, int32_t right) { return left + alpha * right; });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_INT32, outSpec, expected));
    }

    {
        const std::string name = "AddInt64AxpyV2";
        TensorSpec selfSpec{{3}, {}, {}, 0, ACL_INT64, ACL_FORMAT_ND};
        TensorSpec otherSpec{{3}, {}, {}, 0, ACL_INT64, ACL_FORMAT_ND};
        TensorSpec outSpec{{3}, {}, {}, 0, ACL_INT64, ACL_FORMAT_ND};
        const int64_t alpha = 2;
        std::vector<int64_t> selfData{10000000000LL, -2LL, 3LL};
        std::vector<int64_t> otherData{2LL, 3LL, -4LL};
        const auto expected = ComputeBroadcastExpected<int64_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](int64_t left, int64_t right) { return left + alpha * right; });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_INT64, outSpec, expected));
    }

    {
        const std::string name = "AddInt16AiCpuRoute";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_INT16, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_INT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_INT16, ACL_FORMAT_ND};
        const int16_t alpha = 1;
        std::vector<int16_t> selfData{100, -20, 3, 17};
        std::vector<int16_t> otherData{2, 30, -4, 5};
        const auto expected = ComputeBroadcastExpected<int16_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](int16_t left, int16_t right) {
                return static_cast<int16_t>(left + alpha * right);
            });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_INT16, outSpec, expected));
    }

    {
        const std::string name = "AddFloatScaledRoute";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = -1.5f;
        std::vector<float> selfData{1.0f, -2.0f, 3.5f, 0.25f};
        std::vector<float> otherData{4.0f, 0.5f, -1.0f, 8.0f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, float right) { return left + alpha * right; });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddFloatSpecialValues";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            -0.0f};
        std::vector<float> otherData{1.0f, -1.0f, 2.0f, std::numeric_limits<float>::infinity()};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, float right) { return left + alpha * right; });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddInt32CastOutToInt64";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT64, ACL_FORMAT_ND};
        const int32_t alpha = 2;
        std::vector<int32_t> selfData{1, -2, 3, 4};
        std::vector<int32_t> otherData{5, 6, -7, 8};
        const auto expected = ComputeBroadcastExpected<int64_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](int32_t left, int32_t right) {
                return static_cast<int64_t>(left + alpha * right);
            });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_INT32, outSpec, expected));
    }

    {
        const std::string name = "AddInt8Wraparound";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_INT8, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_INT8, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT8, ACL_FORMAT_ND};
        const int8_t alpha = 2;
        std::vector<int8_t> selfData{120, -120, 10, -10};
        std::vector<int8_t> otherData{2, 2, 13, 13};
        const auto expected = ComputeBroadcastExpected<int8_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](int8_t left, int8_t right) {
                return NarrowingAddInt8(left, static_cast<int8_t>(static_cast<int32_t>(alpha) * static_cast<int32_t>(right)));
            });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_INT8, outSpec, expected));
    }

    {
        const std::string name = "AddUInt8Wraparound";
        TensorSpec selfSpec{{3}, {}, {}, 0, ACL_UINT8, ACL_FORMAT_ND};
        TensorSpec otherSpec{{3}, {}, {}, 0, ACL_UINT8, ACL_FORMAT_ND};
        TensorSpec outSpec{{3}, {}, {}, 0, ACL_UINT8, ACL_FORMAT_ND};
        const uint8_t alpha = 2U;
        std::vector<uint8_t> selfData{250U, 20U, 30U};
        std::vector<uint8_t> otherData{2U, 20U, 8U};
        const auto expected = ComputeBroadcastExpected<uint8_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](uint8_t left, uint8_t right) {
                return NarrowingAddUInt8(left, static_cast<uint8_t>(static_cast<uint32_t>(alpha) * static_cast<uint32_t>(right)));
            });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_UINT8, outSpec, expected));
    }

    {
        const std::string name = "AddBool";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        const bool alpha = true;
        std::vector<uint8_t> selfData{1U, 0U, 1U, 1U};
        std::vector<uint8_t> otherData{1U, 1U, 0U, 1U};
        const auto expected = ComputeBroadcastExpected<uint8_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](uint8_t left, uint8_t right) {
                return alpha ? BoolAdd(left, right) : left;
            });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_BOOL, outSpec, expected));
    }

    {
        const std::string name = "AddBoolToFloatOut";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const bool alpha = true;
        std::vector<uint8_t> selfData{1U, 0U, 1U, 0U};
        std::vector<uint8_t> otherData{0U, 1U, 1U, 0U};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](uint8_t left, uint8_t right) {
                const bool result = alpha ? BoolAdd(left, right) : (left != 0U);
                return result ? 1.0f : 0.0f;
            });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_BOOL, outSpec, expected));
    }

    {
        const std::string name = "AddComplex64ScaledFallback";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        const float alpha = -0.5f;
        std::vector<std::complex<float> > selfData{
            std::complex<float>(1.0f, 2.0f), std::complex<float>(-1.0f, 0.5f)};
        std::vector<std::complex<float> > otherData{
            std::complex<float>(3.0f, -1.0f), std::complex<float>(0.0f, 2.0f)};
        const auto expected = ComputeBroadcastExpected<std::complex<float> >(
            selfSpec,
            selfData,
            otherSpec,
            otherData,
            [alpha](const std::complex<float> &left, const std::complex<float> &right) { return left + alpha * right; });
        ok &= ReportCase(
            name, RunAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddEmptyTensor";
        TensorSpec selfSpec{{0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{1, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData;
        std::vector<float> otherData{1.0f, 2.0f, 3.0f};
        std::vector<float> outData;
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alphaHolder;
        bool caseOk = CreateTensor(name + "/self", selfData, selfSpec, &self) &&
                      CreateTensor(name + "/other", otherData, otherSpec, &other) &&
                      CreateScalar(name + "/alpha", alpha, ACL_FLOAT, &alphaHolder) &&
                      CreateTensor(name + "/out", outData, outSpec, &out);
        uint64_t workspaceSize = 1;
        ExecutorHolder executor;
        if (caseOk) {
            const auto ret =
                aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alphaHolder.scalar, out.tensor, &workspaceSize, &executor.executor);
            caseOk = ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS) && CheckWorkspaceZero(name, workspaceSize);
        }
        ok &= ReportCase(name, caseOk);
    }

    return ok;
}

bool RunAddsExecutionCases(RuntimeContext &runtime)
{
    bool ok = true;

    {
        const std::string name = "AddsFloatBasic";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = -0.5f;
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        const auto expected = ComputeScalarExpected<float>(
            selfSpec, selfData, other, [alpha](float left, float scalar) { return left + alpha * scalar; });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_FLOAT, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddsFloatAxpy";
        TensorSpec selfSpec{{1, 2, 2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_NCHW};
        TensorSpec outSpec{{1, 2, 2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_NCHW};
        const float other = 2.5f;
        const float alpha = -0.5f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 5.0f};
        const auto expected = ComputeScalarExpected<float>(
            selfSpec, selfData, other, [alpha](float left, float scalar) { return left + alpha * scalar; });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_FLOAT, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddsFloatNonContiguous";
        TensorSpec selfSpec{{2, 3}, {3, 2}, {1, 2}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const auto expected = ComputeScalarExpected<float>(
            selfSpec, selfData, other, [alpha](float left, float scalar) { return left + alpha * scalar; });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_FLOAT, alpha, ACL_FLOAT, outSpec, expected));
    }

    {
        const std::string name = "AddsFp16";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        const Fp16 other(1.5f);
        const Fp16 alpha(1.0f);
        std::vector<Fp16> selfData{Fp16(1.0f), Fp16(-2.0f), Fp16(0.5f), Fp16(4.0f)};
        const auto expected = ComputeScalarExpected<Fp16>(
            selfSpec, selfData, other, [alpha](Fp16 left, Fp16 scalar) { return Fp16(static_cast<float>(left) + static_cast<float>(alpha) * static_cast<float>(scalar)); });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_FLOAT16, alpha, ACL_FLOAT16, outSpec, expected, true));
    }

    {
        const std::string name = "AddsBf16";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        const Bf16 other(0.75f);
        const Bf16 alpha(1.0f);
        std::vector<Bf16> selfData{Bf16(1.0f), Bf16(-2.0f), Bf16(0.5f), Bf16(4.0f)};
        const auto expected = ComputeScalarExpected<Bf16>(
            selfSpec, selfData, other, [alpha](Bf16 left, Bf16 scalar) { return Bf16(static_cast<float>(left) + static_cast<float>(alpha) * static_cast<float>(scalar)); });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_BF16, alpha, ACL_BF16, outSpec, expected, true));
    }

    {
        const std::string name = "AddsBf16ScaledFallback";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        const Bf16 other(0.75f);
        const Bf16 alpha(0.5f);
        std::vector<Bf16> selfData{Bf16(1.0f), Bf16(-2.0f), Bf16(0.5f), Bf16(4.0f)};
        const auto expected = ComputeScalarExpected<Bf16>(
            selfSpec, selfData, other, [alpha](Bf16 left, Bf16 scalar) {
                return Bf16(static_cast<float>(left) + static_cast<float>(alpha) * static_cast<float>(scalar));
            });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_BF16, alpha, ACL_BF16, outSpec, expected, true));
    }

    {
        const std::string name = "AddsInt32AxpyV2";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const int32_t other = 7;
        const int32_t alpha = -3;
        std::vector<int32_t> selfData{1, -2, 3, 4};
        const auto expected = ComputeScalarExpected<int32_t>(
            selfSpec, selfData, other, [alpha](int32_t left, int32_t scalar) { return left + alpha * scalar; });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_INT32, alpha, ACL_INT32, outSpec, expected));
    }

    {
        const std::string name = "AddsBoolToFloatSpecial";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const bool other = true;
        const bool alpha = true;
        std::vector<uint8_t> selfData{1U, 0U, 1U, 0U};
        const auto expected = ComputeScalarExpected<float>(
            selfSpec, selfData, other, [alpha](uint8_t left, bool scalar) {
                const bool result = (left != 0U) || (alpha && scalar);
                return result ? 1.0f : 0.0f;
            });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_BOOL, alpha, ACL_BOOL, outSpec, expected));
    }

    {
        const std::string name = "AddsFp16Complex64Promote";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        const std::complex<float> other(1.5f, -0.5f);
        const float alpha = 1.0f;
        std::vector<Fp16> selfData{Fp16(2.0f), Fp16(-3.0f)};
        std::vector<std::complex<float> > expected;
        expected.reserve(selfData.size());
        for (Fp16 value : selfData) {
            expected.emplace_back(std::complex<float>(static_cast<float>(value), 0.0f) + alpha * other);
        }
        ok &= ReportCase(
            name,
            RunAddsExecuteCase(
                runtime, name, selfSpec, selfData, other, ACL_COMPLEX64, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddsFloatComplexScalarFallback";
        TensorSpec selfSpec{{3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{3}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        const std::complex<float> other(1.5f, -0.25f);
        const float alpha = -0.5f;
        std::vector<float> selfData{1.0f, -2.0f, 3.5f};
        std::vector<std::complex<float> > expected;
        expected.reserve(selfData.size());
        for (float value : selfData) {
            expected.emplace_back(std::complex<float>(value, 0.0f) + alpha * other);
        }
        ok &= ReportCase(
            name,
            RunAddsExecuteCase(
                runtime, name, selfSpec, selfData, other, ACL_COMPLEX64, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddsComplex64FloatScalarFallback";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        const float other = 2.0f;
        const float alpha = -0.5f;
        std::vector<std::complex<float> > selfData{
            std::complex<float>(1.0f, 2.0f), std::complex<float>(-3.0f, 0.5f)};
        std::vector<std::complex<float> > expected;
        expected.reserve(selfData.size());
        for (const auto &value : selfData) {
            expected.emplace_back(value + std::complex<float>(alpha * other, 0.0f));
        }
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_FLOAT, alpha, ACL_FLOAT, outSpec, expected, true));
    }

    {
        const std::string name = "AddsDoubleComplex128Promote";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_DOUBLE, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_COMPLEX128, ACL_FORMAT_ND};
        const std::complex<double> other(1.5, -0.25);
        const double alpha = -0.5;
        std::vector<double> selfData{2.0, -3.5};
        std::vector<std::complex<double> > expected;
        expected.reserve(selfData.size());
        for (double value : selfData) {
            expected.emplace_back(std::complex<double>(value, 0.0) + alpha * other);
        }
        ok &= ReportCase(
            name,
            RunAddsExecuteCase(
                runtime, name, selfSpec, selfData, other, ACL_COMPLEX128, alpha, ACL_DOUBLE, outSpec, expected, true));
    }

    {
        const std::string name = "AddsDoubleSpecialValues";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_DOUBLE, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_DOUBLE, ACL_FORMAT_ND};
        const double other = std::numeric_limits<double>::infinity();
        const double alpha = -1.0;
        std::vector<double> selfData{
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN(),
            1.0};
        const auto expected = ComputeScalarExpected<double>(
            selfSpec, selfData, other, [alpha](double left, double scalar) { return left + alpha * scalar; });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_DOUBLE, alpha, ACL_DOUBLE, outSpec, expected));
    }

    {
        const std::string name = "AddsInt32CastOutToDouble";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_DOUBLE, ACL_FORMAT_ND};
        const int32_t other = 7;
        const int32_t alpha = -3;
        std::vector<int32_t> selfData{1, -2, 3, 4};
        const auto expected = ComputeScalarExpected<double>(
            selfSpec, selfData, other, [alpha](int32_t left, int32_t scalar) {
                return static_cast<double>(left + alpha * scalar);
            });
        ok &= ReportCase(
            name, RunAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_INT32, alpha, ACL_INT32, outSpec, expected));
    }

    {
        const std::string name = "AddsEmptyTensor";
        TensorSpec selfSpec{{0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> selfData;
        std::vector<float> outData;
        TensorHolder self;
        TensorHolder out;
        ScalarHolder otherHolder;
        ScalarHolder alphaHolder;
        bool caseOk = CreateTensor(name + "/self", selfData, selfSpec, &self) &&
                      CreateScalar(name + "/other", other, ACL_FLOAT, &otherHolder) &&
                      CreateScalar(name + "/alpha", alpha, ACL_FLOAT, &alphaHolder) &&
                      CreateTensor(name + "/out", outData, outSpec, &out);
        uint64_t workspaceSize = 1;
        ExecutorHolder executor;
        if (caseOk) {
            const auto ret = aclnnAddsGetWorkspaceSize(
                self.tensor, otherHolder.scalar, alphaHolder.scalar, out.tensor, &workspaceSize, &executor.executor);
            caseOk = ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS) && CheckWorkspaceZero(name, workspaceSize);
        }
        ok &= ReportCase(name, caseOk);
    }

    return ok;
}

bool RunInplaceExecutionCases(RuntimeContext &runtime)
{
    bool ok = true;

    {
        const std::string name = "InplaceAddFloat";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.5f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        std::vector<float> otherData{2.0f, 3.0f, -4.0f, 0.25f, -2.0f, 8.0f};
        const auto expected = ComputeBroadcastExpected<float>(
            selfSpec, selfData, otherSpec, otherData, [alpha](float left, float right) { return left + alpha * right; });
        ok &= ReportCase(
            name, RunInplaceAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_FLOAT, expected));
    }

    {
        const std::string name = "InplaceAddInt32Broadcast";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec otherSpec{{3}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const int32_t alpha = 2;
        std::vector<int32_t> selfData{1, 2, 3, 4, 5, 6};
        std::vector<int32_t> otherData{10, 20, 30};
        const auto expected = ComputeBroadcastExpected<int32_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](int32_t left, int32_t right) { return left + alpha * right; });
        ok &= ReportCase(
            name, RunInplaceAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_INT32, expected));
    }

    {
        const std::string name = "InplaceAddBool";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        const bool alpha = true;
        std::vector<uint8_t> selfData{1U, 0U, 1U, 0U};
        std::vector<uint8_t> otherData{1U, 1U, 0U, 1U};
        const auto expected = ComputeBroadcastExpected<uint8_t>(
            selfSpec, selfData, otherSpec, otherData, [alpha](uint8_t left, uint8_t right) {
                return alpha ? BoolAdd(left, right) : left;
            });
        ok &= ReportCase(
            name, RunInplaceAddExecuteCase(runtime, name, selfSpec, selfData, otherSpec, otherData, alpha, ACL_BOOL, expected));
    }

    {
        const std::string name = "InplaceAddsFloat";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = 2.5f;
        const float alpha = -0.5f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        const auto expected = ComputeScalarExpected<float>(
            selfSpec, selfData, other, [alpha](float left, float scalar) { return left + alpha * scalar; });
        ok &= ReportCase(
            name, RunInplaceAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_FLOAT, alpha, ACL_FLOAT, expected));
    }

    {
        const std::string name = "InplaceAddsInt64";
        TensorSpec selfSpec{{3}, {}, {}, 0, ACL_INT64, ACL_FORMAT_ND};
        const int64_t other = 5;
        const int64_t alpha = 2;
        std::vector<int64_t> selfData{10LL, -2LL, 3LL};
        const auto expected = ComputeScalarExpected<int64_t>(
            selfSpec, selfData, other, [alpha](int64_t left, int64_t scalar) { return left + alpha * scalar; });
        ok &= ReportCase(
            name, RunInplaceAddsExecuteCase(runtime, name, selfSpec, selfData, other, ACL_INT64, alpha, ACL_INT64, expected));
    }

    return ok;
}

bool RunAddV3ExecutionCases(RuntimeContext &runtime)
{
    bool ok = true;

    {
        const std::string name = "AddV3FloatAlphaOne";
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.25f;
        const float alpha = 1.0f;
        std::vector<float> otherData{2.0f, 3.0f, -4.0f, 0.25f, -2.0f, 8.0f};
        const auto expected = ComputeScalarExpected<float>(
            otherSpec, otherData, otherData.front(), [](float, float) { return 0.0f; });
        std::vector<float> expectedData;
        expectedData.reserve(otherData.size());
        for (float value : otherData) {
            expectedData.push_back(self + alpha * value);
        }
        ok &= ReportCase(
            name, RunAddV3ExecuteCase(runtime, name, self, ACL_FLOAT, otherSpec, otherData, alpha, ACL_FLOAT, outSpec, expectedData));
    }

    {
        const std::string name = "AddV3Fp16Axpy";
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        const Fp16 self(1.0f);
        const Fp16 alpha(0.5f);
        std::vector<Fp16> otherData{Fp16(2.0f), Fp16(3.0f), Fp16(-1.0f), Fp16(0.25f)};
        std::vector<Fp16> expected;
        expected.reserve(otherData.size());
        for (Fp16 value : otherData) {
            expected.emplace_back(static_cast<float>(self) + static_cast<float>(alpha) * static_cast<float>(value));
        }
        ok &= ReportCase(
            name, RunAddV3ExecuteCase(runtime, name, self, ACL_FLOAT16, otherSpec, otherData, alpha, ACL_FLOAT16, outSpec, expected, true));
    }

    {
        const std::string name = "AddV3Int32Axpy";
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const int32_t self = 7;
        const int32_t alpha = -2;
        std::vector<int32_t> otherData{1, 2, -3, 4};
        std::vector<int32_t> expected;
        expected.reserve(otherData.size());
        for (int32_t value : otherData) {
            expected.push_back(self + alpha * value);
        }
        ok &= ReportCase(
            name, RunAddV3ExecuteCase(runtime, name, self, ACL_INT32, otherSpec, otherData, alpha, ACL_INT32, outSpec, expected));
    }

    {
        const std::string name = "AddV3Int8Fallback";
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_INT8, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT8, ACL_FORMAT_ND};
        const int8_t self = 7;
        const int8_t alpha = 2;
        std::vector<int8_t> otherData{1, 2, -3, 4};
        std::vector<int8_t> expected;
        expected.reserve(otherData.size());
        for (int8_t value : otherData) {
            expected.push_back(static_cast<int8_t>(self + alpha * value));
        }
        ok &= ReportCase(
            name, RunAddV3ExecuteCase(runtime, name, self, ACL_INT8, otherSpec, otherData, alpha, ACL_INT8, outSpec, expected));
    }

    {
        const std::string name = "AddV3Int32CastOutToInt64";
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT64, ACL_FORMAT_ND};
        const int32_t self = 7;
        const int32_t alpha = -2;
        std::vector<int32_t> otherData{1, 2, -3, 4};
        std::vector<int64_t> expected;
        expected.reserve(otherData.size());
        for (int32_t value : otherData) {
            expected.push_back(static_cast<int64_t>(self + alpha * value));
        }
        ok &= ReportCase(
            name, RunAddV3ExecuteCase(runtime, name, self, ACL_INT32, otherSpec, otherData, alpha, ACL_INT32, outSpec, expected));
    }

    {
        const std::string name = "InplaceAddV3Float";
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.25f;
        const float alpha = -0.5f;
        std::vector<float> otherData{2.0f, 3.0f, -4.0f, 0.25f, -2.0f, 8.0f};
        std::vector<float> expected;
        expected.reserve(otherData.size());
        for (float value : otherData) {
            expected.push_back(self + alpha * value);
        }
        ok &= ReportCase(
            name, RunInplaceAddV3ExecuteCase(runtime, name, self, ACL_FLOAT, otherSpec, otherData, alpha, ACL_FLOAT, expected));
    }

    {
        const std::string name = "AddV3Bf16Fallback";
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_BF16, ACL_FORMAT_ND};
        const Bf16 self(1.0f);
        const Bf16 alpha(0.5f);
        std::vector<Bf16> otherData{Bf16(2.0f), Bf16(-4.0f), Bf16(0.25f), Bf16(8.0f)};
        std::vector<Bf16> expected;
        expected.reserve(otherData.size());
        for (Bf16 value : otherData) {
            expected.emplace_back(static_cast<float>(self) + static_cast<float>(alpha) * static_cast<float>(value));
        }
        ok &= ReportCase(
            name, RunAddV3ExecuteCase(runtime, name, self, ACL_BF16, otherSpec, otherData, alpha, ACL_BF16, outSpec, expected, true));
    }

    {
        const std::string name = "AddV3EmptyTensor";
        TensorSpec otherSpec{{0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> otherData;
        std::vector<float> outData;
        ScalarHolder selfHolder;
        TensorHolder other;
        ScalarHolder alphaHolder;
        TensorHolder out;
        bool caseOk = CreateScalar(name + "/self", self, ACL_FLOAT, &selfHolder) &&
                      CreateTensor(name + "/other", otherData, otherSpec, &other) &&
                      CreateScalar(name + "/alpha", alpha, ACL_FLOAT, &alphaHolder) &&
                      CreateTensor(name + "/out", outData, outSpec, &out);
        uint64_t workspaceSize = 1;
        ExecutorHolder executor;
        if (caseOk) {
            const auto ret = aclnnAddV3GetWorkspaceSize(
                selfHolder.scalar, other.tensor, alphaHolder.scalar, out.tensor, &workspaceSize, &executor.executor);
            caseOk = ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS) && CheckWorkspaceZero(name, workspaceSize);
        }
        ok &= ReportCase(name, caseOk);
    }

    return ok;
}

bool RunValidationCases()
{
    bool ok = true;

    {
        const std::string name = "AddNullptr";
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> otherData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase<float, float, float, float>(name,
                                                                      ACLNN_ERR_PARAM_NULLPTR,
                                                                      nullptr,
                                                                      nullptr,
                                                                      &otherSpec,
                                                                      &otherData,
                                                                      &alpha,
                                                                      ACL_FLOAT,
                                                                      &outSpec,
                                                                      &outData,
                                                                      true,
                                                                      false,
                                                                      false,
                                                                      false));
    }

    {
        const std::string name = "AddNullOther";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase<float, float, float, float>(name,
                                                                      ACLNN_ERR_PARAM_NULLPTR,
                                                                      &selfSpec,
                                                                      &selfData,
                                                                      nullptr,
                                                                      nullptr,
                                                                      &alpha,
                                                                      ACL_FLOAT,
                                                                      &outSpec,
                                                                      &outData,
                                                                      false,
                                                                      true,
                                                                      false,
                                                                      false));
    }

    {
        const std::string name = "AddNullAlpha";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> otherData{-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase<float, float, float, float>(name,
                                                                      ACLNN_ERR_PARAM_NULLPTR,
                                                                      &selfSpec,
                                                                      &selfData,
                                                                      &otherSpec,
                                                                      &otherData,
                                                                      nullptr,
                                                                      ACL_FLOAT,
                                                                      &outSpec,
                                                                      &outData,
                                                                      false,
                                                                      false,
                                                                      true,
                                                                      false));
    }

    {
        const std::string name = "AddNullOut";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> otherData{-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f};
        ok &= ReportCase(name,
                         RunAddStatusCase<float, float, float, float>(name,
                                                                      ACLNN_ERR_PARAM_NULLPTR,
                                                                      &selfSpec,
                                                                      &selfData,
                                                                      &otherSpec,
                                                                      &otherData,
                                                                      &alpha,
                                                                      ACL_FLOAT,
                                                                      nullptr,
                                                                      nullptr,
                                                                      false,
                                                                      false,
                                                                      false,
                                                                      true));
    }

    {
        const std::string name = "AddOutDtypeInvalid";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<std::complex<float> > selfData{
            std::complex<float>(1.0f, 2.0f), std::complex<float>(-3.0f, 0.5f)};
        std::vector<std::complex<float> > otherData{
            std::complex<float>(0.0f, 1.0f), std::complex<float>(2.0f, -4.0f)};
        std::vector<int32_t> outData = MakeOutputInit<int32_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_FLOAT,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddUnsupportedSelfDtype";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_UINT16, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<uint16_t> selfData{1U, 2U};
        std::vector<float> otherData{3.0f, 4.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_FLOAT,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddUnsupportedOtherDtype";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2}, {}, {}, 0, ACL_UINT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{3.0f, 4.0f};
        std::vector<uint16_t> otherData{1U, 2U};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_FLOAT,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddAlphaCastInvalid";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const std::complex<float> alpha(0.5f, 1.0f);
        std::vector<int32_t> selfData{1, 2};
        std::vector<int32_t> otherData{3, 4};
        std::vector<int32_t> outData = MakeOutputInit<int32_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_COMPLEX64,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddBroadcastInvalid";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> otherData{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_FLOAT,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddOutputShapeInvalid";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{3, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> otherData{10.0f, 20.0f, 30.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_FLOAT,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddBoolFloatAlphaInvalid";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        const float alpha = 0.5f;
        std::vector<uint8_t> selfData{1U, 0U, 1U, 0U};
        std::vector<uint8_t> otherData{1U, 1U, 0U, 1U};
        std::vector<uint8_t> outData = MakeOutputInit<uint8_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_FLOAT,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddRankTooHigh";
        TensorSpec selfSpec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData(1, 1.0f);
        std::vector<float> otherData(1, 2.0f);
        std::vector<float> outData(1, 0.0f);
        ok &= ReportCase(name,
                         RunAddStatusCase(name,
                                          ACLNN_ERR_PARAM_INVALID,
                                          &selfSpec,
                                          &selfData,
                                          &otherSpec,
                                          &otherData,
                                          &alpha,
                                          ACL_FLOAT,
                                          &outSpec,
                                          &outData));
    }

    {
        const std::string name = "AddsNullScalar";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddsStatusCase<float, float, float, float>(name,
                                                                       ACLNN_ERR_PARAM_NULLPTR,
                                                                       &selfSpec,
                                                                       &selfData,
                                                                       nullptr,
                                                                       ACL_FLOAT,
                                                                       &alpha,
                                                                       ACL_FLOAT,
                                                                       &outSpec,
                                                                       &outData,
                                                                       false,
                                                                       true,
                                                                       false,
                                                                       false));
    }

    {
        const std::string name = "AddsNullOut";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = 2.0f;
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        ok &= ReportCase(name,
                         RunAddsStatusCase<float, float, float, float>(name,
                                                                       ACLNN_ERR_PARAM_NULLPTR,
                                                                       &selfSpec,
                                                                       &selfData,
                                                                       &other,
                                                                       ACL_FLOAT,
                                                                       &alpha,
                                                                       ACL_FLOAT,
                                                                       nullptr,
                                                                       nullptr,
                                                                       false,
                                                                       false,
                                                                       false,
                                                                       true));
    }

    {
        const std::string name = "AddsNullAlpha";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = 2.0f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddsStatusCase<float, float, float, float>(name,
                                                                       ACLNN_ERR_PARAM_NULLPTR,
                                                                       &selfSpec,
                                                                       &selfData,
                                                                       &other,
                                                                       ACL_FLOAT,
                                                                       nullptr,
                                                                       ACL_FLOAT,
                                                                       &outSpec,
                                                                       &outData,
                                                                       false,
                                                                       false,
                                                                       true,
                                                                       false));
    }

    {
        const std::string name = "AddsAlphaCastInvalid";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const int32_t other = 7;
        const std::complex<float> alpha(0.5f, 1.0f);
        std::vector<int32_t> selfData{1, 2};
        std::vector<int32_t> outData = MakeOutputInit<int32_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddsStatusCase(name,
                                           ACLNN_ERR_PARAM_INVALID,
                                           &selfSpec,
                                           &selfData,
                                           &other,
                                           ACL_INT32,
                                           &alpha,
                                           ACL_COMPLEX64,
                                           &outSpec,
                                           &outData));
    }

    {
        const std::string name = "AddsOutDtypeInvalid";
        TensorSpec selfSpec{{2}, {}, {}, 0, ACL_COMPLEX64, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const std::complex<float> other(1.5f, -0.25f);
        const float alpha = -0.5f;
        std::vector<std::complex<float> > selfData{
            std::complex<float>(1.0f, 2.0f), std::complex<float>(-3.0f, 0.5f)};
        std::vector<int32_t> outData = MakeOutputInit<int32_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddsStatusCase(name,
                                           ACLNN_ERR_PARAM_INVALID,
                                           &selfSpec,
                                           &selfData,
                                           &other,
                                           ACL_COMPLEX64,
                                           &alpha,
                                           ACL_FLOAT,
                                           &outSpec,
                                           &outData));
    }

    {
        const std::string name = "AddsBoolFloatAlphaInvalid";
        TensorSpec selfSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        const bool other = true;
        const float alpha = 0.5f;
        std::vector<uint8_t> selfData{1U, 0U, 1U, 0U};
        std::vector<uint8_t> outData = MakeOutputInit<uint8_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddsStatusCase(name,
                                           ACLNN_ERR_PARAM_INVALID,
                                           &selfSpec,
                                           &selfData,
                                           &other,
                                           ACL_BOOL,
                                           &alpha,
                                           ACL_FLOAT,
                                           &outSpec,
                                           &outData));
    }

    {
        const std::string name = "AddsRankTooHigh";
        TensorSpec selfSpec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = 2.0f;
        const float alpha = 1.0f;
        std::vector<float> selfData(1, 1.0f);
        std::vector<float> outData(1, 0.0f);
        ok &= ReportCase(name,
                         RunAddsStatusCase(name,
                                           ACLNN_ERR_PARAM_INVALID,
                                           &selfSpec,
                                           &selfData,
                                           &other,
                                           ACL_FLOAT,
                                           &alpha,
                                           ACL_FLOAT,
                                           &outSpec,
                                           &outData));
    }

    {
        const std::string name = "AddsOutputShapeInvalid";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{3, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float other = 2.0f;
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddsStatusCase(name,
                                           ACLNN_ERR_PARAM_INVALID,
                                           &selfSpec,
                                           &selfData,
                                           &other,
                                           ACL_FLOAT,
                                           &alpha,
                                           ACL_FLOAT,
                                           &outSpec,
                                           &outData));
    }

    {
        const std::string name = "InplaceAddBroadcastInvalid";
        TensorSpec selfSpec{{3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float alpha = 1.0f;
        std::vector<float> selfData{1.0f, 2.0f, 3.0f};
        std::vector<float> otherData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        ok &= ReportCase(name,
                         RunInplaceAddStatusCase(name,
                                                 ACLNN_ERR_PARAM_INVALID,
                                                 &selfSpec,
                                                 &selfData,
                                                 &otherSpec,
                                                 &otherData,
                                                 &alpha,
                                                 ACL_FLOAT));
    }

    {
        const std::string name = "AddV3Nullptr";
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddV3StatusCase<float, float, float, float>(name,
                                                                        ACLNN_ERR_PARAM_NULLPTR,
                                                                        &self,
                                                                        ACL_FLOAT,
                                                                        nullptr,
                                                                        nullptr,
                                                                        &alpha,
                                                                        ACL_FLOAT,
                                                                        &outSpec,
                                                                        &outData,
                                                                        true,
                                                                        true,
                                                                        true,
                                                                        false));
    }

    {
        const std::string name = "AddV3NullOther";
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddV3StatusCase<float, float, float, float>(name,
                                                                        ACLNN_ERR_PARAM_NULLPTR,
                                                                        &self,
                                                                        ACL_FLOAT,
                                                                        nullptr,
                                                                        nullptr,
                                                                        &alpha,
                                                                        ACL_FLOAT,
                                                                        &outSpec,
                                                                        &outData,
                                                                        false,
                                                                        true,
                                                                        false,
                                                                        false));
    }

    {
        const std::string name = "AddV3NullAlpha";
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.0f;
        std::vector<float> otherData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddV3StatusCase<float, float, float, float>(name,
                                                                        ACLNN_ERR_PARAM_NULLPTR,
                                                                        &self,
                                                                        ACL_FLOAT,
                                                                        &otherSpec,
                                                                        &otherData,
                                                                        nullptr,
                                                                        ACL_FLOAT,
                                                                        &outSpec,
                                                                        &outData,
                                                                        false,
                                                                        false,
                                                                        true,
                                                                        false));
    }

    {
        const std::string name = "AddV3NullOut";
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> otherData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        ok &= ReportCase(name,
                         RunAddV3StatusCase<float, float, float, float>(name,
                                                                        ACLNN_ERR_PARAM_NULLPTR,
                                                                        &self,
                                                                        ACL_FLOAT,
                                                                        &otherSpec,
                                                                        &otherData,
                                                                        &alpha,
                                                                        ACL_FLOAT,
                                                                        nullptr,
                                                                        nullptr,
                                                                        false,
                                                                        false,
                                                                        false,
                                                                        true));
    }

    {
        const std::string name = "AddV3UnsupportedBoolOther";
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        const bool self = true;
        const bool alpha = true;
        std::vector<uint8_t> otherData{1U, 0U, 1U, 0U};
        std::vector<uint8_t> outData = MakeOutputInit<uint8_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddV3StatusCase(name,
                                            ACLNN_ERR_PARAM_INVALID,
                                            &self,
                                            ACL_BOOL,
                                            &otherSpec,
                                            &otherData,
                                            &alpha,
                                            ACL_BOOL,
                                            &outSpec,
                                            &outData));
    }

    {
        const std::string name = "AddV3AlphaCastInvalid";
        TensorSpec otherSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{4}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const int32_t self = 1;
        const std::complex<float> alpha(0.5f, 1.0f);
        std::vector<int32_t> otherData{1, 2, 3, 4};
        std::vector<int32_t> outData = MakeOutputInit<int32_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddV3StatusCase(name,
                                            ACLNN_ERR_PARAM_INVALID,
                                            &self,
                                            ACL_INT32,
                                            &otherSpec,
                                            &otherData,
                                            &alpha,
                                            ACL_COMPLEX64,
                                            &outSpec,
                                            &outData));
    }

    {
        const std::string name = "AddV3RankTooHigh";
        TensorSpec otherSpec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> otherData(1, 2.0f);
        std::vector<float> outData(1, 0.0f);
        ok &= ReportCase(name,
                         RunAddV3StatusCase(name,
                                            ACLNN_ERR_PARAM_INVALID,
                                            &self,
                                            ACL_FLOAT,
                                            &otherSpec,
                                            &otherData,
                                            &alpha,
                                            ACL_FLOAT,
                                            &outSpec,
                                            &outData));
    }

    {
        const std::string name = "AddV3OutDtypeInvalid";
        TensorSpec otherSpec{{2}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        TensorSpec outSpec{{2}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        const int32_t self = 1;
        const int32_t alpha = 2;
        std::vector<int32_t> otherData{3, 4};
        std::vector<uint8_t> outData = MakeOutputInit<uint8_t>(outSpec);
        ok &= ReportCase(name,
                         RunAddV3StatusCase(name,
                                            ACLNN_ERR_PARAM_INVALID,
                                            &self,
                                            ACL_INT32,
                                            &otherSpec,
                                            &otherData,
                                            &alpha,
                                            ACL_INT32,
                                            &outSpec,
                                            &outData));
    }

    {
        const std::string name = "AddV3OutputShapeInvalid";
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{3, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float self = 1.0f;
        const float alpha = 1.0f;
        std::vector<float> otherData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> outData = MakeOutputInit<float>(outSpec);
        ok &= ReportCase(name,
                         RunAddV3StatusCase(name,
                                            ACLNN_ERR_PARAM_INVALID,
                                            &self,
                                            ACL_FLOAT,
                                            &otherSpec,
                                            &otherData,
                                            &alpha,
                                            ACL_FLOAT,
                                            &outSpec,
                                            &outData));
    }

    return ok;
}

}  // namespace add_example

using namespace add_example;

int main()
{
    RuntimeContext runtime;
    if (!runtime.Init()) {
        return 1;
    }

    g_totalCases = 0;
    g_failedCases = 0;

    bool ok = true;
    ok &= RunAddExecutionCases(runtime);
    ok &= RunAddsExecutionCases(runtime);
    ok &= RunInplaceExecutionCases(runtime);
    ok &= RunAddV3ExecutionCases(runtime);
    ok &= RunValidationCases();

    Log("\n=== Summary: %d failed, %d total ===\n", g_failedCases, g_totalCases);

    if (ok) {
        Log("add example suite passed\n");
        return 0;
    }

    Log("add example suite failed\n");
    return 1;
}
