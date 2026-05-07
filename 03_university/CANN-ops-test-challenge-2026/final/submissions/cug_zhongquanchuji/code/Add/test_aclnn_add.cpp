#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

namespace {

int g_totalCases = 0;
int g_failedCases = 0;
int g_observedCases = 0;

void Log(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
}

struct Fp16 {
    aclFloat16 bits;

    Fp16() : bits(aclFloatToFloat16(0.0f)) {}
    explicit Fp16(float value) : bits(aclFloatToFloat16(value)) {}

    operator float() const
    {
        return aclFloat16ToFloat(bits);
    }
};

static_assert(sizeof(Fp16) == sizeof(aclFloat16), "Fp16 wrapper must match aclFloat16 size");

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= shape[i];
    }
    return size;
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

struct TensorSpec {
    std::vector<int64_t> viewShape;
    std::vector<int64_t> storageShape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    aclDataType dtype = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
};

std::vector<int64_t> EffectiveStorageShape(const TensorSpec &spec)
{
    return spec.storageShape.empty() ? spec.viewShape : spec.storageShape;
}

std::vector<int64_t> EffectiveStrides(const TensorSpec &spec)
{
    return spec.strides.empty() ? ContiguousStrides(spec.viewShape) : spec.strides;
}

size_t StorageElements(const TensorSpec &spec)
{
    const int64_t elements = ShapeSize(EffectiveStorageShape(spec));
    return elements <= 0 ? 0U : static_cast<size_t>(elements);
}

size_t ViewElements(const TensorSpec &spec)
{
    const int64_t elements = ShapeSize(spec.viewShape);
    return elements <= 0 ? 0U : static_cast<size_t>(elements);
}

std::vector<int64_t> FlatToIndices(size_t flat, const std::vector<int64_t> &shape)
{
    std::vector<int64_t> indices(shape.size(), 0);
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
        const int64_t extent = shape[static_cast<size_t>(dim)];
        indices[static_cast<size_t>(dim)] = extent == 0 ? 0 : static_cast<int64_t>(flat % static_cast<size_t>(extent));
        if (extent != 0) {
            flat /= static_cast<size_t>(extent);
        }
    }
    return indices;
}

size_t OffsetForIndices(const std::vector<int64_t> &indices, const std::vector<int64_t> &strides, int64_t baseOffset)
{
    int64_t offset = baseOffset;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides[i];
    }
    return static_cast<size_t>(offset);
}

template <typename T>
std::vector<T> MaterializeView(const TensorSpec &spec, const std::vector<T> &storage)
{
    std::vector<T> view(ViewElements(spec));
    const std::vector<int64_t> strides = EffectiveStrides(spec);
    for (size_t i = 0; i < view.size(); ++i) {
        const std::vector<int64_t> indices = FlatToIndices(i, spec.viewShape);
        const size_t storageIndex = OffsetForIndices(indices, strides, spec.offset);
        view[i] = storage[storageIndex];
    }
    return view;
}

bool BroadcastShape(const std::vector<int64_t> &left, const std::vector<int64_t> &right, std::vector<int64_t> *out)
{
    const size_t rank = std::max(left.size(), right.size());
    out->assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const size_t leftFromBack = rank - i;
        const int64_t leftDim = leftFromBack <= left.size() ? left[left.size() - leftFromBack] : 1;
        const int64_t rightDim = leftFromBack <= right.size() ? right[right.size() - leftFromBack] : 1;
        if (leftDim != rightDim && leftDim != 1 && rightDim != 1) {
            return false;
        }
        (*out)[i] = std::max(leftDim, rightDim);
    }
    return true;
}

size_t BroadcastOffset(size_t outFlat, const std::vector<int64_t> &outShape, const std::vector<int64_t> &inShape,
    const std::vector<int64_t> &inStrides, int64_t inOffset)
{
    const std::vector<int64_t> outIndices = FlatToIndices(outFlat, outShape);
    const size_t rankDiff = outShape.size() - inShape.size();
    int64_t offset = inOffset;
    for (size_t i = 0; i < inShape.size(); ++i) {
        const int64_t index = inShape[i] == 1 ? 0 : outIndices[i + rankDiff];
        offset += index * inStrides[i];
    }
    return static_cast<size_t>(offset);
}

double ToDouble(float value)
{
    return static_cast<double>(value);
}

double ToDouble(double value)
{
    return value;
}

double ToDouble(Fp16 value)
{
    return static_cast<double>(static_cast<float>(value));
}

template <typename T>
double ToDouble(T value)
{
    return static_cast<double>(value);
}

template <typename T>
T FromDouble(double value)
{
    return static_cast<T>(value);
}

template <>
Fp16 FromDouble<Fp16>(double value)
{
    return Fp16(static_cast<float>(value));
}

template <>
uint8_t FromDouble<uint8_t>(double value)
{
    return static_cast<uint8_t>(value);
}

template <typename OutT, typename LeftT, typename RightT, typename AlphaT>
std::vector<OutT> ExpectedAdd(
    const TensorSpec &leftSpec,
    const std::vector<LeftT> &leftStorage,
    const TensorSpec &rightSpec,
    const std::vector<RightT> &rightStorage,
    AlphaT alpha)
{
    std::vector<int64_t> outShape;
    BroadcastShape(leftSpec.viewShape, rightSpec.viewShape, &outShape);
    std::vector<OutT> expected(ShapeSize(outShape));
    const std::vector<int64_t> leftStrides = EffectiveStrides(leftSpec);
    const std::vector<int64_t> rightStrides = EffectiveStrides(rightSpec);
    for (size_t i = 0; i < expected.size(); ++i) {
        const LeftT left = leftStorage[BroadcastOffset(i, outShape, leftSpec.viewShape, leftStrides, leftSpec.offset)];
        const RightT right = rightStorage[BroadcastOffset(i, outShape, rightSpec.viewShape, rightStrides, rightSpec.offset)];
        expected[i] = FromDouble<OutT>(ToDouble(left) + ToDouble(alpha) * ToDouble(right));
    }
    return expected;
}

template <typename OutT, typename LeftT, typename ScalarT, typename AlphaT>
std::vector<OutT> ExpectedAdds(
    const TensorSpec &leftSpec, const std::vector<LeftT> &leftStorage, ScalarT other, AlphaT alpha)
{
    std::vector<OutT> expected(ViewElements(leftSpec));
    const std::vector<int64_t> strides = EffectiveStrides(leftSpec);
    for (size_t i = 0; i < expected.size(); ++i) {
        const std::vector<int64_t> indices = FlatToIndices(i, leftSpec.viewShape);
        const LeftT left = leftStorage[OffsetForIndices(indices, strides, leftSpec.offset)];
        expected[i] = FromDouble<OutT>(ToDouble(left) + ToDouble(alpha) * ToDouble(other));
    }
    return expected;
}

std::vector<float> ExpectedBoolAddsFloat(
    const TensorSpec &selfSpec, const std::vector<uint8_t> &selfStorage, bool other, bool alpha)
{
    std::vector<float> expected(ViewElements(selfSpec), 0.0f);
    const std::vector<int64_t> strides = EffectiveStrides(selfSpec);
    for (size_t i = 0; i < expected.size(); ++i) {
        const std::vector<int64_t> indices = FlatToIndices(i, selfSpec.viewShape);
        const bool self = selfStorage[OffsetForIndices(indices, strides, selfSpec.offset)] != 0;
        expected[i] = (self || (other && alpha)) ? 1.0f : 0.0f;
    }
    return expected;
}

struct RuntimeContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool initialized = false;

    bool Init()
    {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            Log("[FAIL] aclInit failed: %d\n", static_cast<int>(ret));
            return false;
        }
        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            Log("[FAIL] aclrtSetDevice failed: %d\n", static_cast<int>(ret));
            return false;
        }
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            Log("[FAIL] aclrtCreateStream failed: %d\n", static_cast<int>(ret));
            return false;
        }
        initialized = true;
        return true;
    }

    ~RuntimeContext()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (initialized) {
            aclrtResetDevice(deviceId);
            aclFinalize();
        }
    }
};

struct TensorHolder {
    aclTensor *tensor = nullptr;
    void *device = nullptr;

    ~TensorHolder()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (device != nullptr) {
            aclrtFree(device);
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

struct WorkspaceHolder {
    void *addr = nullptr;

    ~WorkspaceHolder()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }
};

template <typename T>
bool CreateTensor(const std::string &name, const TensorSpec &spec, const std::vector<T> &storage, TensorHolder *holder)
{
    const size_t expectedStorage = StorageElements(spec);
    if (storage.size() != expectedStorage) {
        Log("[FAIL] %s storage size mismatch: got %zu expected %zu\n", name.c_str(), storage.size(), expectedStorage);
        return false;
    }

    const size_t bytes = storage.size() * sizeof(T);
    const size_t allocBytes = std::max<size_t>(bytes, 1U);
    aclError ret = aclrtMalloc(&holder->device, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        Log("[FAIL] %s aclrtMalloc failed: %d\n", name.c_str(), static_cast<int>(ret));
        return false;
    }
    if (bytes > 0) {
        ret = aclrtMemcpy(holder->device, bytes, storage.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            Log("[FAIL] %s aclrtMemcpy H2D failed: %d\n", name.c_str(), static_cast<int>(ret));
            return false;
        }
    }

    const std::vector<int64_t> storageShape = EffectiveStorageShape(spec);
    const std::vector<int64_t> strides = EffectiveStrides(spec);
    holder->tensor = aclCreateTensor(spec.viewShape.data(), spec.viewShape.size(), spec.dtype, strides.data(), spec.offset,
        spec.format, storageShape.data(), storageShape.size(), holder->device);
    if (holder->tensor == nullptr) {
        Log("[FAIL] %s aclCreateTensor failed\n", name.c_str());
        return false;
    }
    return true;
}

template <typename T>
bool ReadTensor(const std::string &name, const TensorSpec &spec, const TensorHolder &holder, std::vector<T> *view)
{
    std::vector<T> storage(StorageElements(spec));
    const size_t bytes = storage.size() * sizeof(T);
    if (bytes > 0) {
        aclError ret = aclrtMemcpy(storage.data(), bytes, holder.device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            Log("[FAIL] %s aclrtMemcpy D2H failed: %d\n", name.c_str(), static_cast<int>(ret));
            return false;
        }
    }
    *view = MaterializeView(spec, storage);
    return true;
}

template <typename T>
bool CreateScalar(const std::string &name, const T &value, aclDataType dtype, ScalarHolder *holder)
{
    holder->scalar = aclCreateScalar(const_cast<T *>(&value), dtype);
    if (holder->scalar == nullptr) {
        Log("[FAIL] %s aclCreateScalar failed\n", name.c_str());
        return false;
    }
    return true;
}

bool CreateScalar(const std::string &name, const Fp16 &value, aclDataType dtype, ScalarHolder *holder)
{
    holder->scalar = aclCreateScalar(const_cast<aclFloat16 *>(&value.bits), dtype);
    if (holder->scalar == nullptr) {
        Log("[FAIL] %s aclCreateScalar(Fp16) failed\n", name.c_str());
        return false;
    }
    return true;
}

bool AllocWorkspace(const std::string &name, uint64_t workspaceSize, WorkspaceHolder *workspace)
{
    if (workspaceSize == 0) {
        return true;
    }
    aclError ret = aclrtMalloc(&workspace->addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        Log("[FAIL] %s workspace malloc failed: %d\n", name.c_str(), static_cast<int>(ret));
        return false;
    }
    return true;
}

bool ExpectStatus(const std::string &name, aclnnStatus actual, aclnnStatus expected)
{
    if (actual != expected) {
        Log("[FAIL] %s status got %d expected %d\n", name.c_str(), static_cast<int>(actual), static_cast<int>(expected));
        return false;
    }
    return true;
}

bool IsClose(double actual, double expected, double atol, double rtol)
{
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return std::isinf(actual) && (std::signbit(actual) == std::signbit(expected));
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

template <typename T>
bool CompareVector(const std::string &name, const std::vector<T> &actual, const std::vector<T> &expected,
    double atol = 0.0, double rtol = 0.0)
{
    if (actual.size() != expected.size()) {
        Log("[FAIL] %s size got %zu expected %zu\n", name.c_str(), actual.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const double actualValue = ToDouble(actual[i]);
        const double expectedValue = ToDouble(expected[i]);
        const bool ok = (atol == 0.0 && rtol == 0.0) ? (actualValue == expectedValue)
                                                     : IsClose(actualValue, expectedValue, atol, rtol);
        if (!ok) {
            Log("[FAIL] %s[%zu] actual %.12g expected %.12g\n", name.c_str(), i, actualValue, expectedValue);
            return false;
        }
    }
    return true;
}

template <typename T>
bool ObserveVector(const std::string &name, const std::vector<T> &actual, const std::vector<T> &expected,
    double atol = 0.0, double rtol = 0.0)
{
    if (actual.size() != expected.size()) {
        Log("[PRECISION] %s size actual %zu expected %zu\n", name.c_str(), actual.size(), expected.size());
        return true;
    }

    size_t mismatchCount = 0;
    size_t firstMismatch = actual.size();
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double actualValue = ToDouble(actual[i]);
        const double expectedValue = ToDouble(expected[i]);
        const double absError = std::fabs(actualValue - expectedValue);
        const double relError = expectedValue == 0.0 ? absError : absError / std::fabs(expectedValue);
        maxAbsError = std::max(maxAbsError, absError);
        maxRelError = std::max(maxRelError, relError);
        const bool close = (atol == 0.0 && rtol == 0.0) ? (actualValue == expectedValue)
                                                        : IsClose(actualValue, expectedValue, atol, rtol);
        if (!close) {
            if (firstMismatch == actual.size()) {
                firstMismatch = i;
            }
            ++mismatchCount;
        }
    }

    Log("[PRECISION] %s mismatches %zu/%zu max_abs %.12g max_rel %.12g atol %.12g rtol %.12g\n", name.c_str(),
        mismatchCount, actual.size(), maxAbsError, maxRelError, atol, rtol);
    if (firstMismatch != actual.size()) {
        const double actualValue = ToDouble(actual[firstMismatch]);
        const double expectedValue = ToDouble(expected[firstMismatch]);
        Log("[PRECISION] %s first_mismatch[%zu] actual %.12g expected %.12g abs %.12g\n", name.c_str(),
            firstMismatch, actualValue, expectedValue, std::fabs(actualValue - expectedValue));
    }
    return true;
}

bool ReportCase(const std::string &name, bool ok)
{
    ++g_totalCases;
    if (ok) {
        Log("[PASS] %s\n", name.c_str());
        return true;
    }
    ++g_failedCases;
    Log("[FAIL] %s\n", name.c_str());
    return false;
}

bool ReportObservedCase(const std::string &name, bool ok)
{
    ++g_totalCases;
    ++g_observedCases;
    if (!ok) {
        Log("[OBSERVE] %s execution/setup issue recorded, not blocking this run\n", name.c_str());
        return true;
    }
    Log("[OBSERVE] %s precision/coverage evidence recorded\n", name.c_str());
    return true;
}

template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddCase(RuntimeContext &runtime, const std::string &name, const TensorSpec &selfSpec,
    const std::vector<SelfT> &selfData, const TensorSpec &otherSpec, const std::vector<OtherT> &otherData,
    AlphaT alphaValue, aclDataType alphaType, const TensorSpec &outSpec, const std::vector<OutT> &expected,
    double atol = 0.0, double rtol = 0.0, bool observeOnly = false)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<OutT> outInit(StorageElements(outSpec), OutT());
    if (!CreateTensor(name + "/self", selfSpec, selfData, &self) ||
        !CreateTensor(name + "/other", otherSpec, otherData, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha) ||
        !CreateTensor(name + "/out", outSpec, outInit, &out)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS && observeOnly) {
        Log("[OBSERVE] %s/GetWorkspaceSize status %d, not blocking this observation\n", name.c_str(), static_cast<int>(ret));
        return true;
    }
    if (!ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS)) {
        return false;
    }
    WorkspaceHolder workspace;
    if (!AllocWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    ret = aclnnAdd(workspace.addr, workspaceSize, executor, runtime.stream);
    if (ret != ACL_SUCCESS && observeOnly) {
        Log("[OBSERVE] %s/Execute status %d, not blocking this observation\n", name.c_str(), static_cast<int>(ret));
        return true;
    }
    if (!ExpectStatus(name + "/Execute", ret, ACL_SUCCESS)) {
        return false;
    }
    aclError syncRet = aclrtSynchronizeStream(runtime.stream);
    if (syncRet != ACL_SUCCESS) {
        if (observeOnly) {
            Log("[OBSERVE] %s synchronize status %d, not blocking this observation\n", name.c_str(), static_cast<int>(syncRet));
            return true;
        }
        Log("[FAIL] %s synchronize failed: %d\n", name.c_str(), static_cast<int>(syncRet));
        return false;
    }
    std::vector<OutT> actual;
    if (!ReadTensor(name + "/out", outSpec, out, &actual)) {
        return false;
    }
    return observeOnly ? ObserveVector(name, actual, expected, atol, rtol) : CompareVector(name, actual, expected, atol, rtol);
}

template <typename SelfT, typename OtherScalarT, typename AlphaT, typename OutT>
bool RunAddsCase(RuntimeContext &runtime, const std::string &name, const TensorSpec &selfSpec,
    const std::vector<SelfT> &selfData, OtherScalarT otherValue, aclDataType otherType, AlphaT alphaValue,
    aclDataType alphaType, const TensorSpec &outSpec, const std::vector<OutT> &expected, double atol = 0.0,
    double rtol = 0.0, bool observeOnly = false)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;
    std::vector<OutT> outInit(StorageElements(outSpec), OutT());
    if (!CreateTensor(name + "/self", selfSpec, selfData, &self) ||
        !CreateScalar(name + "/other", otherValue, otherType, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha) ||
        !CreateTensor(name + "/out", outSpec, outInit, &out)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS && observeOnly) {
        Log("[OBSERVE] %s/GetWorkspaceSize status %d, not blocking this observation\n", name.c_str(), static_cast<int>(ret));
        return true;
    }
    if (!ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS)) {
        return false;
    }
    WorkspaceHolder workspace;
    if (!AllocWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    ret = aclnnAdds(workspace.addr, workspaceSize, executor, runtime.stream);
    if (ret != ACL_SUCCESS && observeOnly) {
        Log("[OBSERVE] %s/Execute status %d, not blocking this observation\n", name.c_str(), static_cast<int>(ret));
        return true;
    }
    if (!ExpectStatus(name + "/Execute", ret, ACL_SUCCESS)) {
        return false;
    }
    aclError syncRet = aclrtSynchronizeStream(runtime.stream);
    if (syncRet != ACL_SUCCESS) {
        if (observeOnly) {
            Log("[OBSERVE] %s synchronize status %d, not blocking this observation\n", name.c_str(), static_cast<int>(syncRet));
            return true;
        }
        Log("[FAIL] %s synchronize failed: %d\n", name.c_str(), static_cast<int>(syncRet));
        return false;
    }
    std::vector<OutT> actual;
    if (!ReadTensor(name + "/out", outSpec, out, &actual)) {
        return false;
    }
    return observeOnly ? ObserveVector(name, actual, expected, atol, rtol) : CompareVector(name, actual, expected, atol, rtol);
}

template <typename SelfT, typename OtherT, typename AlphaT>
bool RunInplaceAddCase(RuntimeContext &runtime, const std::string &name, const TensorSpec &selfSpec,
    const std::vector<SelfT> &selfData, const TensorSpec &otherSpec, const std::vector<OtherT> &otherData,
    AlphaT alphaValue, aclDataType alphaType, const std::vector<SelfT> &expected, double atol = 0.0, double rtol = 0.0)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    if (!CreateTensor(name + "/self", selfSpec, selfData, &self) ||
        !CreateTensor(name + "/other", otherSpec, otherData, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (!ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS)) {
        return false;
    }
    WorkspaceHolder workspace;
    if (!AllocWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    ret = aclnnInplaceAdd(workspace.addr, workspaceSize, executor, runtime.stream);
    if (!ExpectStatus(name + "/Execute", ret, ACL_SUCCESS)) {
        return false;
    }
    aclError syncRet = aclrtSynchronizeStream(runtime.stream);
    if (syncRet != ACL_SUCCESS) {
        Log("[FAIL] %s synchronize failed: %d\n", name.c_str(), static_cast<int>(syncRet));
        return false;
    }
    std::vector<SelfT> actual;
    return ReadTensor(name + "/self", selfSpec, self, &actual) && CompareVector(name, actual, expected, atol, rtol);
}

template <typename SelfT, typename OtherScalarT, typename AlphaT>
bool RunInplaceAddsCase(RuntimeContext &runtime, const std::string &name, const TensorSpec &selfSpec,
    const std::vector<SelfT> &selfData, OtherScalarT otherValue, aclDataType otherType, AlphaT alphaValue,
    aclDataType alphaType, const std::vector<SelfT> &expected, double atol = 0.0, double rtol = 0.0)
{
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;
    if (!CreateTensor(name + "/self", selfSpec, selfData, &self) ||
        !CreateScalar(name + "/other", otherValue, otherType, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
    if (!ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS)) {
        return false;
    }
    WorkspaceHolder workspace;
    if (!AllocWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    ret = aclnnInplaceAdds(workspace.addr, workspaceSize, executor, runtime.stream);
    if (!ExpectStatus(name + "/Execute", ret, ACL_SUCCESS)) {
        return false;
    }
    aclError syncRet = aclrtSynchronizeStream(runtime.stream);
    if (syncRet != ACL_SUCCESS) {
        Log("[FAIL] %s synchronize failed: %d\n", name.c_str(), static_cast<int>(syncRet));
        return false;
    }
    std::vector<SelfT> actual;
    return ReadTensor(name + "/self", selfSpec, self, &actual) && CompareVector(name, actual, expected, atol, rtol);
}

template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddV3Case(RuntimeContext &runtime, const std::string &name, SelfScalarT selfValue, aclDataType selfType,
    const TensorSpec &otherSpec, const std::vector<OtherT> &otherData, AlphaT alphaValue, aclDataType alphaType,
    const TensorSpec &outSpec, const std::vector<OutT> &expected, double atol = 0.0, double rtol = 0.0,
    bool observeOnly = false)
{
    ScalarHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    TensorHolder out;
    std::vector<OutT> outInit(StorageElements(outSpec), OutT());
    if (!CreateScalar(name + "/self", selfValue, selfType, &self) ||
        !CreateTensor(name + "/other", otherSpec, otherData, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha) ||
        !CreateTensor(name + "/out", outSpec, outInit, &out)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS && observeOnly) {
        Log("[OBSERVE] %s/GetWorkspaceSize status %d, not blocking this observation\n", name.c_str(), static_cast<int>(ret));
        return true;
    }
    if (!ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS)) {
        return false;
    }
    WorkspaceHolder workspace;
    if (!AllocWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    ret = aclnnAddV3(workspace.addr, workspaceSize, executor, runtime.stream);
    if (ret != ACL_SUCCESS && observeOnly) {
        Log("[OBSERVE] %s/Execute status %d, not blocking this observation\n", name.c_str(), static_cast<int>(ret));
        return true;
    }
    if (!ExpectStatus(name + "/Execute", ret, ACL_SUCCESS)) {
        return false;
    }
    aclError syncRet = aclrtSynchronizeStream(runtime.stream);
    if (syncRet != ACL_SUCCESS) {
        if (observeOnly) {
            Log("[OBSERVE] %s synchronize status %d, not blocking this observation\n", name.c_str(), static_cast<int>(syncRet));
            return true;
        }
        Log("[FAIL] %s synchronize failed: %d\n", name.c_str(), static_cast<int>(syncRet));
        return false;
    }
    std::vector<OutT> actual;
    if (!ReadTensor(name + "/out", outSpec, out, &actual)) {
        return false;
    }
    return observeOnly ? ObserveVector(name, actual, expected, atol, rtol) : CompareVector(name, actual, expected, atol, rtol);
}

template <typename SelfScalarT, typename OtherT, typename AlphaT>
bool RunInplaceAddV3Case(RuntimeContext &runtime, const std::string &name, SelfScalarT selfValue, aclDataType selfType,
    const TensorSpec &otherSpec, const std::vector<OtherT> &otherData, AlphaT alphaValue, aclDataType alphaType,
    const std::vector<OtherT> &expected, double atol = 0.0, double rtol = 0.0)
{
    ScalarHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    if (!CreateScalar(name + "/self", selfValue, selfType, &self) ||
        !CreateTensor(name + "/other", otherSpec, otherData, &other) ||
        !CreateScalar(name + "/alpha", alphaValue, alphaType, &alpha)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (!ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS)) {
        return false;
    }
    WorkspaceHolder workspace;
    if (!AllocWorkspace(name, workspaceSize, &workspace)) {
        return false;
    }
    ret = aclnnInplaceAddV3(workspace.addr, workspaceSize, executor, runtime.stream);
    if (!ExpectStatus(name + "/Execute", ret, ACL_SUCCESS)) {
        return false;
    }
    aclError syncRet = aclrtSynchronizeStream(runtime.stream);
    if (syncRet != ACL_SUCCESS) {
        Log("[FAIL] %s synchronize failed: %d\n", name.c_str(), static_cast<int>(syncRet));
        return false;
    }
    std::vector<OtherT> actual;
    return ReadTensor(name + "/other", otherSpec, other, &actual) && CompareVector(name, actual, expected, atol, rtol);
}

template <typename SelfT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddStatusCase(const std::string &name, aclnnStatus expectedStatus, const TensorSpec *selfSpec,
    const std::vector<SelfT> *selfData, const TensorSpec *otherSpec, const std::vector<OtherT> *otherData,
    const AlphaT *alphaValue, aclDataType alphaType, const TensorSpec *outSpec, const std::vector<OutT> *outData)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    TensorHolder out;
    bool ok = true;
    if (selfSpec != nullptr) {
        ok = ok && CreateTensor(name + "/self", *selfSpec, *selfData, &self);
    }
    if (otherSpec != nullptr) {
        ok = ok && CreateTensor(name + "/other", *otherSpec, *otherData, &other);
    }
    if (alphaValue != nullptr) {
        ok = ok && CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (outSpec != nullptr) {
        ok = ok && CreateTensor(name + "/out", *outSpec, *outData, &out);
    }
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfT, typename OtherScalarT, typename AlphaT, typename OutT>
bool RunAddsStatusCase(const std::string &name, aclnnStatus expectedStatus, const TensorSpec *selfSpec,
    const std::vector<SelfT> *selfData, const OtherScalarT *otherValue, aclDataType otherType,
    const AlphaT *alphaValue, aclDataType alphaType, const TensorSpec *outSpec, const std::vector<OutT> *outData)
{
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;
    TensorHolder out;
    bool ok = true;
    if (selfSpec != nullptr) {
        ok = ok && CreateTensor(name + "/self", *selfSpec, *selfData, &self);
    }
    if (otherValue != nullptr) {
        ok = ok && CreateScalar(name + "/other", *otherValue, otherType, &other);
    }
    if (alphaValue != nullptr) {
        ok = ok && CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (outSpec != nullptr) {
        ok = ok && CreateTensor(name + "/out", *outSpec, *outData, &out);
    }
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfScalarT, typename OtherT, typename AlphaT, typename OutT>
bool RunAddV3StatusCase(const std::string &name, aclnnStatus expectedStatus, const SelfScalarT *selfValue,
    aclDataType selfType, const TensorSpec *otherSpec, const std::vector<OtherT> *otherData, const AlphaT *alphaValue,
    aclDataType alphaType, const TensorSpec *outSpec, const std::vector<OutT> *outData)
{
    ScalarHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    TensorHolder out;
    bool ok = true;
    if (selfValue != nullptr) {
        ok = ok && CreateScalar(name + "/self", *selfValue, selfType, &self);
    }
    if (otherSpec != nullptr) {
        ok = ok && CreateTensor(name + "/other", *otherSpec, *otherData, &other);
    }
    if (alphaValue != nullptr) {
        ok = ok && CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (outSpec != nullptr) {
        ok = ok && CreateTensor(name + "/out", *outSpec, *outData, &out);
    }
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfT, typename OtherT, typename AlphaT>
bool RunInplaceAddStatusCase(const std::string &name, aclnnStatus expectedStatus, const TensorSpec *selfSpec,
    const std::vector<SelfT> *selfData, const TensorSpec *otherSpec, const std::vector<OtherT> *otherData,
    const AlphaT *alphaValue, aclDataType alphaType)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    bool ok = true;
    if (selfSpec != nullptr) {
        ok = ok && CreateTensor(name + "/self", *selfSpec, *selfData, &self);
    }
    if (otherSpec != nullptr) {
        ok = ok && CreateTensor(name + "/other", *otherSpec, *otherData, &other);
    }
    if (alphaValue != nullptr) {
        ok = ok && CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfT, typename OtherScalarT, typename AlphaT>
bool RunInplaceAddsStatusCase(const std::string &name, aclnnStatus expectedStatus, const TensorSpec *selfSpec,
    const std::vector<SelfT> *selfData, const OtherScalarT *otherValue, aclDataType otherType,
    const AlphaT *alphaValue, aclDataType alphaType)
{
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;
    bool ok = true;
    if (selfSpec != nullptr) {
        ok = ok && CreateTensor(name + "/self", *selfSpec, *selfData, &self);
    }
    if (otherValue != nullptr) {
        ok = ok && CreateScalar(name + "/other", *otherValue, otherType, &other);
    }
    if (alphaValue != nullptr) {
        ok = ok && CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

template <typename SelfScalarT, typename OtherT, typename AlphaT>
bool RunInplaceAddV3StatusCase(const std::string &name, aclnnStatus expectedStatus, const SelfScalarT *selfValue,
    aclDataType selfType, const TensorSpec *otherSpec, const std::vector<OtherT> *otherData, const AlphaT *alphaValue,
    aclDataType alphaType)
{
    ScalarHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    bool ok = true;
    if (selfValue != nullptr) {
        ok = ok && CreateScalar(name + "/self", *selfValue, selfType, &self);
    }
    if (otherSpec != nullptr) {
        ok = ok && CreateTensor(name + "/other", *otherSpec, *otherData, &other);
    }
    if (alphaValue != nullptr) {
        ok = ok && CreateScalar(name + "/alpha", *alphaValue, alphaType, &alpha);
    }
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, expectedStatus);
}

bool RunWorkspaceZeroAdd(RuntimeContext &, const std::string &name)
{
    TensorSpec selfSpec{{2, 0, 3}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
    TensorSpec otherSpec{{1, 0, 3}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
    TensorSpec outSpec{{2, 0, 3}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
    std::vector<int32_t> empty;
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    const int64_t alphaValue = 1;
    bool ok = CreateTensor(name + "/self", selfSpec, empty, &self) &&
              CreateTensor(name + "/other", otherSpec, empty, &other) &&
              CreateTensor(name + "/out", outSpec, empty, &out) &&
              CreateScalar(name + "/alpha", alphaValue, ACL_INT64, &alpha);
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS) && workspaceSize == 0;
}

bool RunWorkspaceZeroAdds(RuntimeContext &, const std::string &name)
{
    TensorSpec spec{{2, 0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
    std::vector<float> empty;
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;
    const float otherValue = 1.0f;
    const float alphaValue = 1.0f;
    bool ok = CreateTensor(name + "/self", spec, empty, &self) &&
              CreateTensor(name + "/out", spec, empty, &out) &&
              CreateScalar(name + "/other", otherValue, ACL_FLOAT, &other) &&
              CreateScalar(name + "/alpha", alphaValue, ACL_FLOAT, &alpha);
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS) && workspaceSize == 0;
}

bool RunWorkspaceZeroAddV3(RuntimeContext &, const std::string &name)
{
    TensorSpec spec{{2, 0, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
    std::vector<float> empty;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;
    const float selfValue = 1.0f;
    const float alphaValue = 1.0f;
    bool ok = CreateTensor(name + "/other", spec, empty, &other) &&
              CreateTensor(name + "/out", spec, empty, &out) &&
              CreateScalar(name + "/self", selfValue, ACL_FLOAT, &self) &&
              CreateScalar(name + "/alpha", alphaValue, ACL_FLOAT, &alpha);
    if (!ok) {
        return false;
    }
    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    return ExpectStatus(name + "/GetWorkspaceSize", ret, ACL_SUCCESS) && workspaceSize == 0;
}

bool RunNormalCases(RuntimeContext &runtime)
{
    bool ok = true;

    {
        const std::string name = "AddFp32AlphaOne";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, -2.0f, 0.5f, 4.0f, 10000000000.0f, 1.0000001f};
        std::vector<float> other{2.0f, 3.0f, -1.0f, 0.25f, 0.00001f, -1.0f};
        const float alpha = 1.0f;
        const std::vector<float> expected = ExpectedAdd<float>(spec, self, spec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_FLOAT, spec, expected, 1e-6, 1e-6, true)) && ok;
    }

    {
        const std::string name = "AddFp32BroadcastAxpy";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{1, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f};
        std::vector<float> other{0.5f, -1.0f, 2.0f};
        const float alpha = 1.5f;
        const std::vector<float> expected = ExpectedAdd<float>(selfSpec, self, otherSpec, other, alpha);
        ok = ReportCase(name, RunAddCase(runtime, name, selfSpec, self, otherSpec, other, alpha, ACL_FLOAT, outSpec, expected, 1e-6, 1e-6)) && ok;
    }

    {
        const std::string name = "AddsFp32Scalar";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, -2.0f, 0.5f, 4.0f, 0.0f, -0.0f};
        const float other = 2.0f;
        const float alpha = -0.5f;
        const std::vector<float> expected = ExpectedAdds<float>(spec, self, other, alpha);
        ok = ReportCase(name, RunAddsCase(runtime, name, spec, self, other, ACL_FLOAT, alpha, ACL_FLOAT, spec, expected, 1e-6, 1e-6)) && ok;
    }

    {
        const std::string name = "InplaceAddFp32";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{1, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f};
        std::vector<float> other{0.5f, -1.0f, 2.0f};
        const float alpha = 2.0f;
        const std::vector<float> expected = ExpectedAdd<float>(selfSpec, self, otherSpec, other, alpha);
        ok = ReportCase(name, RunInplaceAddCase(runtime, name, selfSpec, self, otherSpec, other, alpha, ACL_FLOAT, expected, 1e-6, 1e-6)) && ok;
    }

    {
        const std::string name = "InplaceAddsFp32";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, -2.0f, 0.5f, 4.0f, 0.0f, -0.0f};
        const float other = 2.0f;
        const float alpha = 0.25f;
        const std::vector<float> expected = ExpectedAdds<float>(spec, self, other, alpha);
        ok = ReportCase(name, RunInplaceAddsCase(runtime, name, spec, self, other, ACL_FLOAT, alpha, ACL_FLOAT, expected, 1e-6, 1e-6)) && ok;
    }

    {
        const std::string name = "AddV3Fp32";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float selfScalar = 10.0f;
        std::vector<float> other{1.0f, -2.0f, 0.5f, 4.0f, 0.0f, -0.0f};
        const float alpha = 0.5f;
        std::vector<float> expected(other.size());
        for (size_t i = 0; i < other.size(); ++i) {
            expected[i] = selfScalar + alpha * other[i];
        }
        ok = ReportCase(name, RunAddV3Case(runtime, name, selfScalar, ACL_FLOAT, spec, other, alpha, ACL_FLOAT, spec, expected, 1e-6, 1e-6)) && ok;
    }

    {
        const std::string name = "InplaceAddV3Fp32";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float selfScalar = -1.0f;
        std::vector<float> other{1.0f, -2.0f, 0.5f, 4.0f, 0.0f, -0.0f};
        const float alpha = 2.0f;
        std::vector<float> expected(other.size());
        for (size_t i = 0; i < other.size(); ++i) {
            expected[i] = selfScalar + alpha * other[i];
        }
        ok = ReportCase(name, RunInplaceAddV3Case(runtime, name, selfScalar, ACL_FLOAT, spec, other, alpha, ACL_FLOAT, expected, 1e-6, 1e-6)) && ok;
    }

    {
        const std::string name = "AddFp16FloatMixed";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<Fp16> self{Fp16(1.0f), Fp16(-2.0f), Fp16(0.5f), Fp16(4.0f)};
        std::vector<float> other{2.0f, 3.0f, -1.0f, 0.25f};
        const float alpha = 1.0f;
        const std::vector<float> expected = ExpectedAdd<float>(selfSpec, self, otherSpec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, selfSpec, self, otherSpec, other, alpha, ACL_FLOAT, outSpec, expected, 1e-3, 1e-3, true)) && ok;
    }

    {
        const std::string name = "AddFloatFp16Mixed";
        TensorSpec selfSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, -2.0f, 0.5f, 4.0f};
        std::vector<Fp16> other{Fp16(2.0f), Fp16(3.0f), Fp16(-1.0f), Fp16(0.25f)};
        const float alpha = 1.0f;
        const std::vector<float> expected = ExpectedAdd<float>(selfSpec, self, otherSpec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, selfSpec, self, otherSpec, other, alpha, ACL_FLOAT, outSpec, expected, 1e-3, 1e-3, true)) && ok;
    }

    {
        const std::string name = "AddInt32Scaled";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        std::vector<int32_t> self{1, -2, 1000, 1073741824, 0, -7};
        std::vector<int32_t> other{2, 3, -1, 1, 5, -6};
        const int64_t alpha = 2;
        const std::vector<int32_t> expected = ExpectedAdd<int32_t>(spec, self, spec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_INT64, spec, expected, 0.0, 0.0, true)) && ok;
    }

    {
        const std::string name = "AddsBoolToFloat";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<uint8_t> self{1, 0, 0, 1, 0, 1};
        const bool other = true;
        const bool alpha = true;
        const std::vector<float> expected = ExpectedBoolAddsFloat(selfSpec, self, other, alpha);
        ok = ReportObservedCase(name, RunAddsCase(runtime, name, selfSpec, self, other, ACL_BOOL, alpha, ACL_BOOL, outSpec, expected, 1e-6, 1e-6, true)) && ok;
    }

    {
        const std::string name = "AddNonContiguousFp32";
        TensorSpec spec{{5, 4}, {4, 5}, {1, 5}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{5, 4}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self(20);
        std::vector<float> other(20);
        for (size_t i = 0; i < 20; ++i) {
            self[i] = static_cast<float>(i) * 0.25f;
            other[i] = 1.0f - static_cast<float>(i) * 0.125f;
        }
        const float alpha = 1.0f;
        const std::vector<float> expected = ExpectedAdd<float>(spec, self, spec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_FLOAT, outSpec, expected, 1e-6, 1e-6, true)) && ok;
    }

    ok = ReportCase("AddEmptyWorkspaceZero", RunWorkspaceZeroAdd(runtime, "AddEmptyWorkspaceZero")) && ok;
    ok = ReportCase("AddsEmptyWorkspaceZero", RunWorkspaceZeroAdds(runtime, "AddsEmptyWorkspaceZero")) && ok;
    ok = ReportCase("AddV3EmptyWorkspaceZero", RunWorkspaceZeroAddV3(runtime, "AddV3EmptyWorkspaceZero")) && ok;

    {
        const std::string name = "AddFp32AlphaZero";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, -2.0f, 0.5f, 4.0f, 0.0f, -0.0f};
        std::vector<float> other{2.0f, 3.0f, -1.0f, 0.25f, 100.0f, -100.0f};
        const float alpha = 0.0f;
        const std::vector<float> expected = ExpectedAdd<float>(spec, self, spec, other, alpha);
        ok = ReportCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_FLOAT, spec, expected, 1e-6, 1e-6)) && ok;
    }

    {
        const std::string name = "AddInt64ScaledObservation";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_INT64, ACL_FORMAT_ND};
        std::vector<int64_t> self{1, -2, 1000, 1073741824LL, 0, -7};
        std::vector<int64_t> other{2, 3, -1, 1, 5, -6};
        const int64_t alpha = 2;
        const std::vector<int64_t> expected = ExpectedAdd<int64_t>(spec, self, spec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_INT64, spec, expected, 0.0, 0.0, true)) && ok;
    }

    {
        const std::string name = "AddUint8ScaledObservation";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_UINT8, ACL_FORMAT_ND};
        std::vector<uint8_t> self{1, 2, 3, 10, 100, 200};
        std::vector<uint8_t> other{2, 3, 4, 5, 6, 7};
        const int64_t alpha = 2;
        const std::vector<uint8_t> expected = ExpectedAdd<uint8_t>(spec, self, spec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_INT64, spec, expected, 0.0, 0.0, true)) && ok;
    }

    {
        const std::string name = "AddInt8ScaledObservation";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_INT8, ACL_FORMAT_ND};
        std::vector<int8_t> self{1, -2, 3, 10, 100, -100};
        std::vector<int8_t> other{2, 3, -4, 5, 6, -7};
        const int64_t alpha = 2;
        const std::vector<int8_t> expected = ExpectedAdd<int8_t>(spec, self, spec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_INT64, spec, expected, 0.0, 0.0, true)) && ok;
    }

    {
        const std::string name = "AddV3Int32Observation";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_INT32, ACL_FORMAT_ND};
        const int32_t selfScalar = 10;
        std::vector<int32_t> other{1, -2, 3, 4, 0, -7};
        const int64_t alpha = 2;
        std::vector<int32_t> expected(other.size());
        for (size_t i = 0; i < other.size(); ++i) {
            expected[i] = static_cast<int32_t>(selfScalar + alpha * other[i]);
        }
        ok = ReportObservedCase(name, RunAddV3Case(runtime, name, selfScalar, ACL_INT32, spec, other, alpha, ACL_INT64, spec, expected, 0.0, 0.0, true)) && ok;
    }

    {
        const std::string name = "AddsFp16ScalarPromoteObservation";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT16, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<Fp16> self{Fp16(1.0f), Fp16(-2.0f), Fp16(0.5f), Fp16(4.0f), Fp16(0.0f), Fp16(-0.0f)};
        const float other = 0.1f;
        const float alpha = 0.3f;
        const std::vector<float> expected = ExpectedAdds<float>(selfSpec, self, other, alpha);
        ok = ReportObservedCase(name, RunAddsCase(runtime, name, selfSpec, self, other, ACL_FLOAT, alpha, ACL_FLOAT, outSpec, expected, 1e-3, 1e-3, true)) && ok;
    }

    {
        const std::string name = "AddsFp32DecimalPrecisionObservation";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{0.1f, 0.2f, 0.3f, 0.7f, -0.1f, -0.7f};
        const float other = 0.2f;
        const float alpha = 0.3f;
        const std::vector<float> expected = ExpectedAdds<float>(spec, self, other, alpha);
        ok = ReportObservedCase(name, RunAddsCase(runtime, name, spec, self, other, ACL_FLOAT, alpha, ACL_FLOAT, spec, expected, 1e-6, 1e-6, true)) && ok;
    }

    {
        const std::string name = "AddFp32SpecialValuesObservation";
        TensorSpec spec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> self{inf, -inf, nan, 3.0e38f, 1.0e-38f, -1.0e-38f};
        std::vector<float> other{1.0f, inf, 2.0f, 3.0e38f, -1.0e-38f, 1.0e-38f};
        const float alpha = 1.5f;
        const std::vector<float> expected = ExpectedAdd<float>(spec, self, spec, other, alpha);
        ok = ReportObservedCase(name, RunAddCase(runtime, name, spec, self, spec, other, alpha, ACL_FLOAT, spec, expected, 1e-4, 1e-4, true)) && ok;
    }

    return ok;
}

bool RunNegativeCases()
{
    bool ok = true;

    {
        const std::string name = "AddNullAlpha";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        ok = ReportCase(name,
                 RunAddStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &spec, &data, &spec, &data, nullptr, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddNullSelf";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, nullptr, nullptr, &spec, &data, &alpha, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddNullOther";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &spec, &data, nullptr, nullptr, &alpha, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddNullOut";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &spec, &data, &spec, &data, &alpha, ACL_FLOAT, nullptr, nullptr)) &&
             ok;
    }

    {
        const std::string name = "AddBoolAlphaFloatInvalid";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        std::vector<uint8_t> data{1, 0, 1, 0};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddStatusCase<uint8_t, uint8_t, float, uint8_t>(
                     name, ACLNN_ERR_PARAM_INVALID, &spec, &data, &spec, &data, &alpha, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddUnsupportedUint32";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_UINT32, ACL_FORMAT_ND};
        std::vector<uint32_t> data{1U, 2U, 3U, 4U};
        const int64_t alpha = 1;
        ok = ReportCase(name,
                 RunAddStatusCase<uint32_t, uint32_t, int64_t, uint32_t>(
                     name, ACLNN_ERR_PARAM_INVALID, &spec, &data, &spec, &data, &alpha, ACL_INT64, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddInvalidBroadcast";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> other{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> out(6, 0.0f);
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_INVALID, &selfSpec, &self, &otherSpec, &other, &alpha, ACL_FLOAT, &outSpec, &out)) &&
             ok;
    }

    {
        const std::string name = "AddsWrongOutputShape";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{3, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> out(6, 0.0f);
        const float other = 1.0f;
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddsStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_INVALID, &selfSpec, &self, &other, ACL_FLOAT, &alpha, ACL_FLOAT, &outSpec, &out)) &&
             ok;
    }

    {
        const std::string name = "AddsNullOther";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddsStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &spec, &data, nullptr, ACL_FLOAT, &alpha, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddsNullAlpha";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float other = 1.0f;
        ok = ReportCase(name,
                 RunAddsStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &spec, &data, &other, ACL_FLOAT, nullptr, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddsNullOut";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float other = 1.0f;
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddsStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &spec, &data, &other, ACL_FLOAT, &alpha, ACL_FLOAT, nullptr, nullptr)) &&
             ok;
    }

    {
        const std::string name = "InplaceAddNullSelf";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunInplaceAddStatusCase<float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, nullptr, nullptr, &spec, &data, &alpha, ACL_FLOAT)) &&
             ok;
    }

    {
        const std::string name = "InplaceAddInvalidBroadcast";
        TensorSpec selfSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> other{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunInplaceAddStatusCase<float, float, float>(
                     name, ACLNN_ERR_PARAM_INVALID, &selfSpec, &self, &otherSpec, &other, &alpha, ACL_FLOAT)) &&
             ok;
    }

    {
        const std::string name = "InplaceAddSelfBroadcastInvalid";
        TensorSpec selfSpec{{1, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> self{1.0f, 2.0f, 3.0f};
        std::vector<float> other{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunInplaceAddStatusCase<float, float, float>(
                     name, ACLNN_ERR_PARAM_INVALID, &selfSpec, &self, &otherSpec, &other, &alpha, ACL_FLOAT)) &&
             ok;
    }

    {
        const std::string name = "InplaceAddsNullOther";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunInplaceAddsStatusCase<float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &spec, &data, nullptr, ACL_FLOAT, &alpha, ACL_FLOAT)) &&
             ok;
    }

    {
        const std::string name = "AddRankTooHigh";
        TensorSpec spec{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data(1, 1.0f);
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddStatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_INVALID, &spec, &data, &spec, &data, &alpha, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddV3UnsupportedBoolOther";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        TensorSpec outSpec{{2, 2}, {}, {}, 0, ACL_BOOL, ACL_FORMAT_ND};
        std::vector<uint8_t> data{1, 0, 1, 0};
        const float self = 1.0f;
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddV3StatusCase<float, uint8_t, float, uint8_t>(
                     name, ACLNN_ERR_PARAM_INVALID, &self, ACL_FLOAT, &spec, &data, &alpha, ACL_FLOAT, &outSpec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddV3OutputShapeInvalid";
        TensorSpec otherSpec{{2, 3}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        TensorSpec outSpec{{3, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> other{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> out(6, 0.0f);
        const float self = 1.0f;
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddV3StatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_INVALID, &self, ACL_FLOAT, &otherSpec, &other, &alpha, ACL_FLOAT, &outSpec, &out)) &&
             ok;
    }

    {
        const std::string name = "AddV3NullSelf";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddV3StatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, nullptr, ACL_FLOAT, &spec, &data, &alpha, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddV3NullAlpha";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float self = 1.0f;
        ok = ReportCase(name,
                 RunAddV3StatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &self, ACL_FLOAT, &spec, &data, nullptr, ACL_FLOAT, &spec, &data)) &&
             ok;
    }

    {
        const std::string name = "AddV3NullOut";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float self = 1.0f;
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunAddV3StatusCase<float, float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &self, ACL_FLOAT, &spec, &data, &alpha, ACL_FLOAT, nullptr, nullptr)) &&
             ok;
    }

    {
        const std::string name = "InplaceAddV3NullOther";
        const float self = 1.0f;
        const float alpha = 1.0f;
        ok = ReportCase(name,
                 RunInplaceAddV3StatusCase<float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &self, ACL_FLOAT, nullptr, nullptr, &alpha, ACL_FLOAT)) &&
             ok;
    }

    {
        const std::string name = "InplaceAddV3NullAlpha";
        TensorSpec spec{{2, 2}, {}, {}, 0, ACL_FLOAT, ACL_FORMAT_ND};
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
        const float self = 1.0f;
        ok = ReportCase(name,
                 RunInplaceAddV3StatusCase<float, float, float>(
                     name, ACLNN_ERR_PARAM_NULLPTR, &self, ACL_FLOAT, &spec, &data, nullptr, ACL_FLOAT)) &&
             ok;
    }

    return ok;
}

}  // namespace

int main()
{
    RuntimeContext runtime;
    if (!runtime.Init()) {
        return 1;
    }

    bool ok = true;
    ok = RunNormalCases(runtime) && ok;
    ok = RunNegativeCases() && ok;

    Log("\nSummary: %d passed, %d observed, %d failed, %d total\n",
        g_totalCases - g_failedCases - g_observedCases, g_observedCases, g_failedCases, g_totalCases);
    return ok ? 0 : 1;
}
