/**
 * Extended end-to-end tests for aclnnCumsum / aclnnCumsumV2.
 * Target: Ascend 910_93, CANN ops-math math/cumsum example.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static int g_passed = 0;
static int g_failed = 0;

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1; // scalar tensor
    }
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

static int64_t NormalizeDim(const std::vector<int64_t>& shape, int64_t dim)
{
    int64_t dimNum = static_cast<int64_t>(shape.size());
    if (dimNum == 0) {
        dimNum = 1;
    }
    return dim < 0 ? dim + dimNum : dim;
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

static std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() >= 2) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }
    return strides;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
{
    const int64_t elemNum = GetShapeSize(shape);
    const size_t size = static_cast<size_t>(elemNum) * sizeof(T);
    *deviceAddr = nullptr;
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
    }
    std::vector<int64_t> strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
        shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

static uint32_t FloatBits(float x)
{
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static float BitsFloat(uint32_t u)
{
    float x = 0.0f;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

static uint16_t FloatToHalf(float value)
{
    uint32_t f = FloatBits(value);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((f >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = f & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> static_cast<uint32_t>(1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return BitsFloat(f);
}

static uint16_t FloatToBf16(float value)
{
    uint32_t u = FloatBits(value);
    return static_cast<uint16_t>((u + 0x8000u) >> 16);
}

static float Bf16ToFloat(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    return BitsFloat(u);
}

template <typename T>
static T CastStorage(double x, aclDataType dtype)
{
    if (dtype == ACL_FLOAT16) {
        return static_cast<T>(FloatToHalf(static_cast<float>(x)));
    }
    if (dtype == ACL_BF16) {
        return static_cast<T>(FloatToBf16(static_cast<float>(x)));
    }
    return static_cast<T>(x);
}

template <typename T>
static double StorageToDouble(T x, aclDataType dtype)
{
    if (dtype == ACL_FLOAT16) {
        return static_cast<double>(HalfToFloat(static_cast<uint16_t>(x)));
    }
    if (dtype == ACL_BF16) {
        return static_cast<double>(Bf16ToFloat(static_cast<uint16_t>(x)));
    }
    return static_cast<double>(x);
}

static bool IsFloatType(aclDataType dtype)
{
    return dtype == ACL_FLOAT || dtype == ACL_FLOAT16 || dtype == ACL_BF16 || dtype == ACL_DOUBLE;
}

static void GetTolerance(aclDataType dtype, double* atol, double* rtol)
{
    if (dtype == ACL_FLOAT16) {
        *atol = 5e-2;
        *rtol = 5e-2;
    } else if (dtype == ACL_BF16) {
        *atol = 1e-1;
        *rtol = 1e-1;
    } else if (dtype == ACL_FLOAT) {
        *atol = 2e-3;
        *rtol = 2e-3;
    } else if (dtype == ACL_DOUBLE) {
        *atol = 1e-8;
        *rtol = 1e-8;
    } else {
        *atol = 0.0;
        *rtol = 0.0;
    }
}

static std::vector<double> CpuCumsum(const std::vector<double>& input, const std::vector<int64_t>& shape, int64_t dim,
    bool exclusive, bool reverse)
{
    const int64_t total = GetShapeSize(shape);
    std::vector<double> out(total, 0.0);
    if (total == 0) {
        return out;
    }
    if (shape.empty()) {
        out[0] = exclusive ? 0.0 : input[0];
        return out;
    }

    int64_t axis = NormalizeDim(shape, dim);
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[i];
    }
    int64_t axisLen = shape[axis];
    int64_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) {
        inner *= shape[i];
    }

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            double sum = 0.0;
            if (!reverse) {
                for (int64_t a = 0; a < axisLen; ++a) {
                    int64_t idx = (o * axisLen + a) * inner + in;
                    if (exclusive) {
                        out[idx] = sum;
                        sum += input[idx];
                    } else {
                        sum += input[idx];
                        out[idx] = sum;
                    }
                }
            } else {
                for (int64_t a = axisLen - 1; a >= 0; --a) {
                    int64_t idx = (o * axisLen + a) * inner + in;
                    if (exclusive) {
                        out[idx] = sum;
                        sum += input[idx];
                    } else {
                        sum += input[idx];
                        out[idx] = sum;
                    }
                }
            }
        }
    }
    return out;
}

static std::vector<double> MakeInputData(const std::vector<int64_t>& shape, const std::string& pattern)
{
    const int64_t total = GetShapeSize(shape);
    std::vector<double> data(static_cast<size_t>(total), 0.0);
    for (int64_t i = 0; i < total; ++i) {
        if (pattern == "ones") {
            data[i] = 1.0;
        } else if (pattern == "small") {
            data[i] = 0.1;
        } else if (pattern == "mixed_sign") {
            data[i] = (i % 2 == 0) ? 3.0 : -2.0;
        } else if (pattern == "large_small") {
            data[i] = (i % 2 == 0) ? 10000.0 : 0.25;
        } else {
            data[i] = static_cast<double>((i % 13) - 6);
        }
    }
    return data;
}

template <typename StorageT>
static bool CompareAndPrint(const std::string& name, const std::vector<StorageT>& actualRaw,
    const std::vector<double>& expected, aclDataType dtype)
{
    double atol = 0.0;
    double rtol = 0.0;
    GetTolerance(dtype, &atol, &rtol);
    double maxErr = 0.0;
    int64_t maxIdx = -1;
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        double actual = StorageToDouble(actualRaw[i], dtype);
        double err = std::fabs(actual - expected[i]);
        if (err > maxErr) {
            maxErr = err;
            maxIdx = static_cast<int64_t>(i);
        }
        if (IsFloatType(dtype)) {
            double bound = atol + rtol * std::fabs(expected[i]);
            if (err > bound) {
                ok = false;
            }
        } else {
            if (actual != expected[i]) {
                ok = false;
            }
        }
    }
    LOG_PRINT("  Max error: %.10f at index %ld\n", maxErr, maxIdx);
    if (!expected.empty()) {
        LOG_PRINT("  Sample expected[0]=%.10f actual[0]=%.10f\n", expected[0], StorageToDouble(actualRaw[0], dtype));
        size_t last = expected.size() - 1;
        LOG_PRINT("  Sample expected[last]=%.10f actual[last]=%.10f\n", expected[last], StorageToDouble(actualRaw[last], dtype));
    }
    LOG_PRINT("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) {
        ++g_passed;
    } else {
        ++g_failed;
    }
    return ok;
}

template <typename StorageT>
static bool RunCumsumCase(const std::string& name, aclrtStream stream, aclDataType dtype,
    const std::vector<int64_t>& shape, int64_t dim, bool useV2, bool exclusive, bool reverse,
    const std::string& pattern)
{
    LOG_PRINT("\nTest: %s\n", name.c_str());
    const int64_t total = GetShapeSize(shape);
    std::vector<double> inputDouble = MakeInputData(shape, pattern);
    std::vector<StorageT> input(static_cast<size_t>(total));
    std::vector<StorageT> output(static_cast<size_t>(total), static_cast<StorageT>(0));
    for (int64_t i = 0; i < total; ++i) {
        input[static_cast<size_t>(i)] = CastStorage<StorageT>(inputDouble[static_cast<size_t>(i)], dtype);
        inputDouble[static_cast<size_t>(i)] = StorageToDouble(input[static_cast<size_t>(i)], dtype);
    }

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(input, shape, &selfDeviceAddr, dtype, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create self tensor failed\n"); ++g_failed; return false);
    ret = CreateAclTensor(output, shape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create out tensor failed\n"); ++g_failed; return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    }
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWorkspaceSize failed. ERROR: %d\n", ret);
        ++g_failed;
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
        if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] allocate workspace failed. ERROR: %d\n", ret);
            ++g_failed;
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
            if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
            return false;
        }
    }

    ret = useV2 ? aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream)
                : aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] Execute failed. ERROR: %d\n", ret);
        ++g_failed;
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
        if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        ++g_failed;
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
        if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
        return false;
    }

    std::vector<StorageT> result(static_cast<size_t>(total), static_cast<StorageT>(0));
    const size_t bytes = static_cast<size_t>(total) * sizeof(StorageT);
    if (bytes > 0) {
        ret = aclrtMemcpy(result.data(), bytes, outDeviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] copy result D2H failed. ERROR: %d\n", ret);
            ++g_failed;
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
            if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
            if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
            return false;
        }
    }

    std::vector<double> expected = CpuCumsum(inputDouble, shape, dim, exclusive && useV2, reverse && useV2);
    bool ok = CompareAndPrint(name, result, expected, dtype);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}



template <typename SelfT, typename OutT>
static bool RunCumsumCastCase(const std::string& name, aclrtStream stream, aclDataType selfDtype, aclDataType outDtype,
    const std::vector<int64_t>& shape, int64_t dim, bool useV2, bool exclusive, bool reverse,
    const std::string& pattern)
{
    LOG_PRINT("\nTest: %s\n", name.c_str());
    const int64_t total = GetShapeSize(shape);
    std::vector<double> inputDouble = MakeInputData(shape, pattern);
    std::vector<SelfT> input(static_cast<size_t>(total));
    std::vector<OutT> output(static_cast<size_t>(total), static_cast<OutT>(0));
    for (int64_t i = 0; i < total; ++i) {
        input[static_cast<size_t>(i)] = CastStorage<SelfT>(inputDouble[static_cast<size_t>(i)], selfDtype);
        inputDouble[static_cast<size_t>(i)] = StorageToDouble(input[static_cast<size_t>(i)], selfDtype);
    }

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(input, shape, &selfDeviceAddr, selfDtype, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create self tensor failed\n"); ++g_failed; return false);
    ret = CreateAclTensor(output, shape, &outDeviceAddr, outDtype, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create out tensor failed\n"); ++g_failed; return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self, dim, outDtype, out, &workspaceSize, &executor);
    }
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWorkspaceSize failed. ERROR: %d\n", ret);
        ++g_failed;
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
        if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] allocate workspace failed. ERROR: %d\n", ret);
            ++g_failed;
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
            if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
            return false;
        }
    }

    ret = useV2 ? aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream)
                : aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] Execute failed. ERROR: %d\n", ret);
        ++g_failed;
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
        if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        ++g_failed;
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
        if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
        return false;
    }

    std::vector<OutT> result(static_cast<size_t>(total), static_cast<OutT>(0));
    const size_t bytes = static_cast<size_t>(total) * sizeof(OutT);
    if (bytes > 0) {
        ret = aclrtMemcpy(result.data(), bytes, outDeviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] copy result D2H failed. ERROR: %d\n", ret);
            ++g_failed;
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
            if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
            if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
            return false;
        }
    }

    std::vector<double> expected = CpuCumsum(inputDouble, shape, dim, exclusive && useV2, reverse && useV2);
    bool ok = CompareAndPrint(name, result, expected, outDtype);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}

static bool EmptyTensorCaseV2(const std::string& name)
{
    LOG_PRINT("\nTest: %s\n", name.c_str());
    std::vector<int64_t> shape = {2, 0, 4};
    std::vector<float> empty;
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(empty, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create empty self failed\n"); ++g_failed; return false);
    ret = CreateAclTensor(empty, shape, &outDeviceAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create empty out failed\n"); ++g_failed; return false);
    uint64_t workspaceSize = 123;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(self, 1, true, true, out, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS && workspaceSize == 0);
    LOG_PRINT("  Return code: %d, workspaceSize: %lu\n", ret, workspaceSize);
    LOG_PRINT("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) ++g_passed; else ++g_failed;
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    return ok;
}

static bool ExpectDtypeArgFailure(const std::string& name)
{
    LOG_PRINT("\nNegative Test: %s\n", name.c_str());
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> in = {1, 2, 3, 4, 5, 6};
    std::vector<float> outHost(6, 0);
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(in, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create self failed\n"); ++g_failed; return false);
    ret = CreateAclTensor(outHost, shape, &outDeviceAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create out failed\n"); ++g_failed; return false);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, 1, ACL_FLOAT16, out, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    LOG_PRINT("  Return code: %d\n", ret);
    LOG_PRINT("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) ++g_passed; else ++g_failed;
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}

static bool ExpectV2DtypeMismatchFailure(const std::string& name)
{
    LOG_PRINT("\nNegative Test: %s\n", name.c_str());
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> in = {1, 2, 3, 4, 5, 6};
    std::vector<int32_t> outHost(6, 0);
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(in, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create self failed\n"); ++g_failed; return false);
    ret = CreateAclTensor(outHost, shape, &outDeviceAddr, ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create out failed\n"); ++g_failed; return false);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    LOG_PRINT("  Return code: %d\n", ret);
    LOG_PRINT("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) ++g_passed; else ++g_failed;
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}

template <typename StorageT>
static bool ExpectWorkspaceFailure(const std::string& name, aclDataType dtype, const std::vector<int64_t>& selfShape,
    const std::vector<int64_t>& outShape, int64_t dim, bool useNullSelf, bool useNullOut, bool useV2)
{
    LOG_PRINT("\nNegative Test: %s\n", name.c_str());
    std::vector<double> inD = MakeInputData(selfShape, "seq");
    std::vector<StorageT> in(static_cast<size_t>(GetShapeSize(selfShape)));
    std::vector<StorageT> outHost(static_cast<size_t>(GetShapeSize(outShape)), static_cast<StorageT>(0));
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = CastStorage<StorageT>(inD[i], dtype);
    }

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    if (!useNullSelf) {
        int ret = CreateAclTensor(in, selfShape, &selfDeviceAddr, dtype, &self);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create self failed\n"); ++g_failed; return false);
    }
    if (!useNullOut) {
        int ret = CreateAclTensor(outHost, outShape, &outDeviceAddr, dtype, &out);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create out failed\n"); ++g_failed; return false);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = useV2 ? aclnnCumsumV2GetWorkspaceSize(self, dim, false, false, out, &workspaceSize, &executor)
                    : aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    LOG_PRINT("  Return code: %d\n", ret);
    LOG_PRINT("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) {
        ++g_passed;
    } else {
        ++g_failed;
    }

    if (self != nullptr) aclDestroyTensor(self);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
    if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
    return ok;
}

static bool EmptyTensorCase(const std::string& name)
{
    LOG_PRINT("\nTest: %s\n", name.c_str());
    std::vector<int64_t> shape = {0, 4};
    std::vector<float> empty;
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = CreateAclTensor(empty, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create empty self failed\n"); ++g_failed; return false);
    ret = CreateAclTensor(empty, shape, &outDeviceAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  [FAIL] create empty out failed\n"); ++g_failed; return false);
    uint64_t workspaceSize = 123;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, 1, ACL_FLOAT, out, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS && workspaceSize == 0);
    LOG_PRINT("  Return code: %d, workspaceSize: %lu\n", ret, workspaceSize);
    LOG_PRINT("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) ++g_passed; else ++g_failed;
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    return ok;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Basic API and V2 semantic coverage.
    RunCumsumCase<float>("float32_basic_dim0_shape_2x2", stream, ACL_FLOAT, {2, 2}, 0, false, false, false, "seq");
    RunCumsumCase<float>("float32_basic_dim1_shape_2x3", stream, ACL_FLOAT, {2, 3}, 1, false, false, false, "seq");
    RunCumsumCase<float>("float32_negative_dim", stream, ACL_FLOAT, {2, 3}, -1, true, false, false, "seq");
    RunCumsumCase<float>("v2_normal", stream, ACL_FLOAT, {1, 4}, 1, true, false, false, "ones");
    RunCumsumCase<float>("v2_exclusive", stream, ACL_FLOAT, {1, 4}, 1, true, true, false, "ones");
    RunCumsumCase<float>("v2_reverse", stream, ACL_FLOAT, {1, 4}, 1, true, false, true, "ones");
    RunCumsumCase<float>("v2_exclusive_reverse", stream, ACL_FLOAT, {1, 4}, 1, true, true, true, "ones");
    RunCumsumCase<float>("scalar_0dim", stream, ACL_FLOAT, {}, 0, false, false, false, "seq");

    // Boundary dim success cases: cover dim == -rank and dim == rank - 1 on both API variants.
    RunCumsumCase<float>("float32_dim_negative_rank_boundary", stream, ACL_FLOAT, {2, 3, 4}, -3, false, false, false, "seq");
    RunCumsumCase<float>("float32_dim_last_boundary_v2", stream, ACL_FLOAT, {2, 3, 4}, 2, true, false, false, "seq");
    RunCumsumCase<float>("float32_dim_negative_rank_v2_exclusive", stream, ACL_FLOAT, {2, 3, 4}, -3, true, true, false, "mixed_sign");

    // Integer tiling coverage. Keep values small to avoid overflow and dtype wraparound.
    RunCumsumCase<int32_t>("int32_basic_last_dim", stream, ACL_INT32, {4, 8}, 1, false, false, false, "seq");
    RunCumsumCase<int32_t>("int32_axis0_v2", stream, ACL_INT32, {8, 4}, 0, true, false, false, "seq");
    RunCumsumCase<int32_t>("int32_middle_axis_v2", stream, ACL_INT32, {4, 16, 8}, 1, true, false, false, "mixed_sign");
    RunCumsumCase<int64_t>("int64_basic", stream, ACL_INT64, {2, 32}, 1, true, false, false, "seq");
    RunCumsumCase<int8_t>("int8_basic", stream, ACL_INT8, {8, 16}, 1, true, false, false, "mixed_sign");
    RunCumsumCase<uint8_t>("uint8_basic", stream, ACL_UINT8, {8, 16}, 1, true, false, false, "ones");

    RunCumsumCase<int32_t>("int32_v2_reverse_last_dim", stream, ACL_INT32, {4, 32}, 1, true, false, true, "mixed_sign");
    RunCumsumCase<int32_t>("int32_v2_exclusive_axis0", stream, ACL_INT32, {32, 4}, 0, true, true, false, "seq");
    RunCumsumCase<int64_t>("int64_v2_exclusive_reverse", stream, ACL_INT64, {2, 16}, 1, true, true, true, "seq");

    // Float tiling branch coverage. These shapes are intended to stress N/R/M split decisions.
    RunCumsumCase<float>("float32_rn_greater_shape_64x64x8", stream, ACL_FLOAT, {64, 64, 8}, 1, false, false, false, "small");
    RunCumsumCase<float>("float32_n_greater_shape_64x8x128", stream, ACL_FLOAT, {64, 8, 128}, 1, false, false, false, "mixed_sign");
    RunCumsumCase<float>("float32_r_large_shape_16x1024x16", stream, ACL_FLOAT, {16, 1024, 16}, 1, false, false, false, "small");
    RunCumsumCase<float>("float32_borrow_r_shape_1x4096x128", stream, ACL_FLOAT, {1, 4096, 128}, 1, false, false, false, "small");

    RunCumsumCase<float>("float32_v2_exclusive_large_rn", stream, ACL_FLOAT, {32, 128, 16}, 1, true, true, false, "small");
    RunCumsumCase<float>("float32_v2_reverse_large_n", stream, ACL_FLOAT, {32, 8, 256}, 1, true, false, true, "mixed_sign");
    RunCumsumCase<float>("float32_v2_exclusive_reverse_borrow_r", stream, ACL_FLOAT, {1, 2048, 64}, 1, true, true, true, "small");
    RunCumsumCase<uint16_t>("float16_cast_shape_64x512x8", stream, ACL_FLOAT16, {64, 512, 8}, 1, false, false, false, "small");
    RunCumsumCase<uint16_t>("bf16_cast_shape_32x512x8", stream, ACL_BF16, {32, 512, 8}, 1, false, false, false, "small");
    RunCumsumCase<float>("float32_axis0_large_inner_shape_8x4096", stream, ACL_FLOAT, {8, 4096}, 0, false, false, false, "small");
    RunCumsumCase<float>("float32_last_axis_long_shape_256x1024", stream, ACL_FLOAT, {256, 1024}, 1, false, false, false, "small");
    RunCumsumCase<float>("float32_7d_negative_dim_shape", stream, ACL_FLOAT, {2, 1, 3, 1, 4, 1, 5}, -5, false, false, false, "seq");
    RunCumsumCase<uint16_t>("fp16_6d_dim4_shape", stream, ACL_FLOAT16, {7, 1, 16, 16, 1, 16}, 4, false, false, false, "small");
    RunCumsumCase<uint16_t>("bf16_7d_negative_dim_shape", stream, ACL_BF16, {1, 1, 17, 7, 17, 1, 15}, -5, false, false, false, "small");

    // API cast and AiCPU fallback coverage. DOUBLE and INT16 are accepted by aclnn front end but route to AiCPU in cumsum.cpp.
    RunCumsumCastCase<int32_t, float>("api_cast_int32_to_float", stream, ACL_INT32, ACL_FLOAT, {4, 8}, 1, false, false, false, "seq");
    RunCumsumCastCase<uint16_t, float>("api_cast_fp16_to_float", stream, ACL_FLOAT16, ACL_FLOAT, {16, 64}, 1, false, false, false, "small");

    //RunCumsumCastCase<int8_t, int32_t>("api_cast_int8_to_int32", stream, ACL_INT8, ACL_INT32, {8, 32}, 1, false, false, false, "mixed_sign");
    //RunCumsumCastCase<uint16_t, float>("api_cast_bf16_to_float", stream, ACL_BF16, ACL_FLOAT, {8, 128}, 1, false, false, false, "small");
    RunCumsumCase<double>("double_aicpu_basic", stream, ACL_DOUBLE, {2, 8}, 1, false, false, false, "small");
    RunCumsumCase<double>("double_aicpu_v2_reverse", stream, ACL_DOUBLE, {1, 16}, 1, true, false, true, "small");
    RunCumsumCase<int16_t>("int16_aicpu_basic", stream, ACL_INT16, {4, 16}, 1, false, false, false, "seq");
    RunCumsumCase<int16_t>("int16_aicpu_v2_exclusive", stream, ACL_INT16, {2, 4, 8}, 2, true, true, false, "seq");

    // Extra integer tiling pressure: right-axis split, left-axis split, R-axis group split, and 7-D shapes.
    RunCumsumCase<int32_t>("int32_right_large_axis0_shape_4x2048", stream, ACL_INT32, {4, 2048}, 0, true, false, false, "seq");
    RunCumsumCase<int32_t>("int32_left_large_last_axis_shape_4096x4", stream, ACL_INT32, {4096, 4}, 1, true, false, false, "seq");
    RunCumsumCase<int32_t>("int32_r_group_candidate_shape_1x4096x4", stream, ACL_INT32, {1, 4096, 4}, 1, true, true, true, "mixed_sign");
    RunCumsumCase<int8_t>("int8_7d_axis0_shape", stream, ACL_INT8, {1, 1, 1, 16, 15, 15, 8}, 0, true, false, false, "mixed_sign");
    RunCumsumCase<uint8_t>("uint8_7d_last_axis_shape", stream, ACL_UINT8, {1, 16, 1, 1, 8, 16, 15}, 6, true, true, false, "ones");

    RunCumsumCase<int32_t>("int32_axis_len_one_middle", stream, ACL_INT32, {64, 1, 64}, 1, true, false, false, "seq");
    RunCumsumCase<int32_t>("int32_inner_one_long_axis", stream, ACL_INT32, {16, 2048}, 1, true, false, false, "mixed_sign");
    RunCumsumCase<int64_t>("int64_axis0_large_inner", stream, ACL_INT64, {8, 512}, 0, true, false, false, "seq");

    // Cube path in aclnnCumsumGetWorkspaceSize: only supports last dim, large batch/channel, float/fp16/bf16 on 910B/910_93.
    RunCumsumCase<uint16_t>("float16_cube_path_shape_12800x512", stream, ACL_FLOAT16, {12800, 512}, 1, false, false,
        false, "ones");

    // Precision-analysis-oriented cases. They should pass with wider tolerance and provide max error information.
    RunCumsumCase<float>("precision_float32_0p1_len10000", stream, ACL_FLOAT, {10000}, 0, false, false, false, "small");
    RunCumsumCase<uint16_t>("precision_float16_0p1_len4096", stream, ACL_FLOAT16, {4096}, 0, false, false, false, "small");
    RunCumsumCase<float>("precision_float32_large_small", stream, ACL_FLOAT, {2048}, 0, false, false, false, "large_small");
    RunCumsumCase<float>("precision_float32_reverse_small", stream, ACL_FLOAT, {4096}, 0, true, false, true, "small");

    // Empty tensor and negative parameter coverage. These cases do not execute kernel.
    EmptyTensorCase("empty_tensor_workspace_zero");
    EmptyTensorCaseV2("empty_tensor_v2_workspace_zero");
    ExpectDtypeArgFailure("dtype_argument_mismatch_out_float_dtype_fp16");
    ExpectV2DtypeMismatchFailure("v2_self_out_dtype_mismatch");
    ExpectWorkspaceFailure<uint8_t>("unsupported_bool_dtype", ACL_BOOL, {2, 3}, {2, 3}, 1, false, false, false);
    ExpectWorkspaceFailure<float>("null_self", ACL_FLOAT, {2, 3}, {2, 3}, 1, true, false, false);
    ExpectWorkspaceFailure<float>("null_out", ACL_FLOAT, {2, 3}, {2, 3}, 1, false, true, false);
    ExpectWorkspaceFailure<float>("dim_out_of_range_positive", ACL_FLOAT, {2, 3}, {2, 3}, 2, false, false, false);
    ExpectWorkspaceFailure<float>("dim_out_of_range_negative", ACL_FLOAT, {2, 3}, {2, 3}, -3, false, false, true);
    ExpectWorkspaceFailure<float>("shape_mismatch", ACL_FLOAT, {2, 3}, {2, 4}, 1, false, false, false);
    ExpectWorkspaceFailure<float>("null_self_v2", ACL_FLOAT, {2, 3}, {2, 3}, 1, true, false, true);
    ExpectWorkspaceFailure<float>("null_out_v2", ACL_FLOAT, {2, 3}, {2, 3}, 1, false, true, true);
    ExpectWorkspaceFailure<float>("shape_mismatch_v2", ACL_FLOAT, {2, 3}, {3, 2}, 1, false, false, true);
    ExpectWorkspaceFailure<float>("dim_equal_negative_rank_minus_one_v1", ACL_FLOAT, {2, 3, 4}, {2, 3, 4}, -4, false, false, false);
    //ExpectWorkspaceFailure<uint16_t>("unsupported_uint16_dtype", ACL_UINT16, {2, 3}, {2, 3}, 1, false, false, false);
    ExpectWorkspaceFailure<float>("max_dim_exceed", ACL_FLOAT, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1},
        0, false, false, false);

    LOG_PRINT("\nSummary: %d passed, %d failed\n", g_passed, g_failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return g_failed == 0 ? 0 : 1;
}
