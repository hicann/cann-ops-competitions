#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#define LOG_PRINT(message, ...) do { std::printf(message, ##__VA_ARGS__); } while (0)

static int g_total = 0;
static int g_pass = 0;
static int g_fail = 0;

static const double FP32_ATOL = 1e-5;
static const double FP32_RTOL = 1e-5;
static const double FP16_ATOL = 1e-3;
static const double FP16_RTOL = 1e-3;
static const double BF16_ATOL = 1e-2;
static const double BF16_RTOL = 1e-2;
static const double FP64_ATOL = 1e-10;
static const double FP64_RTOL = 1e-10;

static int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        n *= shape[i];
    }
    return n;
}

static std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
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

static int64_t NormDim(int64_t dim, const std::vector<int64_t>& shape)
{
    int64_t rank = static_cast<int64_t>(shape.size());
    if (rank == 0) {
        rank = 1;
    }
    return dim < 0 ? dim + rank : dim;
}

static void PrintCaseResult(const std::string& name, bool ok)
{
    ++g_total;
    if (ok) {
        ++g_pass;
        LOG_PRINT("[PASS] %s\n", name.c_str());
    } else {
        ++g_fail;
        LOG_PRINT("[FAIL] %s\n", name.c_str());
    }
}

static int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed, ret=%d\n", static_cast<int>(ret));
        return static_cast<int>(ret);
    }

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed, ret=%d\n", static_cast<int>(ret));
        return static_cast<int>(ret);
    }

    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed, ret=%d\n", static_cast<int>(ret));
        return static_cast<int>(ret);
    }

    return 0;
}

static void FinalizeAcl(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

struct TensorResource {
    aclTensor* tensor;
    void* device;
    std::vector<int64_t> shape;
    std::vector<int64_t> storageShape;
    std::vector<int64_t> strides;
    int64_t offset;
    size_t bytes;

    TensorResource() : tensor(nullptr), device(nullptr), offset(0), bytes(0) {}
};

static void DestroyTensor(TensorResource* res)
{
    if (res == nullptr) {
        return;
    }
    if (res->tensor != nullptr) {
        aclDestroyTensor(res->tensor);
        res->tensor = nullptr;
    }
    if (res->device != nullptr) {
        (void)aclrtFree(res->device);
        res->device = nullptr;
    }
    res->bytes = 0;
}

template <typename T>
static int CreateTensor(const std::vector<T>& hostData,
                        const std::vector<int64_t>& shape,
                        aclDataType dtype,
                        TensorResource* res)
{
    if (res == nullptr) {
        return -1;
    }

    res->shape = shape;
    res->storageShape = shape;
    res->strides = MakeStrides(shape);
    res->offset = 0;

    int64_t elemNum = ShapeSize(shape);
    size_t realBytes = static_cast<size_t>(std::max<int64_t>(elemNum, 0)) * sizeof(T);
    res->bytes = std::max<size_t>(realBytes, 1);

    aclError ret = aclrtMalloc(&res->device, res->bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed, ret=%d, bytes=%zu\n", static_cast<int>(ret), res->bytes);
        return static_cast<int>(ret);
    }

    if (realBytes > 0) {
        ret = aclrtMemcpy(res->device, realBytes, hostData.data(), realBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtMemcpy H2D failed, ret=%d\n", static_cast<int>(ret));
            DestroyTensor(res);
            return static_cast<int>(ret);
        }
    }

    res->tensor = aclCreateTensor(shape.data(),
                                  shape.size(),
                                  dtype,
                                  res->strides.data(),
                                  0,
                                  aclFormat::ACL_FORMAT_ND,
                                  shape.data(),
                                  shape.size(),
                                  res->device);
    if (res->tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed\n");
        DestroyTensor(res);
        return -1;
    }

    return 0;
}

template <typename T>
static int CopyOut(const TensorResource& res, std::vector<T>* hostData)
{
    int64_t elemNum = ShapeSize(res.storageShape);
    hostData->assign(static_cast<size_t>(std::max<int64_t>(elemNum, 0)), static_cast<T>(0));
    if (elemNum <= 0) {
        return 0;
    }

    aclError ret = aclrtMemcpy(hostData->data(),
                               static_cast<size_t>(elemNum) * sizeof(T),
                               res.device,
                               static_cast<size_t>(elemNum) * sizeof(T),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy D2H failed, ret=%d\n", static_cast<int>(ret));
        return static_cast<int>(ret);
    }

    return 0;
}

static uint16_t FloatToFp16(float value)
{
    uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000u;
        uint32_t shifted = mant >> static_cast<uint32_t>(1 - exp);
        if ((shifted & 0x00001000u) != 0) {
            shifted += 0x00002000u;
        }
        return static_cast<uint16_t>(sign | (shifted >> 13));
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? ((mant >> 13) | 1u) : 0u));
    }

    if ((mant & 0x00001000u) != 0) {
        mant += 0x00002000u;
        if ((mant & 0x00800000u) != 0) {
            mant = 0;
            ++exp;
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out = 0;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static uint16_t FloatToBf16(float value)
{
    uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;
    return static_cast<uint16_t>(x >> 16);
}

static float Bf16ToFloat(uint16_t h)
{
    uint32_t x = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

static std::vector<uint16_t> EncodeFp16(const std::vector<float>& input)
{
    std::vector<uint16_t> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = FloatToFp16(input[i]);
    }
    return out;
}

static std::vector<uint16_t> EncodeBf16(const std::vector<float>& input)
{
    std::vector<uint16_t> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = FloatToBf16(input[i]);
    }
    return out;
}

static std::vector<double> CpuCumsumDouble(const std::vector<double>& input,
                                           const std::vector<int64_t>& shape,
                                           int64_t dim,
                                           bool exclusive,
                                           bool reverse)
{
    std::vector<double> output(input.size(), 0.0);
    if (input.empty()) {
        return output;
    }

    int64_t axis = NormDim(dim, shape);
    int64_t m = 1;
    int64_t r = shape.empty() ? 1 : shape[static_cast<size_t>(axis)];
    int64_t n = 1;

    for (int64_t i = 0; i < axis; ++i) {
        m *= shape[static_cast<size_t>(i)];
    }
    for (size_t i = static_cast<size_t>(axis + 1); i < shape.size(); ++i) {
        n *= shape[i];
    }

    for (int64_t mi = 0; mi < m; ++mi) {
        for (int64_t ni = 0; ni < n; ++ni) {
            double acc = 0.0;
            if (!reverse) {
                for (int64_t ri = 0; ri < r; ++ri) {
                    size_t idx = static_cast<size_t>((mi * r + ri) * n + ni);
                    double old = acc;
                    acc += input[idx];
                    output[idx] = exclusive ? old : acc;
                }
            } else {
                for (int64_t ri = r - 1; ri >= 0; --ri) {
                    size_t idx = static_cast<size_t>((mi * r + ri) * n + ni);
                    double old = acc;
                    acc += input[idx];
                    output[idx] = exclusive ? old : acc;
                }
            }
        }
    }

    return output;
}

template <typename T, typename U>
static std::vector<T> CpuCumsumWrap(const std::vector<T>& input,
                                    const std::vector<int64_t>& shape,
                                    int64_t dim,
                                    bool exclusive,
                                    bool reverse)
{
    std::vector<T> output(input.size(), static_cast<T>(0));
    if (input.empty()) {
        return output;
    }

    int64_t axis = NormDim(dim, shape);
    int64_t m = 1;
    int64_t r = shape.empty() ? 1 : shape[static_cast<size_t>(axis)];
    int64_t n = 1;

    for (int64_t i = 0; i < axis; ++i) {
        m *= shape[static_cast<size_t>(i)];
    }
    for (size_t i = static_cast<size_t>(axis + 1); i < shape.size(); ++i) {
        n *= shape[i];
    }

    for (int64_t mi = 0; mi < m; ++mi) {
        for (int64_t ni = 0; ni < n; ++ni) {
            U acc = 0;
            if (!reverse) {
                for (int64_t ri = 0; ri < r; ++ri) {
                    size_t idx = static_cast<size_t>((mi * r + ri) * n + ni);
                    U old = acc;
                    acc = static_cast<U>(acc + static_cast<U>(input[idx]));
                    output[idx] = static_cast<T>(exclusive ? old : acc);
                }
            } else {
                for (int64_t ri = r - 1; ri >= 0; --ri) {
                    size_t idx = static_cast<size_t>((mi * r + ri) * n + ni);
                    U old = acc;
                    acc = static_cast<U>(acc + static_cast<U>(input[idx]));
                    output[idx] = static_cast<T>(exclusive ? old : acc);
                }
            }
        }
    }

    return output;
}

struct CompareStat {
    bool ok;
    size_t worstIndex;
    double maxAbs;
    double maxRel;
    double actual;
    double expected;

    CompareStat() : ok(true), worstIndex(0), maxAbs(0.0), maxRel(0.0), actual(0.0), expected(0.0) {}
};

static CompareStat CompareFloatVec(const std::vector<double>& actual,
                                   const std::vector<double>& expected,
                                   double atol,
                                   double rtol)
{
    CompareStat s;
    if (actual.size() != expected.size()) {
        s.ok = false;
        s.maxAbs = std::numeric_limits<double>::infinity();
        return s;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        double a = actual[i];
        double e = expected[i];
        double absErr = std::fabs(a - e);
        double relErr = absErr / std::max(std::fabs(e), std::numeric_limits<double>::min());
        bool ok = absErr <= atol + rtol * std::fabs(e);

        if (!ok) {
            s.ok = false;
        }

        if (i == 0 || absErr > s.maxAbs) {
            s.worstIndex = i;
            s.maxAbs = absErr;
            s.maxRel = relErr;
            s.actual = a;
            s.expected = e;
        }
    }

    return s;
}

static void PrintStat(const CompareStat& st)
{
    std::ostringstream os;
    os << std::setprecision(17)
       << "  Expected[worst]=" << st.expected
       << ", Actual[worst]=" << st.actual
       << ", MaxAbs=" << st.maxAbs
       << ", MaxRel=" << st.maxRel
       << ", WorstIndex=" << st.worstIndex << "\n";
    LOG_PRINT("%s", os.str().c_str());
}

template <typename T>
static bool CompareIntVec(const std::vector<T>& actual, const std::vector<T>& expected)
{
    if (actual.size() != expected.size()) {
        LOG_PRINT("  size mismatch actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            LOG_PRINT("  mismatch index=%zu actual=%lld expected=%lld\n",
                      i,
                      static_cast<long long>(actual[i]),
                      static_cast<long long>(expected[i]));
            return false;
        }
    }

    LOG_PRINT("  integer exact match, elements=%zu\n", actual.size());
    return true;
}

enum ApiMode {
    API_CUMSUM = 0,
    API_CUMSUM_V2 = 1
};

static bool LaunchCumsum(aclrtStream stream,
                         TensorResource* self,
                         TensorResource* out,
                         int64_t dim,
                         aclDataType dtype,
                         ApiMode mode,
                         bool exclusive,
                         bool reverse,
                         bool expectSuccess)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = static_cast<aclnnStatus>(ACL_SUCCESS);

    if (mode == API_CUMSUM) {
        st = aclnnCumsumGetWorkspaceSize(self == nullptr ? nullptr : self->tensor,
                                         dim,
                                         dtype,
                                         out == nullptr ? nullptr : out->tensor,
                                         &workspaceSize,
                                         &executor);
    } else {
        st = aclnnCumsumV2GetWorkspaceSize(self == nullptr ? nullptr : self->tensor,
                                           dim,
                                           exclusive,
                                           reverse,
                                           out == nullptr ? nullptr : out->tensor,
                                           &workspaceSize,
                                           &executor);
    }

    if (!expectSuccess) {
        LOG_PRINT("  expected failure status=%d\n", static_cast<int>(st));
        return st != ACL_SUCCESS;
    }

    if (st != ACL_SUCCESS) {
        LOG_PRINT("  GetWorkspaceSize failed, status=%d\n", static_cast<int>(st));
        return false;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclError ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  workspace malloc failed, ret=%d\n", static_cast<int>(ret));
            return false;
        }
    }

    if (mode == API_CUMSUM) {
        st = aclnnCumsum(workspace, workspaceSize, executor, stream);
    } else {
        st = aclnnCumsumV2(workspace, workspaceSize, executor, stream);
    }

    if (st != ACL_SUCCESS) {
        LOG_PRINT("  execute failed, status=%d\n", static_cast<int>(st));
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
        }
        return false;
    }

    aclError ret = aclrtSynchronizeStream(stream);
    if (workspace != nullptr) {
        (void)aclrtFree(workspace);
    }

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  synchronize failed, ret=%d\n", static_cast<int>(ret));
        return false;
    }

    return true;
}

static std::vector<float> MakeFloatData(int64_t n, float scale)
{
    std::vector<float> data(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; ++i) {
        int v = static_cast<int>(i % 17) - 8;
        data[static_cast<size_t>(i)] = static_cast<float>(v) * scale;
    }
    return data;
}

static std::vector<double> MakeDoubleData(int64_t n, double scale)
{
    std::vector<double> data(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; ++i) {
        int v = static_cast<int>(i % 19) - 9;
        data[static_cast<size_t>(i)] = static_cast<double>(v) * scale;
    }
    return data;
}

static std::vector<double> ToDouble(const std::vector<float>& input)
{
    std::vector<double> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = static_cast<double>(input[i]);
    }
    return out;
}

static bool RunFp32(aclrtStream stream,
                    const std::string& name,
                    const std::vector<int64_t>& shape,
                    const std::vector<float>& input,
                    int64_t dim,
                    ApiMode mode,
                    bool exclusive,
                    bool reverse)
{
    TensorResource self;
    TensorResource out;

    std::vector<float> zeros(static_cast<size_t>(std::max<int64_t>(ShapeSize(shape), 0)), 0.0f);

    bool ok = (CreateTensor(input, shape, ACL_FLOAT, &self) == 0 &&
               CreateTensor(zeros, shape, ACL_FLOAT, &out) == 0);

    if (ok) {
        ok = LaunchCumsum(stream, &self, &out, dim, ACL_FLOAT, mode, exclusive, reverse, true);
    }

    if (ok) {
        std::vector<float> actualRaw;
        ok = (CopyOut(out, &actualRaw) == 0);

        std::vector<double> actual(actualRaw.begin(), actualRaw.end());
        std::vector<double> expected = CpuCumsumDouble(ToDouble(input), shape, dim, exclusive, reverse);

        CompareStat st = CompareFloatVec(actual, expected, FP32_ATOL, FP32_RTOL);
        PrintStat(st);
        ok = st.ok;
    }

    DestroyTensor(&self);
    DestroyTensor(&out);
    (void)name;
    return ok;
}

static bool RunFp64(aclrtStream stream,
                    const std::string& name,
                    const std::vector<int64_t>& shape,
                    const std::vector<double>& input,
                    int64_t dim,
                    ApiMode mode,
                    bool exclusive,
                    bool reverse)
{
    TensorResource self;
    TensorResource out;

    std::vector<double> zeros(static_cast<size_t>(std::max<int64_t>(ShapeSize(shape), 0)), 0.0);

    bool ok = (CreateTensor(input, shape, ACL_DOUBLE, &self) == 0 &&
               CreateTensor(zeros, shape, ACL_DOUBLE, &out) == 0);

    if (ok) {
        ok = LaunchCumsum(stream, &self, &out, dim, ACL_DOUBLE, mode, exclusive, reverse, true);
    }

    if (ok) {
        std::vector<double> actual;
        ok = (CopyOut(out, &actual) == 0);

        std::vector<double> expected = CpuCumsumDouble(input, shape, dim, exclusive, reverse);

        CompareStat st = CompareFloatVec(actual, expected, FP64_ATOL, FP64_RTOL);
        PrintStat(st);
        ok = st.ok;
    }

    DestroyTensor(&self);
    DestroyTensor(&out);
    (void)name;
    return ok;
}

static bool RunFp16(aclrtStream stream,
                    const std::string& name,
                    const std::vector<int64_t>& shape,
                    const std::vector<float>& inputFloat,
                    int64_t dim,
                    ApiMode mode,
                    bool exclusive,
                    bool reverse)
{
    std::vector<uint16_t> input = EncodeFp16(inputFloat);

    TensorResource self;
    TensorResource out;

    std::vector<uint16_t> zeros(static_cast<size_t>(std::max<int64_t>(ShapeSize(shape), 0)), 0);

    bool ok = (CreateTensor(input, shape, ACL_FLOAT16, &self) == 0 &&
               CreateTensor(zeros, shape, ACL_FLOAT16, &out) == 0);

    if (ok) {
        ok = LaunchCumsum(stream, &self, &out, dim, ACL_FLOAT16, mode, exclusive, reverse, true);
    }

    if (ok) {
        std::vector<uint16_t> actualRaw;
        ok = (CopyOut(out, &actualRaw) == 0);

        std::vector<double> actual(actualRaw.size());
        std::vector<double> inDouble(input.size());

        for (size_t i = 0; i < actualRaw.size(); ++i) {
            actual[i] = static_cast<double>(Fp16ToFloat(actualRaw[i]));
        }
        for (size_t i = 0; i < input.size(); ++i) {
            inDouble[i] = static_cast<double>(Fp16ToFloat(input[i]));
        }

        std::vector<double> expected = CpuCumsumDouble(inDouble, shape, dim, exclusive, reverse);

        CompareStat st = CompareFloatVec(actual, expected, FP16_ATOL, FP16_RTOL);
        PrintStat(st);
        ok = st.ok;
    }

    DestroyTensor(&self);
    DestroyTensor(&out);
    (void)name;
    return ok;
}

static bool RunBf16(aclrtStream stream,
                    const std::string& name,
                    const std::vector<int64_t>& shape,
                    const std::vector<float>& inputFloat,
                    int64_t dim,
                    ApiMode mode,
                    bool exclusive,
                    bool reverse)
{
    std::vector<uint16_t> input = EncodeBf16(inputFloat);

    TensorResource self;
    TensorResource out;

    std::vector<uint16_t> zeros(static_cast<size_t>(std::max<int64_t>(ShapeSize(shape), 0)), 0);

    bool ok = (CreateTensor(input, shape, ACL_BF16, &self) == 0 &&
               CreateTensor(zeros, shape, ACL_BF16, &out) == 0);

    if (ok) {
        ok = LaunchCumsum(stream, &self, &out, dim, ACL_BF16, mode, exclusive, reverse, true);
    }

    if (ok) {
        std::vector<uint16_t> actualRaw;
        ok = (CopyOut(out, &actualRaw) == 0);

        std::vector<double> actual(actualRaw.size());
        std::vector<double> inDouble(input.size());

        for (size_t i = 0; i < actualRaw.size(); ++i) {
            actual[i] = static_cast<double>(Bf16ToFloat(actualRaw[i]));
        }
        for (size_t i = 0; i < input.size(); ++i) {
            inDouble[i] = static_cast<double>(Bf16ToFloat(input[i]));
        }

        std::vector<double> expected = CpuCumsumDouble(inDouble, shape, dim, exclusive, reverse);

        CompareStat st = CompareFloatVec(actual, expected, BF16_ATOL, BF16_RTOL);
        PrintStat(st);
        ok = st.ok;
    }

    DestroyTensor(&self);
    DestroyTensor(&out);
    (void)name;
    return ok;
}

static bool RunInt32(aclrtStream stream,
                     const std::string& name,
                     const std::vector<int64_t>& shape,
                     const std::vector<int32_t>& input,
                     int64_t dim,
                     ApiMode mode,
                     bool exclusive,
                     bool reverse)
{
    TensorResource self;
    TensorResource out;

    std::vector<int32_t> zeros(static_cast<size_t>(std::max<int64_t>(ShapeSize(shape), 0)), 0);

    bool ok = (CreateTensor(input, shape, ACL_INT32, &self) == 0 &&
               CreateTensor(zeros, shape, ACL_INT32, &out) == 0);

    if (ok) {
        ok = LaunchCumsum(stream, &self, &out, dim, ACL_INT32, mode, exclusive, reverse, true);
    }

    if (ok) {
        std::vector<int32_t> actual;
        ok = (CopyOut(out, &actual) == 0);

        std::vector<int32_t> expected = CpuCumsumWrap<int32_t, uint32_t>(input, shape, dim, exclusive, reverse);
        ok = CompareIntVec(actual, expected);
    }

    DestroyTensor(&self);
    DestroyTensor(&out);
    (void)name;
    return ok;
}

template <typename T, typename U>
static bool RunIntLike(aclrtStream stream,
                       const std::string& name,
                       const std::vector<int64_t>& shape,
                       const std::vector<T>& input,
                       aclDataType dtype,
                       int64_t dim,
                       ApiMode mode,
                       bool exclusive,
                       bool reverse)
{
    TensorResource self;
    TensorResource out;

    std::vector<T> zeros(static_cast<size_t>(std::max<int64_t>(ShapeSize(shape), 0)), static_cast<T>(0));

    bool ok = (CreateTensor(input, shape, dtype, &self) == 0 &&
               CreateTensor(zeros, shape, dtype, &out) == 0);

    if (ok) {
        ok = LaunchCumsum(stream, &self, &out, dim, dtype, mode, exclusive, reverse, true);
    }

    if (ok) {
        std::vector<T> actual;
        ok = (CopyOut(out, &actual) == 0);

        std::vector<T> expected = CpuCumsumWrap<T, U>(input, shape, dim, exclusive, reverse);
        ok = CompareIntVec(actual, expected);
    }

    DestroyTensor(&self);
    DestroyTensor(&out);
    (void)name;
    return ok;
}

static std::vector<int32_t> MakeInt32Data(int64_t n)
{
    std::vector<int32_t> data(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = static_cast<int32_t>((i % 9) - 4);
    }
    return data;
}

static std::vector<int8_t> MakeInt8Data(int64_t n)
{
    std::vector<int8_t> data(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = static_cast<int8_t>((i % 5) - 2);
    }
    return data;
}

static std::vector<uint8_t> MakeUint8Data(int64_t n)
{
    std::vector<uint8_t> data(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = static_cast<uint8_t>(i % 7);
    }
    return data;
}

static std::vector<int64_t> MakeInt64Data(int64_t n)
{
    std::vector<int64_t> data(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = static_cast<int64_t>((i % 13) - 6);
    }
    return data;
}

static void RunNegativeCases(aclrtStream stream)
{
    std::vector<int64_t> shape = {2, 2};

    TensorResource selfF;
    TensorResource outF;

    std::vector<float> data = {1, 2, 3, 4};
    std::vector<float> zeros = {0, 0, 0, 0};

    bool ok = (CreateTensor(data, shape, ACL_FLOAT, &selfF) == 0 &&
               CreateTensor(zeros, shape, ACL_FLOAT, &outF) == 0);

    if (!ok) {
        PrintCaseResult("NEG-setup", false);
        DestroyTensor(&selfF);
        DestroyTensor(&outF);
        return;
    }

    PrintCaseResult("NEG-null-self",
                    LaunchCumsum(stream, nullptr, &outF, 0, ACL_FLOAT, API_CUMSUM, false, false, false));

    PrintCaseResult("NEG-null-out",
                    LaunchCumsum(stream, &selfF, nullptr, 0, ACL_FLOAT, API_CUMSUM, false, false, false));

    PrintCaseResult("NEG-dim-too-large",
                    LaunchCumsum(stream, &selfF, &outF, 2, ACL_FLOAT, API_CUMSUM, false, false, false));

    PrintCaseResult("NEG-dim-too-small",
                    LaunchCumsum(stream, &selfF, &outF, -3, ACL_FLOAT, API_CUMSUM, false, false, false));

    PrintCaseResult("NEG-invalid-dtype-param",
                    LaunchCumsum(stream,
                                 &selfF,
                                 &outF,
                                 0,
                                 static_cast<aclDataType>(-1),
                                 API_CUMSUM,
                                 false,
                                 false,
                                 false));

    TensorResource outI32;
    std::vector<int32_t> zerosI32 = {0, 0, 0, 0};
    if (CreateTensor(zerosI32, shape, ACL_INT32, &outI32) == 0) {
        PrintCaseResult("NEG-dtype-mismatch-v2",
                        LaunchCumsum(stream, &selfF, &outI32, 0, ACL_FLOAT, API_CUMSUM_V2, false, false, false));
    }
    DestroyTensor(&outI32);

    TensorResource selfBool;
    TensorResource outBool;
    std::vector<uint8_t> boolData = {1, 0, 1, 0};
    std::vector<uint8_t> boolOut = {0, 0, 0, 0};
    if (CreateTensor(boolData, shape, ACL_BOOL, &selfBool) == 0 &&
        CreateTensor(boolOut, shape, ACL_BOOL, &outBool) == 0) {
        PrintCaseResult("NEG-BOOL-unsupported",
                        LaunchCumsum(stream, &selfBool, &outBool, 0, ACL_BOOL, API_CUMSUM, false, false, false));
    }
    DestroyTensor(&selfBool);
    DestroyTensor(&outBool);

    TensorResource self9;
    TensorResource out9;
    std::vector<int64_t> shape9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    if (CreateTensor(std::vector<float>(1, 1.0f), shape9, ACL_FLOAT, &self9) == 0 &&
        CreateTensor(std::vector<float>(1, 0.0f), shape9, ACL_FLOAT, &out9) == 0) {
        PrintCaseResult("NEG-rank-greater-than-8",
                        LaunchCumsum(stream, &self9, &out9, 0, ACL_FLOAT, API_CUMSUM, false, false, false));
    }

    DestroyTensor(&self9);
    DestroyTensor(&out9);
    DestroyTensor(&selfF);
    DestroyTensor(&outF);
}

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    int32_t deviceId = 0;
    const char* dev = std::getenv("ASCEND_DEVICE_ID");
    if (dev != nullptr) {
        deviceId = std::atoi(dev);
    }

    aclrtStream stream = nullptr;
    int ret = InitAcl(deviceId, &stream);
    if (ret != 0) {
        return ret;
    }

    LOG_PRINT("========== Cumsum branch-coverage tests start ==========\n");

    PrintCaseResult("TC001-FP32-basic-dim0",
                    RunFp32(stream,
                            "TC001",
                            {2, 3},
                            std::vector<float>{1, 2, 3, 4, 5, 6},
                            0,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC002-FP32-basic-dim1",
                    RunFp32(stream,
                            "TC002",
                            {2, 3},
                            std::vector<float>{1, -2, 3, 4, -5, 6},
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC003-FP32-negative-dim-minus1",
                    RunFp32(stream,
                            "TC003",
                            {3, 4},
                            MakeFloatData(12, 0.5f),
                            -1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC004-V2-ff",
                    RunFp32(stream,
                            "TC004",
                            {2, 5},
                            MakeFloatData(10, 0.25f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC005-V2-tf",
                    RunFp32(stream,
                            "TC005",
                            {2, 5},
                            MakeFloatData(10, 0.25f),
                            1,
                            API_CUMSUM_V2,
                            true,
                            false));

    PrintCaseResult("TC006-V2-ft",
                    RunFp32(stream,
                            "TC006",
                            {2, 5},
                            MakeFloatData(10, 0.25f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            true));

    PrintCaseResult("TC007-V2-tt",
                    RunFp32(stream,
                            "TC007",
                            {2, 5},
                            MakeFloatData(10, 0.25f),
                            1,
                            API_CUMSUM_V2,
                            true,
                            true));

    PrintCaseResult("TC008-V2-dim0",
                    RunFp32(stream,
                            "TC008",
                            {4, 3},
                            MakeFloatData(12, 0.25f),
                            0,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC009-FP16-basic",
                    RunFp16(stream,
                            "TC009",
                            {4, 8},
                            MakeFloatData(32, 0.125f),
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC010-BF16-basic",
                    RunBf16(stream,
                            "TC010",
                            {4, 8},
                            MakeFloatData(32, 0.125f),
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC011-DOUBLE-aicpu-standard",
                    RunFp64(stream,
                            "TC011",
                            {3, 3},
                            MakeDoubleData(9, 0.125),
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC012-DOUBLE-aicpu-v2",
                    RunFp64(stream,
                            "TC012",
                            {3, 3},
                            MakeDoubleData(9, 0.125),
                            1,
                            API_CUMSUM_V2,
                            true,
                            true));

    PrintCaseResult("TC013-INT32-last-axis",
                    RunInt32(stream,
                             "TC013",
                             {4, 8},
                             MakeInt32Data(32),
                             1,
                             API_CUMSUM,
                             false,
                             false));

    PrintCaseResult("TC014-INT32-axis0",
                    RunInt32(stream,
                             "TC014",
                             {8, 4},
                             MakeInt32Data(32),
                             0,
                             API_CUMSUM,
                             false,
                             false));

    PrintCaseResult("TC015-INT32-middle-axis-v2",
                    RunInt32(stream,
                             "TC015",
                             {2, 5, 3},
                             MakeInt32Data(30),
                             1,
                             API_CUMSUM_V2,
                             true,
                             true));

    PrintCaseResult("TC016-INT64-basic",
                    RunIntLike<int64_t, uint64_t>(stream,
                                                  "TC016",
                                                  {2, 8},
                                                  MakeInt64Data(16),
                                                  ACL_INT64,
                                                  1,
                                                  API_CUMSUM,
                                                  false,
                                                  false));

    PrintCaseResult("TC017-INT8-basic",
                    RunIntLike<int8_t, uint8_t>(stream,
                                                "TC017",
                                                {1, 16, 32},
                                                MakeInt8Data(512),
                                                ACL_INT8,
                                                1,
                                                API_CUMSUM,
                                                false,
                                                false));

    PrintCaseResult("TC018-UINT8-basic",
                    RunIntLike<uint8_t, uint8_t>(stream,
                                                 "TC018",
                                                 {2, 8, 16},
                                                 MakeUint8Data(256),
                                                 ACL_UINT8,
                                                 1,
                                                 API_CUMSUM,
                                                 false,
                                                 false));

    PrintCaseResult("TC019-empty-standard",
                    RunFp32(stream,
                            "TC019",
                            {0, 4},
                            std::vector<float>(),
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC020-empty-v2",
                    RunFp32(stream,
                            "TC020",
                            {0, 4},
                            std::vector<float>(),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC021-cube-fast-path",
                    RunFp32(stream,
                            "TC021",
                            {12800, 512},
                            MakeFloatData(12800LL * 512LL, 0.001f),
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC022-float-NGreaterCl-RFullLoad-MEnough",
                    RunFp32(stream,
                            "TC022",
                            {64, 2, 64},
                            MakeFloatData(64LL * 2LL * 64LL, 0.01f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC023-float-NGreaterCl-RFullLoad-MNotEnough-borrowN",
                    RunFp32(stream,
                            "TC023",
                            {1, 2, 1024},
                            MakeFloatData(1LL * 2LL * 1024LL, 0.01f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC024-float-NGreaterCl-RNotFullLoad-MEnough",
                    RunFp32(stream,
                            "TC024",
                            {64, 4096, 64},
                            MakeFloatData(64LL * 4096LL * 64LL, 0.0001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC025-float-NGreaterCl-RNotFullLoad-borrowR",
                    RunFp32(stream,
                            "TC025",
                            {1, 4096, 64},
                            MakeFloatData(1LL * 4096LL * 64LL, 0.0001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC026-float-RNGreaterCl-oneway-smallN",
                    RunFp32(stream,
                            "TC026",
                            {64, 128, 1},
                            MakeFloatData(64LL * 128LL, 0.01f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC027-float-RNGreaterCl-borrowR-twoway",
                    RunFp32(stream,
                            "TC027",
                            {1, 4096, 1},
                            MakeFloatData(4096, 0.001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC028-float-MRNGreaterCl",
                    RunFp32(stream,
                            "TC028",
                            {128, 2, 2},
                            MakeFloatData(128LL * 2LL * 2LL, 0.01f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC029-float-MRNLesserCl",
                    RunFp32(stream,
                            "TC029",
                            {2, 2, 2},
                            MakeFloatData(8, 0.01f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC030-float-RNGreaterCl-RNotFullLoadNotBorrowR",
                    RunFp32(stream,
                            "TC030",
                            {64, 32768, 1},
                            MakeFloatData(64LL * 32768LL, 0.00001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC031-float-RNGreaterCl-RNotFullLoadBorrowRTwoway",
                    RunFp32(stream,
                            "TC031",
                            {1, 32768, 1},
                            MakeFloatData(32768LL, 0.00001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC032-float-JudgeSklansky-alignN-large",
                    RunFp32(stream,
                            "TC032",
                            {1, 2048, 1024},
                            MakeFloatData(1LL * 2048LL * 1024LL, 0.00001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC033-float-UB-SS-TWOWAY-candidate",
                    RunFp32(stream,
                            "TC033",
                            {2, 8192, 32},
                            MakeFloatData(2LL * 8192LL * 32LL, 0.00001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            false));

    PrintCaseResult("TC034-float-UB-tail-nonzero",
                    RunFp32(stream,
                            "TC034",
                            {3, 7777, 33},
                            MakeFloatData(3LL * 7777LL * 33LL, 0.00001f),
                            1,
                            API_CUMSUM_V2,
                            true,
                            false));

    PrintCaseResult("TC035-float-fold-equal-candidate",
                    RunFp32(stream,
                            "TC035",
                            {1, 16384, 16},
                            MakeFloatData(1LL * 16384LL * 16LL, 0.00001f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            true));

    PrintCaseResult("TC036-FP16-v2-reverse",
                    RunFp16(stream,
                            "TC036",
                            {2, 16},
                            MakeFloatData(32, 0.03125f),
                            1,
                            API_CUMSUM_V2,
                            false,
                            true));

    PrintCaseResult("TC037-BF16-v2-exclusive-reverse",
                    RunBf16(stream,
                            "TC037",
                            {2, 16},
                            MakeFloatData(32, 0.03125f),
                            1,
                            API_CUMSUM_V2,
                            true,
                            true));

    PrintCaseResult("TC038-INT32-negative-dim-minus2",
                    RunInt32(stream,
                             "TC038",
                             {2, 3, 4},
                             MakeInt32Data(24),
                             -2,
                             API_CUMSUM_V2,
                             true,
                             false));

    PrintCaseResult("TC039-int-AdjustTensor4TDRA-rightAxis-large",
                    RunInt32(stream,
                             "TC039",
                             {1, 4, 2048},
                             MakeInt32Data(1LL * 4LL * 2048LL),
                             1,
                             API_CUMSUM_V2,
                             false,
                             false));

    PrintCaseResult("TC040-int-AdjustTensor4TDLA-leftAxis-large",
                    RunInt32(stream,
                             "TC040",
                             {256, 4, 1},
                             MakeInt32Data(256LL * 4LL),
                             1,
                             API_CUMSUM_V2,
                             false,
                             false));

    PrintCaseResult("TC041-int-AdjustTensor4TDR-midAxis-large",
                    RunInt32(stream,
                             "TC041",
                             {1, 4096, 1},
                             MakeInt32Data(4096LL),
                             1,
                             API_CUMSUM_V2,
                             false,
                             false));

    PrintCaseResult("TC042-int-CheckBGC-AdjustLARLpUnit-candidate",
                    RunInt32(stream,
                             "TC042",
                             {128, 256, 4},
                             MakeInt32Data(128LL * 256LL * 4LL),
                             1,
                             API_CUMSUM_V2,
                             true,
                             false));

    PrintCaseResult("TC043-int-RA-axis-weight",
                    RunInt32(stream,
                             "TC043",
                             {1, 4, 4096},
                             MakeInt32Data(1LL * 4LL * 4096LL),
                             1,
                             API_CUMSUM_V2,
                             false,
                             true));

    PrintCaseResult("TC044-int-R-axis-block-CUM_WITH_GROUP",
                    RunInt32(stream,
                             "TC044",
                             {1, 8192, 1},
                             MakeInt32Data(8192LL),
                             1,
                             API_CUMSUM_V2,
                             false,
                             false));

    PrintCaseResult("TC045-int-CalcAxisWeight-divisible",
                    RunInt32(stream,
                             "TC045",
                             {64, 1024, 1},
                             MakeInt32Data(64LL * 1024LL),
                             1,
                             API_CUMSUM_V2,
                             true,
                             true));

    PrintCaseResult("TC046-int8-dtypeSize1-rightAxis-large",
                    RunIntLike<int8_t, uint8_t>(stream,
                                                "TC046",
                                                {1, 4, 2048},
                                                MakeInt8Data(1LL * 4LL * 2048LL),
                                                ACL_INT8,
                                                1,
                                                API_CUMSUM_V2,
                                                false,
                                                false));

    PrintCaseResult("TC047-uint8-dtypeSize1-rightAxis-large",
                    RunIntLike<uint8_t, uint8_t>(stream,
                                                 "TC047",
                                                 {1, 4, 2048},
                                                 MakeUint8Data(1LL * 4LL * 2048LL),
                                                 ACL_UINT8,
                                                 1,
                                                 API_CUMSUM_V2,
                                                 false,
                                                 false));

    PrintCaseResult("TC048-int8-negative-dim-dtypeSize1",
                    RunIntLike<int8_t, uint8_t>(stream,
                                                "TC048",
                                                {1, 4, 1024},
                                                MakeInt8Data(4096),
                                                ACL_INT8,
                                                -2,
                                                API_CUMSUM_V2,
                                                true,
                                                true));

    PrintCaseResult("TC049-uint8-R-axis-group",
                    RunIntLike<uint8_t, uint8_t>(stream,
                                                 "TC049",
                                                 {1, 8192, 1},
                                                 MakeUint8Data(8192),
                                                 ACL_UINT8,
                                                 1,
                                                 API_CUMSUM_V2,
                                                 false,
                                                 false));

    PrintCaseResult("TC050-int64-AiCpu-or-RegBase-candidate",
                    RunIntLike<int64_t, uint64_t>(stream,
                                                  "TC050",
                                                  {1, 512, 16},
                                                  MakeInt64Data(1LL * 512LL * 16LL),
                                                  ACL_INT64,
                                                  1,
                                                  API_CUMSUM_V2,
                                                  false,
                                                  true));

    PrintCaseResult("TC051-INT8-extreme-values",
                    RunIntLike<int8_t, uint8_t>(stream,
                                                "TC051",
                                                {1, 8},
                                                std::vector<int8_t>{
                                                    std::numeric_limits<int8_t>::min(),
                                                    -1,
                                                    1,
                                                    std::numeric_limits<int8_t>::max(),
                                                    1,
                                                    -2,
                                                    3,
                                                    -4},
                                                ACL_INT8,
                                                1,
                                                API_CUMSUM_V2,
                                                true,
                                                false));

    PrintCaseResult("TC052-UINT8-extreme-values",
                    RunIntLike<uint8_t, uint8_t>(stream,
                                                 "TC052",
                                                 {1, 8},
                                                 std::vector<uint8_t>{0, 1, 254, 255, 1, 2, 3, 4},
                                                 ACL_UINT8,
                                                 1,
                                                 API_CUMSUM_V2,
                                                 false,
                                                 true));

    PrintCaseResult("TC053-FP32-single-element",
                    RunFp32(stream,
                            "TC053",
                            {1},
                            std::vector<float>{42.0f},
                            0,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("TC054-FP32-all-zero",
                    RunFp32(stream,
                            "TC054",
                            {4, 16},
                            std::vector<float>(64, 0.0f),
                            -1,
                            API_CUMSUM_V2,
                            true,
                            true));

    PrintCaseResult("P01-FP32-long-0p1-error-accumulation",
                    RunFp32(stream,
                            "P01",
                            {1, 4096},
                            std::vector<float>(4096, 0.1f),
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("P02-FP32-large-small-mixing",
                    RunFp32(stream,
                            "P02",
                            {1, 8},
                            std::vector<float>{1e8f, 1.0f, 1.0f, 1.0f, -1e8f, 1.0f, 1.0f, 1.0f},
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("P03-FP16-long-small-values",
                    RunFp16(stream,
                            "P03",
                            {1, 2048},
                            std::vector<float>(2048, 0.001f),
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("P04-FP32-cancellation",
                    RunFp32(stream,
                            "P04",
                            {1, 6},
                            std::vector<float>{100.0f, 0.1f, -100.0f, 0.1f, -0.1f, 0.1f},
                            1,
                            API_CUMSUM,
                            false,
                            false));

    PrintCaseResult("P05-INT32-overflow",
                    RunInt32(stream,
                             "P05",
                             {1, 4},
                             std::vector<int32_t>{std::numeric_limits<int32_t>::max(), 1, 1, -2},
                             1,
                             API_CUMSUM,
                             false,
                             false));

    RunNegativeCases(stream);

    LOG_PRINT("========== Summary ==========\n");
    LOG_PRINT("TOTAL=%d PASS=%d FAIL=%d\n", g_total, g_pass, g_fail);

    FinalizeAcl(deviceId, stream);
    return g_fail == 0 ? 0 : 1;
}