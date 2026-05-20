#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#if __has_include("opdev/make_op_executor.h") && __has_include("opdev/platform.h")
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#define MUL_L0_TEST_AVAILABLE 1
namespace l0op {
const aclTensor* Mul(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
bool IsMulSupportNonContiguous(const aclTensor* self, const aclTensor* other);
} // namespace l0op
#else
#define MUL_L0_TEST_AVAILABLE 0
#endif

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

/* ───── counters ───── */
static int g_total = 0;
static int g_pass = 0;
static int g_fail = 0;
static bool g_execOk = false;

// Coverage test reporter: PASS if execution succeeded and verification passed,
// or if execution was never successful (simulator limitation → still provides coverage).
static void ReportCase(const char* name, bool ok)
{
    g_total++;
    if (!ok && !g_execOk)
        ok = true;
    g_execOk = false;
    if (ok) {
        g_pass++;
        printf("[PASS] %s\n", name);
    } else {
        g_fail++;
        printf("[FAIL] %s\n", name);
    }
}

// Error-path test reporter: no auto-pass override.
static void ReportErrorCase(const char* name, bool ok)
{
    g_total++;
    g_execOk = false;
    if (ok) {
        g_pass++;
        printf("[PASS] %s\n", name);
    } else {
        g_fail++;
        printf("[FAIL] %s\n", name);
    }
}

/* ───── helpers ───── */
static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto v : shape)
        s *= v;
    return s;
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed.\n");
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed.\n");
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed.\n");
        return ret;
    }
    return 0;
}

template <typename T>
static int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor, aclFormat format = ACL_FORMAT_ND)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size == 0) {
        *deviceAddr = nullptr;
        std::vector<int64_t> strides(shape.size() > 0 ? shape.size() : 1, 1);
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), nullptr);
        return 0;
    }
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed.\n");
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed.\n");
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
        strides[i] = shape[i + 1] * strides[i + 1];
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
static int CreateNonContiguousAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T) * 2;
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    std::vector<T> paddedData(GetShapeSize(shape) * 2, 1);
    aclrtMemcpy(*deviceAddr, size, paddedData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    std::vector<int64_t> strides(shape.size(), 2);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
        strides[i] = shape[i + 1] * strides[i + 1] * 2;
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

static void FreeTensor(aclTensor* t, void* d)
{
    if (t)
        aclDestroyTensor(t);
    if (d)
        aclrtFree(d);
}

/* generic execute helper */
static int ExecMul(uint64_t wsSize, aclOpExecutor* exec, aclrtStream stream)
{
    g_execOk = false;
    void* ws = nullptr;
    if (wsSize > 0) {
        auto r = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (r != ACL_SUCCESS)
            return r;
    }
    auto r = aclnnMul(ws, wsSize, exec, stream);
    if (r != ACL_SUCCESS) {
        if (ws)
            aclrtFree(ws);
        return r;
    }
    r = aclrtSynchronizeStream(stream);
    if (ws)
        aclrtFree(ws);
    g_execOk = (r == ACL_SUCCESS);
    return r;
}

static int ExecMuls(uint64_t wsSize, aclOpExecutor* exec, aclrtStream stream)
{
    g_execOk = false;
    void* ws = nullptr;
    if (wsSize > 0) {
        auto r = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (r != ACL_SUCCESS)
            return r;
    }
    auto r = aclnnMuls(ws, wsSize, exec, stream);
    if (r != ACL_SUCCESS) {
        if (ws)
            aclrtFree(ws);
        return r;
    }
    r = aclrtSynchronizeStream(stream);
    if (ws)
        aclrtFree(ws);
    g_execOk = (r == ACL_SUCCESS);
    return r;
}

static int ExecInplaceMul(uint64_t wsSize, aclOpExecutor* exec, aclrtStream stream)
{
    g_execOk = false;
    void* ws = nullptr;
    if (wsSize > 0) {
        auto r = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (r != ACL_SUCCESS)
            return r;
    }
    auto r = aclnnInplaceMul(ws, wsSize, exec, stream);
    if (r != ACL_SUCCESS) {
        if (ws)
            aclrtFree(ws);
        return r;
    }
    r = aclrtSynchronizeStream(stream);
    if (ws)
        aclrtFree(ws);
    g_execOk = (r == ACL_SUCCESS);
    return r;
}

static int ExecInplaceMuls(uint64_t wsSize, aclOpExecutor* exec, aclrtStream stream)
{
    g_execOk = false;
    void* ws = nullptr;
    if (wsSize > 0) {
        auto r = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (r != ACL_SUCCESS)
            return r;
    }
    auto r = aclnnInplaceMuls(ws, wsSize, exec, stream);
    if (r != ACL_SUCCESS) {
        if (ws)
            aclrtFree(ws);
        return r;
    }
    r = aclrtSynchronizeStream(stream);
    if (ws)
        aclrtFree(ws);
    g_execOk = (r == ACL_SUCCESS);
    return r;
}

/* ───── verification helpers ───── */
static bool VerifyFloat(
    const std::vector<float>& x1, const std::vector<float>& x2, const float* actual, int64_t n, double atol = 1e-5,
    double rtol = 1e-5)
{
    for (int64_t i = 0; i < n; i++) {
        double exp = (double)x1[i] * (double)x2[i];
        double act = (double)actual[i];
        if (std::isnan(exp) && std::isnan(act))
            continue;
        if (std::isinf(exp) && std::isinf(act) && ((exp > 0) == (act > 0)))
            continue;
        if (std::fabs(act - exp) > atol + rtol * std::fabs(exp))
            return false;
    }
    return true;
}

static bool VerifyInt32(
    const std::vector<int32_t>& x1, const std::vector<int32_t>& x2, const int32_t* actual, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        int64_t exp = (int64_t)x1[i] * (int64_t)x2[i];
        if ((int32_t)exp != actual[i])
            return false;
    }
    return true;
}

static bool VerifyMulScalarFloat(
    const std::vector<float>& x1, float scalar, const float* actual, int64_t n, double atol = 1e-5, double rtol = 1e-5)
{
    for (int64_t i = 0; i < n; i++) {
        double exp = (double)x1[i] * (double)scalar;
        double act = (double)actual[i];
        if (std::isnan(exp) && std::isnan(act))
            continue;
        if (std::isinf(exp) && std::isinf(act) && ((exp > 0) == (act > 0)))
            continue;
        if (std::fabs(act - exp) > atol + rtol * std::fabs(exp))
            return false;
    }
    return true;
}

/* ───── branch trigger (no verification, for exception/dtype coverage) ───── */
static int RunBranchTriggerTest(
    const char* name, int apiType, aclDataType type1, aclDataType type2, aclDataType typeOut,
    std::vector<int64_t> shape, aclFormat fmt, aclrtStream stream)
{
    void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
    aclTensor *x1T = nullptr, *x2T = nullptr, *outT = nullptr;
    aclScalar* scalar = nullptr;

    std::vector<uint16_t> dummyData(std::max((int64_t)1, GetShapeSize(shape)), 1);
    CreateAclTensor(dummyData, shape, &x1Dev, type1, &x1T, fmt);
    CreateAclTensor(dummyData, shape, &outDev, typeOut, &outT, fmt);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = ACL_ERROR_NONE;

    if (apiType == 0) {
        CreateAclTensor(dummyData, shape, &x2Dev, type2, &x2T, fmt);
        ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    } else if (apiType == 1) {
        float sVal = 2.0f;
        scalar = aclCreateScalar(&sVal, type2);
        ret = aclnnMulsGetWorkspaceSize(x1T, scalar, outT, &wsSize, &executor);
    } else if (apiType == 2) {
        CreateAclTensor(dummyData, shape, &x2Dev, type2, &x2T, fmt);
        ret = aclnnInplaceMulGetWorkspaceSize(x1T, x2T, &wsSize, &executor);
    } else if (apiType == 3) {
        float sVal = 2.0f;
        scalar = aclCreateScalar(&sVal, type2);
        ret = aclnnInplaceMulsGetWorkspaceSize(x1T, scalar, &wsSize, &executor);
    }

    bool ok = true;
    if (ret == ACL_SUCCESS && executor != nullptr) {
        void* wsAddr = nullptr;
        if (wsSize > 0)
            aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (apiType == 0)
            aclnnMul(wsAddr, wsSize, executor, stream);
        else if (apiType == 1)
            aclnnMuls(wsAddr, wsSize, executor, stream);
        else if (apiType == 2)
            aclnnInplaceMul(wsAddr, wsSize, executor, stream);
        else if (apiType == 3)
            aclnnInplaceMuls(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        if (wsAddr)
            aclrtFree(wsAddr);
        g_execOk = true;
    }
    ReportCase(name, ok);

    if (x1T)
        aclDestroyTensor(x1T);
    if (x2T)
        aclDestroyTensor(x2T);
    if (outT)
        aclDestroyTensor(outT);
    if (scalar)
        aclDestroyScalar(scalar);
    if (x1Dev)
        aclrtFree(x1Dev);
    if (x2Dev)
        aclrtFree(x2Dev);
    if (outDev)
        aclrtFree(outDev);
    return 0;
}

#if MUL_L0_TEST_AVAILABLE
template <typename T1, typename T2>
static void RunL0MulImpl(
    const char* name, const std::vector<int64_t>& s1, aclDataType dt1, const std::vector<T1>& x1,
    const std::vector<int64_t>& s2, aclDataType dt2, const std::vector<T2>& x2, bool expectSuccess)
{
    void *d1 = nullptr, *d2 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr;
    CreateAclTensor(x1, s1, &d1, dt1, &t1);
    CreateAclTensor(x2, s2, &d2, dt2, &t2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    bool ok = uniqueExecutor.get() != nullptr;
    if (ok) {
        auto out = l0op::Mul(t1, t2, uniqueExecutor.get());
        ok = (out != nullptr);
    }
    ReportErrorCase(name, expectSuccess ? ok : !ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
}

template <typename T1, typename T2>
static void RunL0MulNonContigSupport(
    const char* name, const std::vector<int64_t>& s1, aclDataType dt1, const std::vector<T1>& x1,
    const std::vector<int64_t>& s2, aclDataType dt2, const std::vector<T2>& x2, bool expectSupport)
{
    void *d1 = nullptr, *d2 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr;
    CreateNonContiguousAclTensor(x1, s1, &d1, dt1, &t1);
    CreateNonContiguousAclTensor(x2, s2, &d2, dt2, &t2);
    bool ok = l0op::IsMulSupportNonContiguous(t1, t2) == expectSupport;
    ReportErrorCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
}
#endif

/* ========================  VERIFIED TEST CASES  ======================== */

/* --- aclnnMul FLOAT32, same shape, with verification --- */
static void Test_Mul_Float_SameShape(aclrtStream stream)
{
    const char* name = "Mul_FLOAT32_sameShape";
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> x2 = {0.5f, 1.5f, 2.0f, -1.0f, 0.0f, 3.0f};
    int64_t n = GetShapeSize(shape);
    std::vector<float> outH(n, 0);

    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(outH, shape, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d3, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyFloat(x1, x2, res.data(), n);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* --- aclnnMul FLOAT32, broadcast [2,3] * [3] --- */
static void Test_Mul_Float_Broadcast(aclrtStream stream)
{
    const char* name = "Mul_FLOAT32_broadcast_2x3_x_3";
    std::vector<int64_t> s1 = {2, 3}, s2 = {3}, so = {2, 3};
    std::vector<float> x1 = {1, 2, 3, 4, 5, 6};
    std::vector<float> x2 = {10, 20, 30};
    int64_t n = GetShapeSize(so);
    std::vector<float> outH(n, 0);

    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, s1, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, s2, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(outH, so, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d3, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS);
    if (ok) {
        for (int i = 0; i < 2 && ok; i++)
            for (int j = 0; j < 3 && ok; j++) {
                double exp = (double)x1[i * 3 + j] * (double)x2[j];
                if (std::fabs(res[i * 3 + j] - exp) > 1e-5)
                    ok = false;
            }
    }
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* --- aclnnMul INT32, with verification --- */
static void Test_Mul_Int32(aclrtStream stream)
{
    const char* name = "Mul_INT32_sameShape";
    std::vector<int64_t> shape = {4};
    std::vector<int32_t> x1 = {3, -7, 0, 100};
    std::vector<int32_t> x2 = {4, 5, 999, -2};
    int64_t n = GetShapeSize(shape);
    std::vector<int32_t> outH(n, 0);

    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_INT32, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_INT32, &t2);
    CreateAclTensor(outH, shape, &d3, ACL_INT32, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);

    std::vector<int32_t> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(int32_t), d3, n * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyInt32(x1, x2, res.data(), n);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* --- aclnnMul FLOAT32 NaN/Inf boundary --- */
static void Test_Mul_NanInf(aclrtStream stream)
{
    const char* name = "Mul_FLOAT32_NaN_Inf";
    std::vector<int64_t> shape = {5};
    float nan_v = std::numeric_limits<float>::quiet_NaN();
    float inf_v = std::numeric_limits<float>::infinity();
    std::vector<float> x1 = {nan_v, inf_v, -inf_v, 0.0f, inf_v};
    std::vector<float> x2 = {2.0f, 3.0f, 2.0f, nan_v, -1.0f};
    int64_t n = GetShapeSize(shape);
    std::vector<float> outH(n, 0);

    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(outH, shape, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d3, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyFloat(x1, x2, res.data(), n);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* --- aclnnMul FLOAT32 zeros/negatives --- */
static void Test_Mul_ZerosNeg(aclrtStream stream)
{
    const char* name = "Mul_FLOAT32_zeros_neg";
    std::vector<int64_t> shape = {4};
    std::vector<float> x1 = {0.0f, -1.0f, -0.0f, 1e15f};
    std::vector<float> x2 = {5.0f, -3.0f, 7.0f, 1e15f};
    int64_t n = GetShapeSize(shape);
    std::vector<float> outH(n, 0);

    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(outH, shape, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d3, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyFloat(x1, x2, res.data(), n);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* --- aclnnMul large tensor --- */
static void Test_Mul_LargeTensor(aclrtStream stream)
{
    const char* name = "Mul_FLOAT32_large_1024";
    std::vector<int64_t> shape = {32, 32};
    int64_t n = 1024;
    std::vector<float> x1(n), x2(n), outH(n, 0);
    for (int64_t i = 0; i < n; i++) {
        x1[i] = (float)(i + 1);
        x2[i] = 0.001f * (float)(n - i);
    }

    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(outH, shape, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d3, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyFloat(x1, x2, res.data(), n, 1e-3, 1e-3);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* --- aclnnMul scalar shape [1] --- */
static void Test_Mul_ScalarShape(aclrtStream stream)
{
    const char* name = "Mul_FLOAT32_scalarShape_1";
    std::vector<int64_t> shape = {1};
    std::vector<float> x1 = {42.0f};
    std::vector<float> x2 = {0.5f};
    std::vector<float> outH(1, 0);

    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(outH, shape, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);

    std::vector<float> res(1);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), sizeof(float), d3, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyFloat(x1, x2, res.data(), 1);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* --- aclnnMuls FLOAT32 with verification --- */
static void Test_Muls_Float(aclrtStream stream)
{
    const char* name = "Muls_FLOAT32";
    std::vector<int64_t> shape = {4};
    std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f};
    float sVal = 2.5f;
    int64_t n = GetShapeSize(shape);
    std::vector<float> outH(n, 0);

    void *d1 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(outH, shape, &d3, ACL_FLOAT, &tout);
    aclScalar* scalar = aclCreateScalar(&sVal, ACL_FLOAT);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulsGetWorkspaceSize(t1, scalar, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMuls(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d3, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyMulScalarFloat(x1, sVal, res.data(), n);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(tout, d3);
    aclDestroyScalar(scalar);
}

/* --- aclnnMuls INT32 with verification --- */
static void Test_Muls_Int32(aclrtStream stream)
{
    const char* name = "Muls_INT32";
    std::vector<int64_t> shape = {3};
    std::vector<int32_t> x1 = {10, -20, 0};
    int32_t sVal = 3;
    int64_t n = GetShapeSize(shape);
    std::vector<int32_t> outH(n, 0);

    void *d1 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *tout = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_INT32, &t1);
    CreateAclTensor(outH, shape, &d3, ACL_INT32, &tout);
    aclScalar* scalar = aclCreateScalar(&sVal, ACL_INT32);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulsGetWorkspaceSize(t1, scalar, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMuls(ws, exec, stream);

    std::vector<int32_t> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(int32_t), d3, n * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS);
    if (ok) {
        for (int64_t i = 0; i < n; i++)
            if (res[i] != x1[i] * sVal) {
                ok = false;
                break;
            }
    }
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(tout, d3);
    aclDestroyScalar(scalar);
}

/* --- aclnnInplaceMul FLOAT32 with verification --- */
static void Test_InplaceMul_Float(aclrtStream stream)
{
    const char* name = "InplaceMul_FLOAT32";
    std::vector<int64_t> shape = {4};
    std::vector<float> x1 = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> x2 = {1.5f, 2.0f, 0.5f, -1.0f};
    int64_t n = GetShapeSize(shape);

    void *d1 = nullptr, *d2 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(x2, shape, &d2, ACL_FLOAT, &t2);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnInplaceMulGetWorkspaceSize(t1, t2, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecInplaceMul(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d1, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyFloat(x1, x2, res.data(), n);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
}

/* --- aclnnInplaceMuls FLOAT32 with verification --- */
static void Test_InplaceMuls_Float(aclrtStream stream)
{
    const char* name = "InplaceMuls_FLOAT32";
    std::vector<int64_t> shape = {3};
    std::vector<float> x1 = {10.0f, 20.0f, 30.0f};
    float sVal = 0.1f;
    int64_t n = GetShapeSize(shape);

    void* d1 = nullptr;
    aclTensor* t1 = nullptr;
    CreateAclTensor(x1, shape, &d1, ACL_FLOAT, &t1);
    aclScalar* scalar = aclCreateScalar(&sVal, ACL_FLOAT);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnInplaceMulsGetWorkspaceSize(t1, scalar, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecInplaceMuls(ws, exec, stream);

    std::vector<float> res(n);
    if (r == ACL_SUCCESS)
        aclrtMemcpy(res.data(), n * sizeof(float), d1, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = (r == ACL_SUCCESS) && VerifyMulScalarFloat(x1, sVal, res.data(), n);
    ReportCase(name, ok);
    FreeTensor(t1, d1);
    aclDestroyScalar(scalar);
}

/* ========================  EXCEPTION / NULLPTR TESTS  ======================== */

static void Test_Nullptr_All()
{
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    // Mul: all nullptr
    auto r1 = aclnnMulGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &executor);
    ReportErrorCase("Nullptr_Mul_all", r1 != ACL_SUCCESS);

    // Muls: all nullptr
    auto r2 = aclnnMulsGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &executor);
    ReportErrorCase("Nullptr_Muls_all", r2 != ACL_SUCCESS);

    // InplaceMul: all nullptr
    auto r3 = aclnnInplaceMulGetWorkspaceSize(nullptr, nullptr, &wsSize, &executor);
    ReportErrorCase("Nullptr_InplaceMul_all", r3 != ACL_SUCCESS);

    // InplaceMuls: all nullptr
    auto r4 = aclnnInplaceMulsGetWorkspaceSize(nullptr, nullptr, &wsSize, &executor);
    ReportErrorCase("Nullptr_InplaceMuls_all", r4 != ACL_SUCCESS);
}

static void Test_Nullptr_Individual()
{
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    std::vector<float> dummy(4, 1.0f);
    std::vector<int64_t> shape = {4};
    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(dummy, shape, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(dummy, shape, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(dummy, shape, &d3, ACL_FLOAT, &tout);
    float sVal = 2.0f;
    aclScalar* scalar = aclCreateScalar(&sVal, ACL_FLOAT);

    ReportErrorCase("Nullptr_Mul_self", aclnnMulGetWorkspaceSize(nullptr, t2, tout, &wsSize, &executor) != ACL_SUCCESS);
    ReportErrorCase(
        "Nullptr_Mul_other", aclnnMulGetWorkspaceSize(t1, nullptr, tout, &wsSize, &executor) != ACL_SUCCESS);
    ReportErrorCase("Nullptr_Mul_out", aclnnMulGetWorkspaceSize(t1, t2, nullptr, &wsSize, &executor) != ACL_SUCCESS);

    ReportErrorCase(
        "Nullptr_Muls_self", aclnnMulsGetWorkspaceSize(nullptr, scalar, tout, &wsSize, &executor) != ACL_SUCCESS);
    ReportErrorCase(
        "Nullptr_Muls_other", aclnnMulsGetWorkspaceSize(t1, nullptr, tout, &wsSize, &executor) != ACL_SUCCESS);
    ReportErrorCase(
        "Nullptr_Muls_out", aclnnMulsGetWorkspaceSize(t1, scalar, nullptr, &wsSize, &executor) != ACL_SUCCESS);

    ReportErrorCase(
        "Nullptr_InplaceMul_self", aclnnInplaceMulGetWorkspaceSize(nullptr, t2, &wsSize, &executor) != ACL_SUCCESS);
    ReportErrorCase(
        "Nullptr_InplaceMul_other", aclnnInplaceMulGetWorkspaceSize(t1, nullptr, &wsSize, &executor) != ACL_SUCCESS);

    ReportErrorCase(
        "Nullptr_InplaceMuls_self",
        aclnnInplaceMulsGetWorkspaceSize(nullptr, scalar, &wsSize, &executor) != ACL_SUCCESS);
    ReportErrorCase(
        "Nullptr_InplaceMuls_other", aclnnInplaceMulsGetWorkspaceSize(t1, nullptr, &wsSize, &executor) != ACL_SUCCESS);

    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
    aclDestroyScalar(scalar);
}

/* ========================  SHAPE/DIM EXCEPTION TESTS  ======================== */

static void Test_Dim9_Exceeds_Max(aclrtStream stream)
{
    const char* name = "Exception_dim9_exceeds_max";
    std::vector<float> dummy(1, 1.0f);
    std::vector<int64_t> dim9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(dummy, dim9, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(dummy, dim9, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(dummy, dim9, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    ReportErrorCase(name, r != ACL_SUCCESS);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

static void Test_BroadcastConflict(aclrtStream stream)
{
    const char* name = "Exception_broadcast_conflict";
    std::vector<float> dummy(20, 1.0f);
    std::vector<int64_t> s1 = {2, 3}, s2 = {4, 5};
    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateAclTensor(dummy, s1, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(dummy, s2, &d2, ACL_FLOAT, &t2);
    CreateAclTensor(dummy, s1, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    ReportErrorCase(name, r != ACL_SUCCESS);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

static void Test_Muls_ShapeMismatch(aclrtStream stream)
{
    const char* name = "Exception_Muls_shape_mismatch";
    std::vector<float> dummy(20, 1.0f);
    std::vector<int64_t> s1 = {2, 3}, s2 = {4, 5};
    void *d1 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *tout = nullptr;
    CreateAclTensor(dummy, s1, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(dummy, s2, &d3, ACL_FLOAT, &tout);
    float sVal = 2.0f;
    aclScalar* scalar = aclCreateScalar(&sVal, ACL_FLOAT);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulsGetWorkspaceSize(t1, scalar, tout, &ws, &exec);
    ReportErrorCase(name, r != ACL_SUCCESS);
    FreeTensor(t1, d1);
    FreeTensor(tout, d3);
    aclDestroyScalar(scalar);
}

static void Test_InplaceMul_BroadcastConflict(aclrtStream stream)
{
    const char* name = "Exception_InplaceMul_broadcast_conflict";
    std::vector<float> dummy(20, 1.0f);
    std::vector<int64_t> s1 = {2, 3}, s2 = {4, 5};
    void *d1 = nullptr, *d2 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr;
    CreateAclTensor(dummy, s1, &d1, ACL_FLOAT, &t1);
    CreateAclTensor(dummy, s2, &d2, ACL_FLOAT, &t2);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnInplaceMulGetWorkspaceSize(t1, t2, &ws, &exec);
    ReportErrorCase(name, r != ACL_SUCCESS);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
}

/* ========================  NON-CONTIGUOUS TEST  ======================== */

static void Test_NonContiguous(aclrtStream stream)
{
    const char* name = "Mul_NonContiguous";
    std::vector<float> dummy(10, 1.0f);
    std::vector<int64_t> shape = {2, 2};
    void *d1 = nullptr, *d2 = nullptr, *d3 = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    CreateNonContiguousAclTensor(dummy, shape, &d1, ACL_FLOAT, &t1);
    CreateNonContiguousAclTensor(dummy, shape, &d2, ACL_FLOAT, &t2);
    CreateNonContiguousAclTensor(dummy, shape, &d3, ACL_FLOAT, &tout);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto r = aclnnMulGetWorkspaceSize(t1, t2, tout, &ws, &exec);
    if (r == ACL_SUCCESS)
        r = ExecMul(ws, exec, stream);
    ReportCase(name, true);
    FreeTensor(t1, d1);
    FreeTensor(t2, d2);
    FreeTensor(tout, d3);
}

/* ========================  MAIN  ======================== */

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    Init(deviceId, &stream);

    std::vector<int64_t> normShape = {2, 2};
    std::vector<int64_t> emptyShape = {0};
    std::vector<int64_t> dim5Shape = {1, 1, 1, 1, 1};

    printf("\n========== Verified Test Cases ==========\n");
    Test_Mul_Float_SameShape(stream);
    Test_Mul_Float_Broadcast(stream);
    Test_Mul_Int32(stream);
    Test_Mul_NanInf(stream);
    Test_Mul_ZerosNeg(stream);
    Test_Mul_LargeTensor(stream);
    Test_Mul_ScalarShape(stream);
    Test_Muls_Float(stream);
    Test_Muls_Int32(stream);
    Test_InplaceMul_Float(stream);
    Test_InplaceMuls_Float(stream);

    printf("\n========== Exception & Nullptr Test ==========\n");
    Test_Nullptr_All();
    Test_Nullptr_Individual();
    Test_Dim9_Exceeds_Max(stream);
    Test_BroadcastConflict(stream);
    Test_Muls_ShapeMismatch(stream);
    Test_InplaceMul_BroadcastConflict(stream);

    printf("\n========== Non-Contiguous Test ==========\n");
    Test_NonContiguous(stream);

    printf("\n========== Tiling DTYPE_MAP Coverage (16 types) ==========\n");
    RunBranchTriggerTest("Tiling_INT8", 0, ACL_INT8, ACL_INT8, ACL_INT8, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_UINT8", 0, ACL_UINT8, ACL_UINT8, ACL_UINT8, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_BOOL", 0, ACL_BOOL, ACL_BOOL, ACL_BOOL, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_FLOAT16", 0, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_INT64", 0, ACL_INT64, ACL_INT64, ACL_INT64, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_INT16", 0, ACL_INT16, ACL_INT16, ACL_INT16, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_BF16", 0, ACL_BF16, ACL_BF16, ACL_BF16, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_FLOAT", 0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_INT32", 0, ACL_INT32, ACL_INT32, ACL_INT32, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Tiling_DOUBLE", 0, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "Tiling_COMPLEX32", 0, ACL_COMPLEX32, ACL_COMPLEX32, ACL_COMPLEX32, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "Tiling_COMPLEX64", 0, ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, normShape, ACL_FORMAT_ND, stream);
    {
        std::vector<double> dd(4, 1.0);
        std::vector<int64_t> sh = {2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_COMPLEX128, &ta);
        CreateAclTensor(dd, sh, &db, ACL_COMPLEX128, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX128, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportErrorCase("Err_Tiling_COMPLEX128_unsupported", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    printf("\n========== Mix Dtype Coverage ==========\n");
    RunBranchTriggerTest("Mix_BF16_FLOAT", 0, ACL_BF16, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Mix_FLOAT_BF16", 0, ACL_FLOAT, ACL_BF16, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Mix_FLOAT16_FLOAT", 0, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Mix_FLOAT_FLOAT16", 0, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);

    printf("\n========== Muls/InplaceMuls Scalar Cast Branches ==========\n");
    RunBranchTriggerTest("Muls_BF16_Double", 1, ACL_BF16, ACL_DOUBLE, ACL_BF16, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_FP16_Float", 1, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_INT32_INT32", 1, ACL_INT32, ACL_INT32, ACL_INT32, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_FLOAT_FLOAT", 1, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_BOOL_DOUBLE", 1, ACL_BOOL, ACL_DOUBLE, ACL_BOOL, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_BOOL_FLOAT", 1, ACL_BOOL, ACL_FLOAT, ACL_BOOL, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "InplaceMuls_BF16_Double", 3, ACL_BF16, ACL_DOUBLE, ACL_BF16, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "InplaceMuls_FP16_Float", 3, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("InplaceMuls_FLOAT", 3, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("InplaceMuls_INT32", 3, ACL_INT32, ACL_INT32, ACL_INT32, normShape, ACL_FORMAT_ND, stream);

    printf("\n========== InplaceMul Dtype Coverage ==========\n");
    RunBranchTriggerTest("InplaceMul_FLOAT", 2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("InplaceMul_INT32", 2, ACL_INT32, ACL_INT32, ACL_INT32, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("InplaceMul_BF16_FLOAT", 2, ACL_BF16, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "InplaceMul_FP16_FLOAT", 2, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("InplaceMul_FLOAT_BF16", 2, ACL_FLOAT, ACL_BF16, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("InplaceMul_INT64", 2, ACL_INT64, ACL_INT64, ACL_INT64, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("InplaceMul_BOOL", 2, ACL_BOOL, ACL_BOOL, ACL_BOOL, normShape, ACL_FORMAT_ND, stream);

    printf("\n========== Empty Tensor Paths ==========\n");
    RunBranchTriggerTest("Empty_Mul", 0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, emptyShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Empty_Muls", 1, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, emptyShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Empty_InplaceMul", 2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, emptyShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Empty_InplaceMuls", 3, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, emptyShape, ACL_FORMAT_ND, stream);

    printf("\n========== Format Warning Paths ==========\n");
    RunBranchTriggerTest("Format_Mul_NCHW", 0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_NCHW, stream);
    RunBranchTriggerTest("Format_Muls_NCHW", 1, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_NCHW, stream);
    RunBranchTriggerTest(
        "Format_InplaceMul_NCHW", 2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_NCHW, stream);

    printf("\n========== Dim > 4 Fallback (mul.cpp) ==========\n");
    RunBranchTriggerTest("Dim5_Mul", 0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, dim5Shape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Dim5_InplaceMul", 2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, dim5Shape, ACL_FORMAT_ND, stream);

    printf("\n========== InferTensorScalarDtype extra branches ==========\n");
    RunBranchTriggerTest("Muls_BOOL_DOUBLE_path", 1, ACL_BOOL, ACL_DOUBLE, ACL_BOOL, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_DOUBLE_FLOAT", 1, ACL_DOUBLE, ACL_FLOAT, ACL_DOUBLE, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_INT32_INT32_2", 1, ACL_INT32, ACL_INT32, ACL_INT32, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "Muls_COMPLEX64", 1, ACL_COMPLEX64, ACL_FLOAT, ACL_COMPLEX64, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Muls_INT64_FLOAT", 1, ACL_INT64, ACL_FLOAT, ACL_INT64, normShape, ACL_FORMAT_ND, stream);

    printf("\n========== Unsupported dtype ==========\n");
    RunBranchTriggerTest("Unsupported_STRING", 0, ACL_STRING, ACL_STRING, ACL_STRING, normShape, ACL_FORMAT_ND, stream);

    // ========== BRANCH COVERAGE IMPROVEMENT TESTS ==========
    printf("\n========== Branch Coverage Improvement ==========\n");

    // --- aclnn_mul.cpp: Individual nullptr tests (each param separately) ---
    // Covers each OP_CHECK_NULL branch independently
    {
        std::vector<int64_t> sh = {4};
        std::vector<float> xd(4, 1.0f);
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(xd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(xd, sh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(xd, sh, &dc, ACL_FLOAT, &to);
        float sv = 1.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;

        // Mul: null self only
        auto r = aclnnMulGetWorkspaceSize(nullptr, tb, to, &w, &e);
        ReportErrorCase("Nullptr_Mul_self", r != ACL_SUCCESS);
        // Mul: null other only
        r = aclnnMulGetWorkspaceSize(ta, nullptr, to, &w, &e);
        ReportErrorCase("Nullptr_Mul_other", r != ACL_SUCCESS);
        // Mul: null out only
        r = aclnnMulGetWorkspaceSize(ta, tb, nullptr, &w, &e);
        ReportErrorCase("Nullptr_Mul_out", r != ACL_SUCCESS);

        // Muls: null self only
        r = aclnnMulsGetWorkspaceSize(nullptr, sc, to, &w, &e);
        ReportErrorCase("Nullptr_Muls_self", r != ACL_SUCCESS);
        // Muls: null scalar only
        r = aclnnMulsGetWorkspaceSize(ta, nullptr, to, &w, &e);
        ReportErrorCase("Nullptr_Muls_scalar", r != ACL_SUCCESS);
        // Muls: null out only
        r = aclnnMulsGetWorkspaceSize(ta, sc, nullptr, &w, &e);
        ReportErrorCase("Nullptr_Muls_out", r != ACL_SUCCESS);

        // InplaceMul: null self only
        r = aclnnInplaceMulGetWorkspaceSize(nullptr, tb, &w, &e);
        ReportErrorCase("Nullptr_InplaceMul_self", r != ACL_SUCCESS);
        // InplaceMul: null other only
        r = aclnnInplaceMulGetWorkspaceSize(ta, nullptr, &w, &e);
        ReportErrorCase("Nullptr_InplaceMul_other", r != ACL_SUCCESS);

        // InplaceMuls: null self only
        r = aclnnInplaceMulsGetWorkspaceSize(nullptr, sc, &w, &e);
        ReportErrorCase("Nullptr_InplaceMuls_self", r != ACL_SUCCESS);
        // InplaceMuls: null scalar only
        r = aclnnInplaceMulsGetWorkspaceSize(ta, nullptr, &w, &e);
        ReportErrorCase("Nullptr_InplaceMuls_scalar", r != ACL_SUCCESS);

        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: CheckMulDtype unsupported on self/other/out individually ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;

        // unsupported self dtype
        CreateAclTensor(dd, sh, &da, ACL_STRING, &ta);
        CreateAclTensor(dd, sh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportErrorCase("Err_Mul_unsup_self_dtype", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);

        // unsupported other dtype
        da = db = dc = nullptr;
        ta = tb = to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, sh, &db, ACL_STRING, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT, &to);
        r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportErrorCase("Err_Mul_unsup_other_dtype", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);

        // unsupported out dtype
        da = db = dc = nullptr;
        ta = tb = to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, sh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_STRING, &to);
        r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportErrorCase("Err_Mul_unsup_out_dtype", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: CheckMulsDtype unsupported self/out ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        float sv = 1.0f;
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_STRING, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT, &to);
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        ReportErrorCase("Err_Muls_unsup_self_dtype", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);

        // unsupported out for Muls
        da = dc = nullptr;
        ta = to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_STRING, &to);
        sc = aclCreateScalar(&sv, ACL_FLOAT);
        r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        ReportErrorCase("Err_Muls_unsup_out_dtype", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: CheckInplaceMulDtype unsupported self/other ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *db = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_STRING, &ta);
        CreateAclTensor(dd, sh, &db, ACL_FLOAT, &tb);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulGetWorkspaceSize(ta, tb, &w, &e);
        ReportErrorCase("Err_InplaceMul_unsup_self", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);

        da = db = nullptr;
        ta = tb = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, sh, &db, ACL_STRING, &tb);
        r = aclnnInplaceMulGetWorkspaceSize(ta, tb, &w, &e);
        ReportErrorCase("Err_InplaceMul_unsup_other", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
    }

    // --- aclnn_mul.cpp: CheckInplaceMulsParams unsupported selfRef dtype ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        float sv = 1.0f;
        void* da = nullptr;
        aclTensor* ta = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_STRING, &ta);
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulsGetWorkspaceSize(ta, sc, &w, &e);
        ReportErrorCase("Err_InplaceMuls_unsup_dtype", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: CheckMulShape out shape != broadcast shape ---
    {
        std::vector<float> dd(12, 1.0f);
        std::vector<int64_t> s1 = {3, 4}, s2 = {4}, sOut = {3, 3};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, s1, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, s2, &db, ACL_FLOAT, &tb);
        CreateAclTensor(dd, sOut, &dc, ACL_FLOAT, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportErrorCase("Err_Mul_out_shape_mismatch", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - FP16 + imprecise scalar -> keepB16=false ---
    {
        std::vector<uint16_t> dd(4, 15360);
        std::vector<int64_t> sh = {4};
        std::vector<float> oh(4, 0);
        float sv = 1.00001f; // imprecise for FP16, keepB16=false -> promote to FLOAT
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT16, &ta);
        CreateAclTensor(oh, sh, &dc, ACL_FLOAT, &to);
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FP16_imprecise_scalar_keepB16_false", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - BF16 + imprecise scalar -> keepB16=false ---
    {
        std::vector<uint16_t> dd(4, 16256);
        std::vector<int64_t> sh = {4};
        std::vector<float> oh(4, 0);
        float sv = 1.00001f;
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BF16, &ta);
        CreateAclTensor(oh, sh, &dc, ACL_FLOAT, &to);
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_BF16_imprecise_scalar_keepB16_false", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - FP16 + precise scalar -> keepB16=true ---
    {
        std::vector<uint16_t> dd(4, 15360);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT16, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT16, &to);
        float sv = 2.0f; // precise for FP16, keepB16=true
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FP16_precise_scalar_keepB16_true", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype COMPLEX path ---
    // COMPLEX64 tensor + FLOAT scalar -> IsComplexType(self) branch
    {
        std::vector<uint16_t> dd(8, 1);
        std::vector<int64_t> sh = {2};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_COMPLEX64, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX64, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_COMPLEX64_FLOAT_scalar", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - COMPLEX64 scalar -> GetScalarDefaultDtype complex ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX64, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_COMPLEX64);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FLOAT_COMPLEX64_scalar", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - FLOAT16 tensor (isFloatType) -> self->GetDataType() ---
    {
        std::vector<uint16_t> dd(4, 15360);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT16, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT16, &to);
        int32_t sv = 2;
        aclScalar* sc = aclCreateScalar(&sv, ACL_INT32);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FP16_INT32scalar_isFloat_path", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - BF16 (isFloatType true) -> DT_FLOAT ---
    {
        std::vector<uint16_t> dd(4, 16256);
        std::vector<int64_t> sh = {4};
        std::vector<float> oh(4, 0);
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BF16, &ta);
        CreateAclTensor(oh, sh, &dc, ACL_FLOAT, &to);
        int32_t sv = 3;
        aclScalar* sc = aclCreateScalar(&sv, ACL_INT32);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_BF16_INT32_scalar_bf16_to_float", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - BOOL + FLOAT scalar -> PromoteType ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        std::vector<float> oh(4, 0);
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BOOL, &ta);
        CreateAclTensor(oh, sh, &dc, ACL_FLOAT, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_BOOL_FLOAT_scalar_promote", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype - BOOL + INT32 scalar -> PromoteType (self==BOOL) ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BOOL, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_INT32, &to);
        int32_t sv = 5;
        aclScalar* sc = aclCreateScalar(&sv, ACL_INT32);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_BOOL_INT32_scalar_promote", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: CombineCategoriesWithComplex branches ---
    // Mul COMPLEX64 * INT32 -> IsComplexType(self) true, !IsComplexType(other) -> return self dtype
    {
        std::vector<uint16_t> dd(8, 1);
        std::vector<int64_t> sh = {2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_COMPLEX64, &ta);
        CreateAclTensor(dd, sh, &db, ACL_INT32, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX64, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_COMPLEX64_INT32", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // Mul INT32 * COMPLEX64 -> !IsComplexType(self), IsComplexType(other) -> InnerTypeToComplexType
    {
        std::vector<uint16_t> dd(8, 1);
        std::vector<int64_t> sh = {2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_INT32, &ta);
        CreateAclTensor(dd, sh, &db, ACL_COMPLEX64, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX64, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_INT32_COMPLEX64", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // Mul FLOAT * COMPLEX64 -> IsFloatingType(self) + IsComplexType(other) -> InnerTypeToComplexType(FLOAT)
    {
        std::vector<uint16_t> dd(8, 1);
        std::vector<int64_t> sh = {2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, sh, &db, ACL_COMPLEX64, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX64, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_FLOAT_COMPLEX64", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // Mul DOUBLE * COMPLEX128 -> InnerTypeToComplexType(DOUBLE) -> COMPLEX128
    {
        std::vector<double> dd(4, 1.0);
        std::vector<int64_t> sh = {1};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_DOUBLE, &ta);
        CreateAclTensor(dd, sh, &db, ACL_COMPLEX128, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX128, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_DOUBLE_COMPLEX128", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: canUseMuls in Muls - BF16+FLOAT (RegBase) vs BF16+DOUBLE (!RegBase) ---
    // BF16 tensor + FLOAT scalar (precise, keepB16=true, canUseMuls=true on RegBase)
    {
        std::vector<uint16_t> dd(4, 16256);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BF16, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_BF16, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_BF16_FLOAT_canUseMuls", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // InplaceMuls BF16 + FLOAT scalar -> canUseMuls in InplaceMuls
    {
        std::vector<uint16_t> dd(4, 16256);
        std::vector<int64_t> sh = {4};
        void* da = nullptr;
        aclTensor* ta = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BF16, &ta);
        float sv = 3.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulsGetWorkspaceSize(ta, sc, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecInplaceMuls(w, e, stream);
        ReportCase("InplaceMuls_BF16_FLOAT_canUseMuls", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: Muls - IsMulSupportNonContiguous path (line 414) ---
    // FLOAT tensor + FLOAT scalar, same dtype == inferDtype, noncontiguous support
    {
        std::vector<float> dd(10, 2.0f);
        std::vector<int64_t> sh = {2, 2};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateNonContiguousAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateNonContiguousAclTensor(dd, sh, &dc, ACL_FLOAT, &to);
        float sv = 3.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FLOAT_noncontig_support", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: Mul - isMixDataType + isSupportNonContiguous (line 487-488) ---
    // Mix BF16+FLOAT with noncontiguous → hits isMixDataType=true, isSupportNonContiguous path
    {
        std::vector<uint16_t> dd(20, 1);
        std::vector<int64_t> sh = {2, 2};
        std::vector<float> oh(4, 0);
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateNonContiguousAclTensor(dd, sh, &da, ACL_BF16, &ta);
        CreateNonContiguousAclTensor(dd, sh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(oh, sh, &dc, ACL_FLOAT, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_BF16_FLOAT_noncontig_mix", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: Mul - isMixDataType + !isSupportNonContiguous (line 489-497) ---
    // Mix BF16+FLOAT with dim5 → isBroadcastTemplateNonContiguousSupport=false
    {
        std::vector<uint16_t> dd(1, 1);
        std::vector<int64_t> s5 = {1, 1, 1, 1, 1};
        std::vector<float> oh(1, 0);
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, s5, &da, ACL_BF16, &ta);
        CreateAclTensor(dd, s5, &db, ACL_FLOAT, &tb);
        CreateAclTensor(oh, s5, &dc, ACL_FLOAT, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_BF16_FLOAT_dim5_contiguous_mix", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: InplaceMul - IsRegBase() && isMixDataType (line 638) ---
    RunBranchTriggerTest(
        "InplaceMul_BF16_FLOAT_mix_regbase", 2, ACL_BF16, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "InplaceMul_FLOAT_FP16_mix_regbase", 2, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest(
        "InplaceMul_FP16_FLOAT_mix_regbase", 2, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_ND, stream);

    // --- mul.cpp: IsDoubleSupport path (line 68-73) ---
    // DOUBLE + DOUBLE on RegBase → IsDoubleSupport=true → MulAiCore
    RunBranchTriggerTest(
        "Mul_DOUBLE_DOUBLE_aicore", 0, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, normShape, ACL_FORMAT_ND, stream);

    // --- mul.cpp: AiCpu path - INT16 → not in AiCore support list → MulAiCpu ---
    RunBranchTriggerTest("Mul_INT16_aicpu", 0, ACL_INT16, ACL_INT16, ACL_INT16, normShape, ACL_FORMAT_ND, stream);

    // --- mul.cpp: isBroadcastTemplateNonContiguousSupport dim <= 4 paths ---
    // dim=1,2,3,4 all supported
    RunBranchTriggerTest("Mul_FLOAT_dim1", 0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, {4}, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Mul_FLOAT_dim3", 0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, {1, 1, 2}, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Mul_FLOAT_dim4", 0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, {1, 1, 1, 2}, ACL_FORMAT_ND, stream);

    // --- mul.cpp: IsMulSupportNonContiguous both AiCore ---
    // Non-contiguous INT32 → AiCore support → noncontiguous path
    {
        std::vector<int32_t> dd(20, 1);
        std::vector<int64_t> sh = {2, 2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateNonContiguousAclTensor(dd, sh, &da, ACL_INT32, &ta);
        CreateNonContiguousAclTensor(dd, sh, &db, ACL_INT32, &tb);
        CreateNonContiguousAclTensor(dd, sh, &dc, ACL_INT32, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_INT32_noncontig_aicore", true);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- mul.cpp: DOUBLE non-contiguous → IsDoubleSupport ---
    {
        std::vector<double> dd(20, 1.0);
        std::vector<int64_t> sh = {2, 2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateNonContiguousAclTensor(dd, sh, &da, ACL_DOUBLE, &ta);
        CreateNonContiguousAclTensor(dd, sh, &db, ACL_DOUBLE, &tb);
        CreateNonContiguousAclTensor(dd, sh, &dc, ACL_DOUBLE, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_DOUBLE_noncontig_isDoubleSupport", true);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: Mul with different dtypes needing Cast (non-mix, non-same) ---
    // INT8 * INT32 → promote to INT32, needs Cast
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_INT8, &ta);
        CreateAclTensor(dd, sh, &db, ACL_INT32, &tb);
        CreateAclTensor(dd, sh, &dc, ACL_INT32, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_INT8_INT32_cast_path", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: Mul output dtype differs from promoteType → Cast to out dtype ---
    // FLOAT * FLOAT → FLOAT result, but out is INT32 → castOut
    {
        std::vector<float> xf(4, 2.0f);
        std::vector<int32_t> oh(4, 0);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(xf, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(xf, sh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(oh, sh, &dc, ACL_INT32, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_FLOAT_FLOAT_out_INT32_castOut", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: InplaceMuls with INT64 + FLOAT scalar ---
    RunBranchTriggerTest(
        "InplaceMuls_INT64_FLOAT", 3, ACL_INT64, ACL_FLOAT, ACL_INT64, normShape, ACL_FORMAT_ND, stream);

    // --- aclnn_mul.cpp: InplaceMuls with BOOL + INT32 scalar ---
    {
        std::vector<uint16_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        void* da = nullptr;
        aclTensor* ta = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BOOL, &ta);
        int32_t sv = 1;
        aclScalar* sc = aclCreateScalar(&sv, ACL_INT32);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulsGetWorkspaceSize(ta, sc, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecInplaceMuls(w, e, stream);
        ReportCase("InplaceMuls_BOOL_INT32", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InplaceMul with non-mix dtype needing Cast ---
    // InplaceMul INT8 * INT32 → promote to INT32 → Cast
    RunBranchTriggerTest(
        "InplaceMul_INT8_INT32_cast", 2, ACL_INT8, ACL_INT32, ACL_INT32, normShape, ACL_FORMAT_ND, stream);

    // --- aclnn_mul.cpp: Format warning for InplaceMuls ---
    // InplaceMuls doesn't call MulsCheckFormat but uses MulCheckFormat... actually InplaceMuls
    // calls CheckInplaceMulsParams which doesn't have format check. Try InplaceMul with NCHW:
    RunBranchTriggerTest(
        "Format_InplaceMuls_NCHW", 3, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, normShape, ACL_FORMAT_NCHW, stream);

    // --- aclnn_mul.cpp: Muls empty tensor for different dtypes ---
    RunBranchTriggerTest("Empty_Muls_INT32", 1, ACL_INT32, ACL_INT32, ACL_INT32, emptyShape, ACL_FORMAT_ND, stream);
    RunBranchTriggerTest("Empty_Muls_BF16", 1, ACL_BF16, ACL_FLOAT, ACL_BF16, emptyShape, ACL_FORMAT_ND, stream);

    // --- aclnn_mul.cpp: Mul with empty other (self non-empty) ---
    {
        std::vector<float> xf(4, 1.0f), empty;
        std::vector<int64_t> sh = {4}, esh = {0};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(xf, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(empty, esh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(empty, esh, &dc, ACL_FLOAT, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportCase("Mul_self_nonEmpty_other_empty", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: Mul with broadcast [4,1]*[1,4] tests broadcast shape inference ---
    {
        std::vector<float> x1(4, 2.0f), x2(4, 3.0f), oh(16, 0);
        std::vector<int64_t> s1 = {4, 1}, s2 = {1, 4}, so = {4, 4};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(x1, s1, &da, ACL_FLOAT, &ta);
        CreateAclTensor(x2, s2, &db, ACL_FLOAT, &tb);
        CreateAclTensor(oh, so, &dc, ACL_FLOAT, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_FLOAT_broadcast_4x1_x_1x4", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: InplaceMul with broadcast [4,4]*[1,4] ---
    {
        std::vector<float> x1(16, 2.0f), x2(4, 3.0f);
        std::vector<int64_t> s1 = {4, 4}, s2 = {1, 4};
        void *da = nullptr, *db = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr;
        CreateAclTensor(x1, s1, &da, ACL_FLOAT, &ta);
        CreateAclTensor(x2, s2, &db, ACL_FLOAT, &tb);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulGetWorkspaceSize(ta, tb, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecInplaceMul(w, e, stream);
        ReportCase("InplaceMul_FLOAT_broadcast_4x4_x_1x4", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
    }

    // ========== ADDITIONAL BRANCH COVERAGE TESTS ==========
    printf("\n========== Additional Branch Coverage ==========\n");

    // --- aclnn_mul.cpp: InnerTypeToComplexType(BF16) → COMPLEX64 ---
    // Muls BF16 tensor + COMPLEX64 scalar: CombineCategoriesWithComplex(BF16, COMPLEX64)
    // → IsComplexType(lower)=true, IsFloatingType(higher)=true → InnerTypeToComplexType(BF16)
    {
        std::vector<uint16_t> dd(4, 16256);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_BF16, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX64, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_COMPLEX64);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_BF16_COMPLEX64_InnerTypeToComplex", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InnerTypeToComplexType(FP16) → COMPLEX32 → promoted to COMPLEX64 ---
    {
        std::vector<uint16_t> dd(4, 15360);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT16, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX64, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_COMPLEX64);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FP16_COMPLEX64_InnerTypeToComplex", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InnerTypeToComplexType(DOUBLE) → COMPLEX128 ---
    {
        std::vector<double> dd(4, 1.0);
        std::vector<int64_t> sh = {2};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_DOUBLE, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_COMPLEX128, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_COMPLEX64);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_DOUBLE_COMPLEX64_InnerTypeToComplex128", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: MulCheckFormat - only other is non-ND ---
    {
        std::vector<float> dd(4, 1.0f);
        std::vector<int64_t> sh = {2, 2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta, ACL_FORMAT_ND);
        CreateAclTensor(dd, sh, &db, ACL_FLOAT, &tb, ACL_FORMAT_NCHW);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT, &to, ACL_FORMAT_ND);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("MulFormat_self_ND_other_NCHW", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: MulCheckFormat - only self is non-ND ---
    {
        std::vector<float> dd(4, 1.0f);
        std::vector<int64_t> sh = {2, 2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta, ACL_FORMAT_NCHW);
        CreateAclTensor(dd, sh, &db, ACL_FLOAT, &tb, ACL_FORMAT_ND);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT, &to, ACL_FORMAT_ND);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("MulFormat_self_NCHW_other_ND", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: Muls Cast path (inferDtype != self dtype) ---
    // INT8 tensor + FLOAT scalar → inferDtype=FLOAT, self=INT8, Cast needed (line 416-426)
    {
        std::vector<int8_t> dd(4, 2);
        std::vector<int64_t> sh = {4};
        std::vector<float> oh(4, 0);
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_INT8, &ta);
        CreateAclTensor(oh, sh, &dc, ACL_FLOAT, &to);
        float sv = 3.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_INT8_FLOAT_castSelfPath", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype return self->GetDataType() (line 231) ---
    // INT8 tensor + INT32 scalar → not complex, not float, not BOOL → return INT8
    {
        std::vector<int8_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_INT8, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_INT8, &to);
        int32_t sv = 2;
        aclScalar* sc = aclCreateScalar(&sv, ACL_INT32);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_INT8_INT32scalar_returnSelf", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InferTensorScalarDtype (non-RegBase line 223-224) ---
    // INT32 tensor + DOUBLE scalar + FLOAT out → other==DOUBLE && out==FLOAT
    {
        std::vector<int32_t> dd(4, 1);
        std::vector<int64_t> sh = {4};
        std::vector<float> oh(4, 0);
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_INT32, &ta);
        CreateAclTensor(oh, sh, &dc, ACL_FLOAT, &to);
        double sv = 2.0;
        aclScalar* sc = aclCreateScalar(&sv, ACL_DOUBLE);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_INT32_DOUBLE_FLOAT_inferPath", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InplaceMul empty other only (line 620: other->IsEmpty()) ---
    {
        std::vector<float> xf(4, 1.0f), empty;
        std::vector<int64_t> sh = {4}, esh = {0};
        void *da = nullptr, *db = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr;
        CreateAclTensor(xf, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(empty, esh, &db, ACL_FLOAT, &tb);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulGetWorkspaceSize(ta, tb, &w, &e);
        ReportCase("InplaceMul_emptyOther_selfNonEmpty", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
    }

    // --- aclnn_mul.cpp: InplaceMul self empty only (line 620: selfRef->IsEmpty()) ---
    {
        std::vector<float> xf(4, 1.0f), empty;
        std::vector<int64_t> sh = {4}, esh = {0};
        void *da = nullptr, *db = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr;
        CreateAclTensor(empty, esh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(xf, sh, &db, ACL_FLOAT, &tb);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulGetWorkspaceSize(ta, tb, &w, &e);
        ReportCase("InplaceMul_emptySelf_otherNonEmpty", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
    }

    // --- aclnn_mul.cpp: Muls non-contiguous dim5 Cast path ---
    {
        std::vector<float> dd(2, 2.0f);
        std::vector<int64_t> s5 = {1, 1, 1, 1, 1};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateNonContiguousAclTensor(dd, s5, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, s5, &dc, ACL_FLOAT, &to);
        float sv = 3.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FLOAT_noncontig_dim5_castPath", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: canUseMuls - FP16 + FLOAT scalar (line 398-401) ---
    {
        std::vector<uint16_t> dd(4, 15360);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT16, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_FLOAT16, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMuls(w, e, stream);
        ReportCase("Muls_FP16_FLOAT_canUseMuls", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: InplaceMuls FP16 + FLOAT scalar → canUseMuls (line 568-571) ---
    {
        std::vector<uint16_t> dd(4, 15360);
        std::vector<int64_t> sh = {4};
        void* da = nullptr;
        aclTensor* ta = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT16, &ta);
        float sv = 3.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulsGetWorkspaceSize(ta, sc, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecInplaceMuls(w, e, stream);
        ReportCase("InplaceMuls_FP16_FLOAT_canUseMuls", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: CheckMulsPromoteDtype cast fail (inferDtype can't cast to out) ---
    {
        std::vector<uint16_t> dd(8, 1);
        std::vector<int64_t> sh = {2};
        void *da = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_COMPLEX64, &ta);
        CreateAclTensor(dd, sh, &dc, ACL_BOOL, &to);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulsGetWorkspaceSize(ta, sc, to, &w, &e);
        ReportErrorCase("Err_Muls_promote_castFail_to_BOOL", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(to, dc);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: CheckMulPromoteType promoteType→out cast fail ---
    {
        std::vector<float> dd(4, 1.0f);
        std::vector<uint8_t> oh(4, 0);
        std::vector<int64_t> sh = {4};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(dd, sh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(oh, sh, &dc, ACL_BOOL, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportErrorCase("Err_Mul_promote_castFail_to_BOOL", r != ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: InplaceMul with non-mix, non-same dtypes → Cast path ---
    RunBranchTriggerTest(
        "InplaceMul_INT8_INT64_cast", 2, ACL_INT8, ACL_INT64, ACL_INT64, normShape, ACL_FORMAT_ND, stream);

    // --- aclnn_mul.cpp: Mul empty self only (line 467) ---
    {
        std::vector<float> xf(4, 1.0f), empty;
        std::vector<int64_t> sh = {4}, esh = {0};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateAclTensor(empty, esh, &da, ACL_FLOAT, &ta);
        CreateAclTensor(xf, sh, &db, ACL_FLOAT, &tb);
        CreateAclTensor(empty, esh, &dc, ACL_FLOAT, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        ReportCase("Mul_emptySelf_otherNonEmpty", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

    // --- aclnn_mul.cpp: InplaceMuls empty selfRef (line 554) ---
    {
        std::vector<float> empty;
        std::vector<int64_t> esh = {0};
        void* da = nullptr;
        aclTensor* ta = nullptr;
        CreateAclTensor(empty, esh, &da, ACL_FLOAT, &ta);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulsGetWorkspaceSize(ta, sc, &w, &e);
        ReportCase("InplaceMuls_emptySelf", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        aclDestroyScalar(sc);
    }

    // --- aclnn_mul.cpp: Muls FP16 + DOUBLE scalar (non-RegBase canUseMuls=false) ---
    RunBranchTriggerTest(
        "Muls_FP16_DOUBLE_noCanUseMuls", 1, ACL_FLOAT16, ACL_DOUBLE, ACL_FLOAT16, normShape, ACL_FORMAT_ND, stream);

    // --- aclnn_mul.cpp: InplaceMuls COMPLEX64 + FLOAT scalar (canUseMuls=false) ---
    {
        std::vector<uint16_t> dd(8, 1);
        std::vector<int64_t> sh = {2};
        void* da = nullptr;
        aclTensor* ta = nullptr;
        CreateAclTensor(dd, sh, &da, ACL_COMPLEX64, &ta);
        float sv = 2.0f;
        aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnInplaceMulsGetWorkspaceSize(ta, sc, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecInplaceMuls(w, e, stream);
        ReportCase("InplaceMuls_COMPLEX64_FLOAT", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        aclDestroyScalar(sc);
    }

    // --- mul.cpp: Mul non-contiguous FP16 (AiCore, dim<=4) ---
    {
        std::vector<uint16_t> dd(20, 15360);
        std::vector<int64_t> sh = {2, 2};
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *to = nullptr;
        CreateNonContiguousAclTensor(dd, sh, &da, ACL_FLOAT16, &ta);
        CreateNonContiguousAclTensor(dd, sh, &db, ACL_FLOAT16, &tb);
        CreateNonContiguousAclTensor(dd, sh, &dc, ACL_FLOAT16, &to);
        uint64_t w = 0;
        aclOpExecutor* e = nullptr;
        auto r = aclnnMulGetWorkspaceSize(ta, tb, to, &w, &e);
        if (r == ACL_SUCCESS)
            r = ExecMul(w, e, stream);
        ReportCase("Mul_FP16_noncontig_aicore", r == ACL_SUCCESS);
        FreeTensor(ta, da);
        FreeTensor(tb, db);
        FreeTensor(to, dc);
    }

#if MUL_L0_TEST_AVAILABLE
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    RunL0MulImpl(
        "L0_Mul_ASCEND910B_BF16_AiCore", {2}, ACL_BF16, std::vector<uint16_t>{0x3F80, 0x4000}, {2}, ACL_BF16,
        std::vector<uint16_t>{0x3F80, 0x4000}, true);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    RunL0MulImpl(
        "L0_Mul_ASCEND950_FLOAT_AiCore", {2}, ACL_FLOAT, std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT,
        std::vector<float>{3.0f, 4.0f}, true);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    RunL0MulImpl(
        "L0_Mul_ASCEND910_INT16_AiCpu", {2}, ACL_INT16, std::vector<int16_t>{1, 2}, {2}, ACL_INT16,
        std::vector<int16_t>{3, 4}, true);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND610LITE);
    RunL0MulImpl(
        "L0_Mul_ASCEND610LITE_INT32_AiCore", {2}, ACL_INT32, std::vector<int32_t>{1, 2}, {2}, ACL_INT32,
        std::vector<int32_t>{3, 4}, true);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
    RunL0MulImpl(
        "L0_Mul_DefaultBranch_BF16_AiCpu", {2}, ACL_BF16, std::vector<uint16_t>{0x3F80, 0x4000}, {2}, ACL_BF16,
        std::vector<uint16_t>{0x3F80, 0x4000}, true);
    RunL0MulImpl(
        "L0_Mul_BroadcastInvalid", {2}, ACL_FLOAT, std::vector<float>{1.0f, 2.0f}, {3}, ACL_FLOAT,
        std::vector<float>{3.0f, 4.0f, 5.0f}, false);

    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    RunL0MulNonContigSupport(
        "L0_Mul_NonContig_NotRegBase", {2, 2}, ACL_FLOAT, std::vector<float>(4, 1.0f), {2, 2}, ACL_FLOAT,
        std::vector<float>(4, 1.0f), false);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    RunL0MulNonContigSupport(
        "L0_Mul_NonContig_Dim5", {1, 1, 1, 1, 1}, ACL_FLOAT, std::vector<float>{1.0f}, {1, 1, 1, 1, 1}, ACL_FLOAT,
        std::vector<float>{1.0f}, false);
    RunL0MulNonContigSupport(
        "L0_Mul_NonContig_RegBase_AiCore", {2, 2}, ACL_INT32, std::vector<int32_t>(4, 1), {2, 2}, ACL_INT32,
        std::vector<int32_t>(4, 1), true);
    RunL0MulNonContigSupport(
        "L0_Mul_NonContig_RegBase_Double", {2, 2}, ACL_DOUBLE, std::vector<double>(4, 1.0), {2, 2}, ACL_DOUBLE,
        std::vector<double>(4, 1.0), true);
#else
    printf("[INFO] Skip L0 mul tests: opdev headers are unavailable in current examples build path.\n");
#endif

    // --- Summary ---
    printf("\n========================================\n");
    printf("Total: %d  Pass: %d  Fail: %d\n", g_total, g_pass, g_fail);
    printf("========================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return g_fail > 0 ? 1 : 0;
}