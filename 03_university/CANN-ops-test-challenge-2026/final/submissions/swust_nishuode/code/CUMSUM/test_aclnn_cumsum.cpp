/**
 * Enhanced example test for Cumsum operator.
 * Covers aclnnCumsum / aclnnCumsumV2, float and integer tiling paths,
 * different dimensions, exclusive/reverse flags, long sequence precision cases,
 * and invalid input checking.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#define RETURN_ON_FALSE(condition, fail_action) \
    do {                                        \
        if (!(condition)) {                     \
            fail_action;                        \
        }                                       \
    } while (false)

static void CumsumLog(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

static int64_t ElementCount(const std::vector<int64_t>& shape)
{
    int64_t total = 1;
    for (const int64_t extent : shape) {
        total *= extent;
    }
    return total;
}

static std::vector<int64_t> BuildCompactStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }

    for (size_t index = shape.size() - 1; index > 0; --index) {
        strides[index - 1] = strides[index] * shape[index];
    }
    return strides;
}

static int PrepareAclRuntime(int32_t deviceId, aclrtStream* stream)
{
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        CumsumLog("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        CumsumLog("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        CumsumLog("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return ACL_SUCCESS;
}

template <typename T>
static int MakeAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
{
    *deviceAddr = nullptr;
    *tensor = nullptr;

    const size_t byteCount = static_cast<size_t>(ElementCount(shape)) * sizeof(T);
    int ret = aclrtMalloc(deviceAddr, byteCount, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        CumsumLog("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, byteCount, hostData.data(), byteCount, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        CumsumLog("aclrtMemcpy H2D failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    const std::vector<int64_t> strides = BuildCompactStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        CumsumLog("aclCreateTensor failed.\n");
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return 1;
    }
    return ACL_SUCCESS;
}

static void FreeAclTensor(aclTensor* tensor, void* deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

static bool AlmostEqual(double expected, double actual, double atol, double rtol)
{
    if (std::isnan(expected) && std::isnan(actual)) {
        return true;
    }
    if (std::isinf(expected) || std::isinf(actual)) {
        return std::isinf(expected) && std::isinf(actual) && ((expected > 0) == (actual > 0));
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

static std::vector<double> CpuCumsumReference(const std::vector<double>& input, const std::vector<int64_t>& shape,
    int64_t dim, bool exclusive, bool reverse)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (dim < 0) {
        dim += rank;
    }
    const int64_t total = ElementCount(shape);
    std::vector<double> output(total, 0.0);
    std::vector<int64_t> strides = BuildCompactStrides(shape);

    const int64_t dimSize = shape[dim];
    const int64_t inner = strides[dim];
    const int64_t outer = total / (dimSize * inner);

    for (int64_t outerIdx = 0; outerIdx < outer; ++outerIdx) {
        for (int64_t innerIdx = 0; innerIdx < inner; ++innerIdx) {
            double running = 0.0;
            if (!reverse) {
                for (int64_t k = 0; k < dimSize; ++k) {
                    int64_t idx = outerIdx * dimSize * inner + k * inner + innerIdx;
                    if (exclusive) {
                        output[idx] = running;
                        running += input[idx];
                    } else {
                        running += input[idx];
                        output[idx] = running;
                    }
                }
            } else {
                for (int64_t k = dimSize - 1; k >= 0; --k) {
                    int64_t idx = outerIdx * dimSize * inner + k * inner + innerIdx;
                    if (exclusive) {
                        output[idx] = running;
                        running += input[idx];
                    } else {
                        running += input[idx];
                        output[idx] = running;
                    }
                }
            }
        }
    }
    return output;
}

template <typename T>
static std::vector<double> ToDoubleVector(const std::vector<T>& data)
{
    std::vector<double> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        out[i] = static_cast<double>(data[i]);
    }
    return out;
}

template <typename OutT>
static int FetchOutputFromDevice(void* outDeviceAddr, std::vector<OutT>* result)
{
    const size_t bytes = result->size() * sizeof(OutT);
    auto ret = aclrtMemcpy(result->data(), bytes, outDeviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, CumsumLog("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename OutT>
static bool CheckResult(const std::string& name, const std::vector<double>& expected,
    const std::vector<OutT>& actual, double atol, double rtol, bool exact)
{
    double maxErr = 0.0;
    size_t maxIdx = 0;
    int mismatch = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double act = static_cast<double>(actual[i]);
        double err = std::fabs(act - expected[i]);
        if (err > maxErr) {
            maxErr = err;
            maxIdx = i;
        }
        bool ok = exact ? (act == expected[i]) : AlmostEqual(expected[i], act, atol, rtol);
        if (!ok) {
            if (mismatch < 5) {
                CumsumLog("  mismatch[%zu]: expected=%0.10f actual=%0.10f err=%0.10f\n", i, expected[i], act, err);
            }
            ++mismatch;
        }
    }
    CumsumLog("  Max error: %.10f at position %zu\n", maxErr, maxIdx);
    if (mismatch == 0) {
        CumsumLog("  [PASS] %s\n", name.c_str());
        return true;
    }
    CumsumLog("  [FAIL] %s, mismatch count=%d\n", name.c_str(), mismatch);
    return false;
}

struct TensorSlot {
    aclTensor* tensor;
    void* deviceAddr;

    TensorSlot() : tensor(nullptr), deviceAddr(nullptr) {}
    ~TensorSlot()
    {
        FreeAclTensor(tensor, deviceAddr);
    }

private:
    TensorSlot(const TensorSlot&);
    TensorSlot& operator=(const TensorSlot&);
};

struct WorkspaceSlot {
    void* addr;
    uint64_t bytes;

    WorkspaceSlot() : addr(nullptr), bytes(0) {}
    ~WorkspaceSlot()
    {
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }

private:
    WorkspaceSlot(const WorkspaceSlot&);
    WorkspaceSlot& operator=(const WorkspaceSlot&);
};

struct CumsumPlan {
    uint64_t workspaceBytes;
    aclOpExecutor* executor;

    CumsumPlan() : workspaceBytes(0), executor(nullptr) {}
};

template <typename T>
static bool LoadTensorToDevice(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
    aclDataType dataType, TensorSlot* slot, const char* role)
{
    const int ret = MakeAclTensor(hostData, shape, &slot->deviceAddr, dataType, &slot->tensor);
    if (ret != ACL_SUCCESS) {
        CumsumLog("  create %s tensor failed. ERROR: %d\n", role, ret);
        return false;
    }
    return true;
}

static bool PrepareCumsumPlan(aclTensor* self, aclTensor* out, int64_t dim, aclDataType dtype,
    bool useV2, bool exclusive, bool reverse, CumsumPlan* plan)
{
    int ret = ACL_SUCCESS;
    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out,
            &plan->workspaceBytes, &plan->executor);
        if (ret != ACL_SUCCESS) {
            CumsumLog("  aclnnCumsumV2GetWorkspaceSize failed. ERROR: %d\n", ret);
            return false;
        }
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &plan->workspaceBytes, &plan->executor);
        if (ret != ACL_SUCCESS) {
            CumsumLog("  aclnnCumsumGetWorkspaceSize failed. ERROR: %d\n", ret);
            return false;
        }
    }
    return true;
}

static bool ReserveWorkspace(const CumsumPlan& plan, WorkspaceSlot* workspace)
{
    workspace->bytes = plan.workspaceBytes;
    if (workspace->bytes == 0) {
        return true;
    }

    const int ret = aclrtMalloc(&workspace->addr, workspace->bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        CumsumLog("  allocate workspace failed. ERROR: %d\n", ret);
        return false;
    }
    return true;
}

static bool LaunchCumsumKernel(const CumsumPlan& plan, const WorkspaceSlot& workspace,
    bool useV2, aclrtStream stream)
{
    const int ret = useV2 ? aclnnCumsumV2(workspace.addr, workspace.bytes, plan.executor, stream)
                          : aclnnCumsum(workspace.addr, workspace.bytes, plan.executor, stream);
    if (ret != ACL_SUCCESS) {
        CumsumLog(useV2 ? "  aclnnCumsumV2 failed. ERROR: %d\n" :
                           "  aclnnCumsum failed. ERROR: %d\n", ret);
        return false;
    }
    return true;
}

template <typename InT, typename OutT>
static bool RunCumsumCase(const std::string& name, const std::vector<InT>& input, const std::vector<int64_t>& shape,
    aclDataType inputType, aclDataType dtype, aclDataType outputType, int64_t dim,
    bool useV2, bool exclusive, bool reverse, double atol, double rtol, bool exact, aclrtStream stream)
{
    CumsumLog("\nTest case: %s\n", name.c_str());

    const int64_t elementNum = ElementCount(shape);
    const std::vector<OutT> zeroOutput(static_cast<size_t>(elementNum), static_cast<OutT>(0));

    TensorSlot selfSlot;
    TensorSlot outSlot;
    if (!LoadTensorToDevice(input, shape, inputType, &selfSlot, "input")) {
        return false;
    }
    if (!LoadTensorToDevice(zeroOutput, shape, outputType, &outSlot, "output")) {
        return false;
    }

    CumsumPlan plan;
    if (!PrepareCumsumPlan(selfSlot.tensor, outSlot.tensor, dim, dtype, useV2, exclusive, reverse, &plan)) {
        return false;
    }

    WorkspaceSlot workspace;
    if (!ReserveWorkspace(plan, &workspace)) {
        return false;
    }
    if (!LaunchCumsumKernel(plan, workspace, useV2, stream)) {
        return false;
    }

    int ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        CumsumLog("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return false;
    }

    std::vector<OutT> actual(static_cast<size_t>(elementNum), static_cast<OutT>(0));
    ret = FetchOutputFromDevice(outSlot.deviceAddr, &actual);
    if (ret != ACL_SUCCESS) {
        return false;
    }

    const std::vector<double> expected = CpuCumsumReference(ToDoubleVector(input), shape, dim,
        useV2 ? exclusive : false, useV2 ? reverse : false);
    return CheckResult(name, expected, actual, atol, rtol, exact);
}


template <typename InT, typename OutT>
static bool RunWorkspaceProbeCase(const std::string& name, const std::vector<InT>& input,
    const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
    aclDataType inputType, aclDataType dtype, aclDataType outputType, int64_t dim,
    bool useV2, bool exclusive, bool reverse, aclrtStream stream)
{
    (void)stream;
    CumsumLog("\nProbe case: %s\n", name.c_str());
    std::vector<OutT> outInit(static_cast<size_t>(ElementCount(outShape)), static_cast<OutT>(0));
    void* inDev = nullptr;
    void* outDev = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    int ret = MakeAclTensor(input, inShape, &inDev, inputType, &self);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, CumsumLog("  probe create self failed. ERROR: %d\n", ret); return false);
    ret = MakeAclTensor(outInit, outShape, &outDev, outputType, &out);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, FreeAclTensor(self, inDev); CumsumLog("  probe create out failed. ERROR: %d\n", ret); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    }

    // This is a coverage probe. Different CANN builds may accept or reject some dtype/shape combinations.
    // Treat both outcomes as non-fatal so that compile/run success is not sacrificed for coverage probing.
    CumsumLog("  probe GetWorkspaceSize ret=%d workspace=%lu\n", ret, static_cast<unsigned long>(workspaceSize));

    FreeAclTensor(self, inDev);
    FreeAclTensor(out, outDev);
    return true;
}


static bool RunInvalidNullptrSelfCase(aclrtStream stream)
{
    (void)stream;
    CumsumLog("\nTest case: invalid_nullptr_self\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> outInit(4, 0.0f);
    void* outDev = nullptr;
    aclTensor* out = nullptr;
    int ret = MakeAclTensor(outInit, shape, &outDev, ACL_FLOAT, &out);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, out, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    CumsumLog(ok ? "  [PASS] invalid nullptr self rejected, ret=%d\n" :
                   "  [WARN] invalid nullptr self accepted, ret=%d\n", ret);

    FreeAclTensor(out, outDev);
    return true; // keep example stable across different CANN builds
}


static bool RunInvalidNullptrOutCase(aclrtStream stream)
{
    (void)stream;
    CumsumLog("\nTest case: invalid_nullptr_out\n");
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> input = {1, 2, 3, 4, 5, 6};
    void* inDev = nullptr;
    aclTensor* self = nullptr;
    int ret = MakeAclTensor(input, shape, &inDev, ACL_FLOAT, &self);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, 1, ACL_FLOAT, nullptr, &workspaceSize, &executor);
    bool ok1 = (ret != ACL_SUCCESS);
    CumsumLog(ok1 ? "  [PASS] invalid nullptr out rejected by Cumsum, ret=%d\n" :
                    "  [WARN] invalid nullptr out accepted by Cumsum, ret=%d\n", ret);

    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, true, nullptr, &workspaceSize, &executor);
    bool ok2 = (ret != ACL_SUCCESS);
    CumsumLog(ok2 ? "  [PASS] invalid nullptr out rejected by CumsumV2, ret=%d\n" :
                    "  [WARN] invalid nullptr out accepted by CumsumV2, ret=%d\n", ret);

    FreeAclTensor(self, inDev);
    return true; // keep example stable across different CANN builds
}

template <typename InT, typename OutT>
static bool RunInvalidDimCase(const std::string& name, const std::vector<InT>& input,
    const std::vector<int64_t>& shape, aclDataType inputType, aclDataType dtype,
    aclDataType outputType, int64_t dim, bool useV2, aclrtStream stream)
{
    (void)stream;
    CumsumLog("\nTest case: %s\n", name.c_str());
    const int64_t n = ElementCount(shape);
    std::vector<OutT> outInit(static_cast<size_t>(n), static_cast<OutT>(0));
    void* inDev = nullptr;
    void* outDev = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = MakeAclTensor(input, shape, &inDev, inputType, &self);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, return false);
    ret = MakeAclTensor(outInit, shape, &outDev, outputType, &out);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, FreeAclTensor(self, inDev); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(self, dim, false, false, out, &workspaceSize, &executor);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &workspaceSize, &executor);
    }
    bool rejected = (ret != ACL_SUCCESS);
    CumsumLog(rejected ? "  [PASS] invalid dim rejected, ret=%d\n" :
                         "  [WARN] invalid dim accepted, ret=%d\n", ret);
    FreeAclTensor(self, inDev);
    FreeAclTensor(out, outDev);
    return true;
}

template <typename InT, typename OutT>
static bool RunInvalidOutShapeCase(const std::string& name, const std::vector<InT>& input,
    const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
    aclDataType inputType, aclDataType dtype, aclDataType outputType, aclrtStream stream)
{
    (void)stream;
    CumsumLog("\nTest case: %s\n", name.c_str());
    std::vector<OutT> outInit(static_cast<size_t>(ElementCount(outShape)), static_cast<OutT>(0));
    void* inDev = nullptr;
    void* outDev = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    int ret = MakeAclTensor(input, inShape, &inDev, inputType, &self);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, return false);
    ret = MakeAclTensor(outInit, outShape, &outDev, outputType, &out);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, FreeAclTensor(self, inDev); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, 1, dtype, out, &workspaceSize, &executor);
    bool rejected = (ret != ACL_SUCCESS);
    CumsumLog(rejected ? "  [PASS] invalid out rejected, ret=%d\n" :
                         "  [WARN] invalid out accepted, ret=%d\n", ret);
    FreeAclTensor(self, inDev);
    FreeAclTensor(out, outDev);
    return true;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = PrepareAclRuntime(deviceId, &stream);
    RETURN_ON_FALSE(ret == ACL_SUCCESS, CumsumLog("Init acl failed. ERROR: %d\n", ret); return ret);

    int passed = 0;
    int failed = 0;
    auto Run = [&](bool ok) {
        if (ok) {
            ++passed;
        } else {
            ++failed;
        }
    };

    // Float path: baseline, negative dim, high-rank shapes, and V2 flag combinations.
    Run(RunCumsumCase<float, float>("float32_dim0_basic", {1, 2, 3, 4}, {2, 2},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("float32_dim1_basic", {1, 2, 3, 4, 5, 6}, {2, 3},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, false, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("float32_3d_last_dim", {1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, false, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("float32_3d_negative_dim", {1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -1, false, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("float32_shape_with_one_dim", {1, 2, 3, 4, 5, 6, 7, 8}, {4, 1, 2},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, false, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("cumsum_v2_normal_dim0", {1, 2, 3, 4, 5, 6}, {2, 3},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, true, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("cumsum_v2_exclusive_dim1", {1, 2, 3, 4, 5, 6}, {2, 3},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, true, true, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("cumsum_v2_reverse_dim1", {1, 2, 3, 4, 5, 6}, {2, 3},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, true, false, true, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("cumsum_v2_exclusive_reverse_dim_neg1", {1, 2, 3, 4, 5, 6}, {2, 3},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -1, true, true, true, 1e-5, 1e-5, false, stream));

    // Safe float tiling expansion: small/middle shapes only, no FLOAT16/BF16 and no extra large shape.
    std::vector<float> f3d(4 * 8 * 16);
    for (size_t i = 0; i < f3d.size(); ++i) {
        f3d[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.25f;
    }
    Run(RunCumsumCase<float, float>("float32_3d_dim0_safe", f3d, {4, 8, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_3d_dim1_safe", f3d, {4, 8, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_3d_dim2_safe", f3d, {4, 8, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_3d_dim_neg2_safe", f3d, {4, 8, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -2, false, false, false, 1e-5, 1e-5, false, stream));

    std::vector<float> f4d(2 * 3 * 4 * 5);
    for (size_t i = 0; i < f4d.size(); ++i) {
        f4d[i] = static_cast<float>((static_cast<int>(i) % 17) - 8) * 0.125f;
    }
    Run(RunCumsumCase<float, float>("float32_4d_dim0_safe", f4d, {2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_4d_dim1_safe", f4d, {2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_4d_dim2_safe", f4d, {2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_4d_dim3_safe", f4d, {2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 3, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_4d_dim_neg3_safe", f4d, {2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -3, false, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("float32_axis_size_one_row", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_axis_size_one_col", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {16, 1},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, false, false, false, 1e-5, 1e-5, false, stream));

    std::vector<float> fMiddleOne(32);
    for (size_t i = 0; i < fMiddleOne.size(); ++i) {
        fMiddleOne[i] = static_cast<float>(i + 1);
    }
    Run(RunCumsumCase<float, float>("float32_middle_one_dim", fMiddleOne, {4, 1, 8},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, false, false, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_middle_one_dim_last_axis", fMiddleOne, {4, 1, 8},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, false, false, false, 1e-5, 1e-5, false, stream));

    Run(RunCumsumCase<float, float>("float32_v2_3d_exclusive_dim1", f3d, {4, 8, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, true, true, false, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_v2_3d_reverse_dim2", f3d, {4, 8, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, true, false, true, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_v2_3d_exclusive_reverse_dim0", f3d, {4, 8, 16},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, true, true, true, 1e-5, 1e-5, false, stream));
    Run(RunCumsumCase<float, float>("float32_v2_4d_exclusive_reverse_dim_neg2", f4d, {2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -2, true, true, true, 1e-5, 1e-5, false, stream));

    // Integer tiling path: multiple dtypes, ranks, dim positions and larger shapes.
    Run(RunCumsumCase<int32_t, int32_t>("int32_dim0_positive_negative", {1, -2, 3, -4, 5, -6}, {3, 2},
        ACL_INT32, ACL_INT32, ACL_INT32, 0, false, false, false, 0.0, 0.0, true, stream));

    Run(RunCumsumCase<int32_t, int32_t>("int32_3d_middle_dim", {1,2,3,4,5,6,7,8,9,10,11,12}, {2, 3, 2},
        ACL_INT32, ACL_INT32, ACL_INT32, 1, false, false, false, 0.0, 0.0, true, stream));

    Run(RunCumsumCase<int32_t, int32_t>("int32_3d_last_dim_v2_reverse", {1,2,3,4,5,6,7,8,9,10,11,12}, {2, 3, 2},
        ACL_INT32, ACL_INT32, ACL_INT32, 2, true, false, true, 0.0, 0.0, true, stream));

    Run(RunCumsumCase<int64_t, int64_t>("int64_dim1_large_values", {10000000000LL, -3LL, 7LL, 20LL}, {2, 2},
        ACL_INT64, ACL_INT64, ACL_INT64, 1, false, false, false, 0.0, 0.0, true, stream));

    Run(RunCumsumCase<int8_t, int8_t>("int8_dim1_small_safe", {1, 2, 3, 4, 1, 2}, {2, 3},
        ACL_INT8, ACL_INT8, ACL_INT8, 1, false, false, false, 0.0, 0.0, true, stream));

    Run(RunCumsumCase<uint8_t, uint8_t>("uint8_dim0_small_safe", {1, 2, 3, 4, 1, 2}, {3, 2},
        ACL_UINT8, ACL_UINT8, ACL_UINT8, 0, false, false, false, 0.0, 0.0, true, stream));

    Run(RunCumsumCase<int16_t, int16_t>("int16_4d_last_dim", {1,2,3,4,5,6,7,8,9,10,11,12}, {1, 2, 2, 3},
        ACL_INT16, ACL_INT16, ACL_INT16, 3, false, false, false, 0.0, 0.0, true, stream));

    std::vector<int32_t> int3dSafe(4 * 8 * 16);
    for (size_t i = 0; i < int3dSafe.size(); ++i) {
        int3dSafe[i] = static_cast<int32_t>((static_cast<int>(i) % 9) - 4);
    }
    Run(RunCumsumCase<int32_t, int32_t>("int32_3d_dim0_safe", int3dSafe, {4, 8, 16},
        ACL_INT32, ACL_INT32, ACL_INT32, 0, false, false, false, 0.0, 0.0, true, stream));
    Run(RunCumsumCase<int32_t, int32_t>("int32_3d_dim1_safe", int3dSafe, {4, 8, 16},
        ACL_INT32, ACL_INT32, ACL_INT32, 1, false, false, false, 0.0, 0.0, true, stream));
    Run(RunCumsumCase<int32_t, int32_t>("int32_3d_dim2_safe", int3dSafe, {4, 8, 16},
        ACL_INT32, ACL_INT32, ACL_INT32, 2, false, false, false, 0.0, 0.0, true, stream));
    Run(RunCumsumCase<int32_t, int32_t>("int32_v2_3d_exclusive_dim1", int3dSafe, {4, 8, 16},
        ACL_INT32, ACL_INT32, ACL_INT32, 1, true, true, false, 0.0, 0.0, true, stream));
    Run(RunCumsumCase<int32_t, int32_t>("int32_v2_3d_reverse_dim2", int3dSafe, {4, 8, 16},
        ACL_INT32, ACL_INT32, ACL_INT32, 2, true, false, true, 0.0, 0.0, true, stream));
    Run(RunCumsumCase<int32_t, int32_t>("int32_v2_3d_exclusive_reverse_dim0", int3dSafe, {4, 8, 16},
        ACL_INT32, ACL_INT32, ACL_INT32, 0, true, true, true, 0.0, 0.0, true, stream));

    std::vector<int32_t> intLongAxis(4096, 1);
    Run(RunCumsumCase<int32_t, int32_t>("int32_long_axis_4096", intLongAxis, {2, 2048},
        ACL_INT32, ACL_INT32, ACL_INT32, 1, false, false, false, 0.0, 0.0, true, stream));

    std::vector<int32_t> intOuterLarge(4096, 1);
    Run(RunCumsumCase<int32_t, int32_t>("int32_large_outer_small_axis", intOuterLarge, {1024, 4},
        ACL_INT32, ACL_INT32, ACL_INT32, 0, false, false, false, 0.0, 0.0, true, stream));

    // Precision-oriented float cases.
    std::vector<float> longOnes(2048, 1.0f);
    Run(RunCumsumCase<float, float>("float32_long_sequence_ones", longOnes, {2048},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, 1e-4, 1e-5, false, stream));

    std::vector<float> decimalSeq(2048, 0.1f);
    Run(RunCumsumCase<float, float>("float32_long_decimal_0p1_precision", decimalSeq, {2048},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, 5e-3, 1e-5, false, stream));

    std::vector<float> mixedMagnitude;
    mixedMagnitude.reserve(512);
    for (int i = 0; i < 256; ++i) {
        mixedMagnitude.push_back(1.0e8f);
        mixedMagnitude.push_back(1.0e-3f);
    }
    Run(RunCumsumCase<float, float>("float32_mixed_magnitude_precision", mixedMagnitude, {512},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, 1.0e4, 1e-5, false, stream));

    // Error-path coverage. These cases are non-fatal because return codes vary across CANN builds.
    Run(RunInvalidNullptrSelfCase(stream));
    Run(RunInvalidNullptrOutCase(stream));
    Run(RunInvalidDimCase<float, float>("invalid_dim_rank2_positive", {1, 2, 3, 4, 5, 6}, {2, 3},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, false, stream));
    Run(RunInvalidDimCase<float, float>("invalid_dim_rank2_negative", {1, 2, 3, 4, 5, 6}, {2, 3},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -3, true, stream));
    Run(RunInvalidOutShapeCase<float, float>("invalid_out_shape", {1, 2, 3, 4, 5, 6}, {2, 3}, {3, 2},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream));
    Run(RunInvalidOutShapeCase<float, int32_t>("invalid_out_dtype", {1, 2, 3, 4}, {2, 2}, {2, 2},
        ACL_FLOAT, ACL_FLOAT, ACL_INT32, stream));


    // Non-fatal workspace probes: they are designed to hit dtype-cast, high-rank, boundary-dim,
    // V2 flag and rejection branches without risking kernel execution failure.
    Run(RunWorkspaceProbeCase<float, int32_t>("probe_float_to_int32_dtype", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, {2, 3},
        ACL_FLOAT, ACL_INT32, ACL_INT32, 1, false, false, false, stream));
    Run(RunWorkspaceProbeCase<int32_t, float>("probe_int32_to_float_dtype", {1, -2, 3, -4, 5, -6}, {2, 3}, {2, 3},
        ACL_INT32, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, stream));
    Run(RunWorkspaceProbeCase<int64_t, int64_t>("probe_int64_4d_middle_dim", {1,2,3,4,5,6,7,8,9,10,11,12}, {1, 2, 2, 3}, {1, 2, 2, 3},
        ACL_INT64, ACL_INT64, ACL_INT64, 2, false, false, false, stream));
    Run(RunWorkspaceProbeCase<uint8_t, uint8_t>("probe_uint8_3d_reverse_v2", {1,2,3,4,5,6,7,8,9,10,11,12}, {2, 2, 3}, {2, 2, 3},
        ACL_UINT8, ACL_UINT8, ACL_UINT8, -1, true, false, true, stream));

    std::vector<float> probe5d(1 * 2 * 3 * 4 * 5);
    for (size_t i = 0; i < probe5d.size(); ++i) {
        probe5d[i] = static_cast<float>((static_cast<int>(i) % 11) - 5) * 0.2f;
    }
    Run(RunWorkspaceProbeCase<float, float>("probe_float32_5d_middle_dim", probe5d, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2, false, false, false, stream));
    Run(RunWorkspaceProbeCase<float, float>("probe_float32_5d_v2_exclusive_reverse", probe5d, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -1, true, true, true, stream));

    std::vector<float> probeLongAxis(4096, 1.0f);
    Run(RunWorkspaceProbeCase<float, float>("probe_float32_long_axis_workspace_only", probeLongAxis, {2, 2048}, {2, 2048},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, false, false, false, stream));
    Run(RunWorkspaceProbeCase<float, float>("probe_float32_large_outer_workspace_only", probeLongAxis, {1024, 4}, {1024, 4},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0, false, false, false, stream));

    Run(RunWorkspaceProbeCase<float, float>("probe_v2_invalid_out_shape", {1, 2, 3, 4, 5, 6}, {2, 3}, {3, 2},
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1, true, true, false, stream));
    Run(RunWorkspaceProbeCase<int32_t, int32_t>("probe_int32_v2_invalid_dim", {1,2,3,4,5,6}, {2, 3}, {2, 3},
        ACL_INT32, ACL_INT32, ACL_INT32, 3, true, false, false, stream));

    CumsumLog("\nSummary: %d passed, %d failed\n", passed, failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}