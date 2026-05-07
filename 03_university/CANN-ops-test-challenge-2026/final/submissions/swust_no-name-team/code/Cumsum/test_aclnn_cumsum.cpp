#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
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

static int gPassed = 0;
static int gFailed = 0;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto v : shape) s *= v;
    return s;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), static_cast<int32_t>(shape.size()), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), static_cast<int32_t>(shape.size()), *deviceAddr);
    return 0;
}

// ========== Float32 helpers ==========

void CpuCumsumFloat32(const std::vector<float>& input, int64_t dim,
                        const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;

    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == 0) {
            (*output)[flat] = static_cast<double>(input[flat]);
        } else {
            (*output)[flat] = (*output)[flat - cumStride] + static_cast<double>(input[flat]);
        }
    }
}

void CpuCumsumExclusiveFloat32(const std::vector<float>& input, int64_t dim,
                                 const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == 0) {
            (*output)[flat] = 0.0;
        } else {
            (*output)[flat] = (*output)[flat - cumStride] + static_cast<double>(input[flat - cumStride]);
        }
    }
}

void CpuCumsumReverseFloat32(const std::vector<float>& input, int64_t dim,
                               const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];
    int64_t cumDimSize = shape[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == cumDimSize - 1) {
            (*output)[flat] = static_cast<double>(input[flat]);
        } else {
            (*output)[flat] = (*output)[flat + cumStride] + static_cast<double>(input[flat]);
        }
    }
}

// exclusive_rev(i) = sum(x[i+1..end]) = total - sum(x[0..i])
void CpuCumsumExclRevFloat32(const std::vector<float>& input, int64_t dim,
                               const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];
    int64_t dimLen = shape[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        int64_t baseFlat = flat - coords[cumDim] * cumStride;
        double prefix = 0.0;
        for (int64_t k = 0; k < coords[cumDim]; k++) {
            prefix += input[baseFlat + k * cumStride];
        }
        double total = 0.0;
        for (int64_t k = 0; k < dimLen; k++) {
            total += input[baseFlat + k * cumStride];
        }
        (*output)[flat] = total - prefix;
    }
}

// ========== Float16 helpers ==========

void FloatToF16(float f, uint16_t* out)
{
    uint16_t bits;
    std::memcpy(&bits, &f, sizeof(uint16_t));
    *out = bits;
}

float F16ToFloat(uint16_t f16)
{
    float f;
    std::memcpy(&f, &f16, sizeof(float));
    return f;
}

void CpuCumsumFloat16(const std::vector<uint16_t>& input, int64_t dim,
                        const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == 0) {
            (*output)[flat] = static_cast<double>(F16ToFloat(input[flat]));
        } else {
            (*output)[flat] = (*output)[flat - cumStride] + static_cast<double>(F16ToFloat(input[flat]));
        }
    }
}

void CpuCumsumExclusiveFloat16(const std::vector<uint16_t>& input, int64_t dim,
                                 const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == 0) {
            (*output)[flat] = 0.0;
        } else {
            (*output)[flat] = (*output)[flat - cumStride] + static_cast<double>(F16ToFloat(input[flat - cumStride]));
        }
    }
}

void CpuCumsumReverseFloat16(const std::vector<uint16_t>& input, int64_t dim,
                               const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];
    int64_t cumDimSize = shape[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == cumDimSize - 1) {
            (*output)[flat] = static_cast<double>(F16ToFloat(input[flat]));
        } else {
            (*output)[flat] = (*output)[flat + cumStride] + static_cast<double>(F16ToFloat(input[flat]));
        }
    }
}

// Float16 V2 exclusive+reverse: exclusive(i) = sum(x[0..i-1]), then reverse cumsum
// exclusive_rev(i) = sum(exclusive(j) for j in [i, end)) = total_sum - prefix_sum[i-1]
void CpuCumsumExclRevFloat16(const std::vector<uint16_t>& input, int64_t dim,
                               const std::vector<int64_t>& shape, std::vector<double>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0.0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];
    int64_t dimLen = shape[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        int64_t baseFlat = flat - coords[cumDim] * cumStride;
        double cumsum = 0.0;
        for (int64_t k = 0; k < coords[cumDim]; k++) {
            cumsum += F16ToFloat(input[baseFlat + k * cumStride]);
        }
        double total = 0.0;
        for (int64_t k = 0; k < dimLen; k++) {
            total += F16ToFloat(input[baseFlat + k * cumStride]);
        }
        (*output)[flat] = total - cumsum;
    }
}

// ========== Int32 helpers ==========

void CpuCumsumInt32(const std::vector<int32_t>& input, int64_t dim,
                      const std::vector<int64_t>& shape, std::vector<int64_t>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == 0) {
            (*output)[flat] = static_cast<int64_t>(input[flat]);
        } else {
            (*output)[flat] = (*output)[flat - cumStride] + static_cast<int64_t>(input[flat]);
        }
    }
}

void CpuCumsumExclusiveInt32(const std::vector<int32_t>& input, int64_t dim,
                               const std::vector<int64_t>& shape, std::vector<int64_t>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        if (coords[cumDim] == 0) {
            (*output)[flat] = 0;
        } else {
            (*output)[flat] = (*output)[flat - cumStride] + static_cast<int64_t>(input[flat - cumStride]);
        }
    }
}

void CpuCumsumReverseInt32(const std::vector<int32_t>& input, int64_t dim,
                              const std::vector<int64_t>& shape, std::vector<int64_t>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];
    int64_t dimLen = shape[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        int64_t pos = coords[cumDim];
        int64_t baseFlat = flat - pos * cumStride;
        double sum = 0.0;
        for (int64_t k = dimLen - 1; k > pos; k--) {
            sum += input[baseFlat + k * cumStride];
        }
        (*output)[flat] = static_cast<int64_t>(sum);
    }
}

void CpuCumsumExclusiveReverseInt32(const std::vector<int32_t>& input, int64_t dim,
                                     const std::vector<int64_t>& shape, std::vector<int64_t>* output)
{
    auto size = GetShapeSize(shape);
    output->assign(size, 0);
    if (size == 0) return;

    int64_t nd = static_cast<int64_t>(shape.size());
    int64_t cumDim = (dim < 0) ? nd + dim : dim;
    std::vector<int64_t> strides(nd, 1);
    for (int64_t i = nd - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t cumStride = strides[cumDim];
    int64_t dimLen = shape[cumDim];

    std::vector<int64_t> coords(nd);
    for (int64_t flat = 0; flat < size; flat++) {
        int64_t rem = flat;
        for (int64_t d = 0; d < nd; d++) {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        int64_t baseFlat = flat - coords[cumDim] * cumStride;
        // exclusive_rev(i) = sum(x[i+1..end]) = total - sum(x[0..i])
        int64_t prefix = 0;
        for (int64_t k = 0; k < coords[cumDim]; k++) {
            prefix += input[baseFlat + k * cumStride];
        }
        int64_t total = 0;
        for (int64_t k = 0; k < dimLen; k++) {
            total += input[baseFlat + k * cumStride];
        }
        (*output)[flat] = total - prefix;
    }
}

// ========== Result checking ==========

void CheckFloat32(const std::vector<float>& result, const std::vector<double>& expected,
                   double atol, double rtol, const char* name)
{
    auto n = result.size();
    double maxErr = 0.0;
    int64_t maxIdx = 0;
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        double err = std::abs(static_cast<double>(result[i]) - expected[i]);
        double tol = atol + rtol * std::abs(expected[i]);
        if (err > maxErr) { maxErr = err; maxIdx = i; }
        if (err > tol) ok = false;
    }
    if (ok) {
        LOG_PRINT("  [PASS]  max_error=%.6e at idx=%ld\n", maxErr, maxIdx);
        gPassed++;
    } else {
        LOG_PRINT("  [FAIL]  max_error=%.6e at idx=%ld (exp=%.6e act=%.6e)\n",
                  maxErr, maxIdx, expected[maxIdx], static_cast<double>(result[maxIdx]));
        gFailed++;
    }
}

void CheckFloat16(const std::vector<uint16_t>& result, const std::vector<double>& expected,
                  double atol, double rtol, const char* name)
{
    auto n = result.size();
    double maxErr = 0.0;
    int64_t maxIdx = 0;
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        float act = F16ToFloat(result[i]);
        double err = std::abs(static_cast<double>(act) - expected[i]);
        double tol = atol + rtol * std::abs(expected[i]);
        if (err > maxErr) { maxErr = err; maxIdx = i; }
        if (err > tol) ok = false;
    }
    if (ok) {
        LOG_PRINT("  [PASS]  max_error=%.6e at idx=%ld\n", maxErr, maxIdx);
        gPassed++;
    } else {
        float act = F16ToFloat(result[maxIdx]);
        LOG_PRINT("  [FAIL]  max_error=%.6e at idx=%ld (exp=%.6e act=%.6e)\n",
                  maxErr, maxIdx, expected[maxIdx], static_cast<double>(act));
        gFailed++;
    }
}

void CheckInt32(const std::vector<int32_t>& result, const std::vector<int64_t>& expected, const char* name)
{
    auto n = result.size();
    bool ok = true;
    int64_t badIdx = -1;
    for (size_t i = 0; i < n; i++) {
        if (result[i] != expected[i]) { ok = false; badIdx = i; break; }
    }
    if (ok) {
        LOG_PRINT("  [PASS]\n");
        gPassed++;
    } else {
        LOG_PRINT("  [FAIL]  idx=%ld expected=%ld actual=%d\n",
                  badIdx, expected[badIdx], result[badIdx]);
        gFailed++;
    }
}

// ========== ACLNN Cumsum wrapper ==========

int RunCumsumFloat32(const char* name, const std::vector<float>& input,
                      const std::vector<int64_t>& shape, int64_t dim, aclrtStream stream,
                      double atol, double rtol)
{
    LOG_PRINT("  %s ... ", name);
    int64_t n = GetShapeSize(shape);
    if (n == 0) {
        LOG_PRINT("  [SKIP] empty tensor\n");
        return 0;
    }

    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* selfT = nullptr; aclTensor* outT = nullptr;
    std::vector<float> initOut(n, 0.0f);

    int ret = CreateAclTensor(input, shape, &selfDev, ACL_FLOAT, &selfT);
    if (ret != 0) { gFailed++; return ret; }
    ret = CreateAclTensor(initOut, shape, &outDev, ACL_FLOAT, &outT);
    if (ret != 0) {
        aclDestroyTensor(selfT); aclrtFree(selfDev);
        gFailed++; return ret;
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exe = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(selfT, dim, ACL_FLOAT, outT, &wsSize, &exe);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWS failed: %d\n", ret);
        gFailed++;
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] alloc ws failed: %d\n", ret);
            gFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(outT);
            aclrtFree(selfDev); aclrtFree(outDev);
            return ret;
        }
    }

    ret = aclnnCumsum(ws, wsSize, exe, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnCumsum failed: %d\n", ret);
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    std::vector<float> result(n, 0.0f);
    aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<double> expected;
    CpuCumsumFloat32(input, dim, shape, &expected);
    CheckFloat32(result, expected, atol, rtol, name);

    if (wsSize > 0) aclrtFree(ws);
    aclDestroyTensor(selfT); aclDestroyTensor(outT);
    aclrtFree(selfDev); aclrtFree(outDev);
    return 0;
}

int RunCumsumFloat16(const char* name, const std::vector<uint16_t>& input,
                      const std::vector<int64_t>& shape, int64_t dim, aclrtStream stream,
                      double atol, double rtol)
{
    LOG_PRINT("  %s ... ", name);
    int64_t n = GetShapeSize(shape);
    if (n == 0) {
        LOG_PRINT("  [SKIP] empty tensor\n");
        return 0;
    }

    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* selfT = nullptr; aclTensor* outT = nullptr;
    std::vector<uint16_t> initOut(n, 0);

    int ret = CreateAclTensor(input, shape, &selfDev, ACL_FLOAT16, &selfT);
    if (ret != 0) { gFailed++; return ret; }
    ret = CreateAclTensor(initOut, shape, &outDev, ACL_FLOAT16, &outT);
    if (ret != 0) {
        aclDestroyTensor(selfT); aclrtFree(selfDev);
        gFailed++; return ret;
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exe = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(selfT, dim, ACL_FLOAT16, outT, &wsSize, &exe);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWS failed: %d\n", ret);
        gFailed++;
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] alloc ws failed: %d\n", ret);
            gFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(outT);
            aclrtFree(selfDev); aclrtFree(outDev);
            return ret;
        }
    }

    ret = aclnnCumsum(ws, wsSize, exe, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnCumsum failed: %d\n", ret);
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    std::vector<uint16_t> result(n, 0);
    aclrtMemcpy(result.data(), n * sizeof(uint16_t), outDev, n * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<double> expected;
    CpuCumsumFloat16(input, dim, shape, &expected);
    CheckFloat16(result, expected, atol, rtol, name);

    if (wsSize > 0) aclrtFree(ws);
    aclDestroyTensor(selfT); aclDestroyTensor(outT);
    aclrtFree(selfDev); aclrtFree(outDev);
    return 0;
}

int RunCumsumInt32(const char* name, const std::vector<int32_t>& input,
                    const std::vector<int64_t>& shape, int64_t dim, aclrtStream stream)
{
    LOG_PRINT("  %s ... ", name);
    int64_t n = GetShapeSize(shape);
    if (n == 0) {
        LOG_PRINT("  [SKIP] empty tensor\n");
        return 0;
    }

    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* selfT = nullptr; aclTensor* outT = nullptr;
    std::vector<int32_t> initOut(n, 0);

    int ret = CreateAclTensor(input, shape, &selfDev, ACL_INT32, &selfT);
    if (ret != 0) { gFailed++; return ret; }
    ret = CreateAclTensor(initOut, shape, &outDev, ACL_INT32, &outT);
    if (ret != 0) {
        aclDestroyTensor(selfT); aclrtFree(selfDev);
        gFailed++; return ret;
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exe = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(selfT, dim, ACL_INT32, outT, &wsSize, &exe);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWS failed: %d\n", ret);
        gFailed++;
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] alloc ws failed: %d\n", ret);
            gFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(outT);
            aclrtFree(selfDev); aclrtFree(outDev);
            return ret;
        }
    }

    ret = aclnnCumsum(ws, wsSize, exe, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnCumsum failed: %d\n", ret);
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    std::vector<int32_t> result(n, 0);
    aclrtMemcpy(result.data(), n * sizeof(int32_t), outDev, n * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<int64_t> expected;
    CpuCumsumInt32(input, dim, shape, &expected);
    CheckInt32(result, expected, name);

    if (wsSize > 0) aclrtFree(ws);
    aclDestroyTensor(selfT); aclDestroyTensor(outT);
    aclrtFree(selfDev); aclrtFree(outDev);
    return 0;
}

int RunCumsumV2Float32(const char* name, const std::vector<float>& input,
                         const std::vector<int64_t>& shape, int64_t dim,
                         bool exclusive, bool reverse, aclrtStream stream,
                         double atol, double rtol)
{
    LOG_PRINT("  %s ... ", name);
    int64_t n = GetShapeSize(shape);
    if (n == 0) {
        LOG_PRINT("  [SKIP] empty tensor\n");
        return 0;
    }

    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* selfT = nullptr; aclTensor* outT = nullptr;
    std::vector<float> initOut(n, 0.0f);

    int ret = CreateAclTensor(input, shape, &selfDev, ACL_FLOAT, &selfT);
    if (ret != 0) { gFailed++; return ret; }
    ret = CreateAclTensor(initOut, shape, &outDev, ACL_FLOAT, &outT);
    if (ret != 0) {
        aclDestroyTensor(selfT); aclrtFree(selfDev);
        gFailed++; return ret;
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exe = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(selfT, dim, exclusive, reverse, outT, &wsSize, &exe);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWS failed: %d\n", ret);
        gFailed++;
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] alloc ws failed: %d\n", ret);
            gFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(outT);
            aclrtFree(selfDev); aclrtFree(outDev);
            return ret;
        }
    }

    ret = aclnnCumsumV2(ws, wsSize, exe, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnCumsumV2 failed: %d\n", ret);
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    std::vector<float> result(n, 0.0f);
    aclrtMemcpy(result.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<double> expected;
    if (exclusive && reverse) {
        CpuCumsumExclRevFloat32(input, dim, shape, &expected);
    } else if (exclusive) {
        CpuCumsumExclusiveFloat32(input, dim, shape, &expected);
    } else if (reverse) {
        CpuCumsumReverseFloat32(input, dim, shape, &expected);
    } else {
        CpuCumsumFloat32(input, dim, shape, &expected);
    }
    CheckFloat32(result, expected, atol, rtol, name);

    if (wsSize > 0) aclrtFree(ws);
    aclDestroyTensor(selfT); aclDestroyTensor(outT);
    aclrtFree(selfDev); aclrtFree(outDev);
    return 0;
}

int RunCumsumV2Float16(const char* name, const std::vector<uint16_t>& input,
                         const std::vector<int64_t>& shape, int64_t dim,
                         bool exclusive, bool reverse, aclrtStream stream,
                         double atol, double rtol)
{
    LOG_PRINT("  %s ... ", name);
    int64_t n = GetShapeSize(shape);
    if (n == 0) {
        LOG_PRINT("  [SKIP] empty tensor\n");
        return 0;
    }

    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* selfT = nullptr; aclTensor* outT = nullptr;
    std::vector<uint16_t> initOut(n, 0);

    int ret = CreateAclTensor(input, shape, &selfDev, ACL_FLOAT16, &selfT);
    if (ret != 0) { gFailed++; return ret; }
    ret = CreateAclTensor(initOut, shape, &outDev, ACL_FLOAT16, &outT);
    if (ret != 0) {
        aclDestroyTensor(selfT); aclrtFree(selfDev);
        gFailed++; return ret;
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exe = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(selfT, dim, exclusive, reverse, outT, &wsSize, &exe);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWS failed: %d\n", ret);
        gFailed++;
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] alloc ws failed: %d\n", ret);
            gFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(outT);
            aclrtFree(selfDev); aclrtFree(outDev);
            return ret;
        }
    }

    ret = aclnnCumsumV2(ws, wsSize, exe, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnCumsumV2 failed: %d\n", ret);
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    std::vector<uint16_t> result(n, 0);
    aclrtMemcpy(result.data(), n * sizeof(uint16_t), outDev, n * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<double> expected;
    if (exclusive && reverse) {
        CpuCumsumExclRevFloat16(input, dim, shape, &expected);
    } else if (exclusive) {
        CpuCumsumExclusiveFloat16(input, dim, shape, &expected);
    } else if (reverse) {
        CpuCumsumReverseFloat16(input, dim, shape, &expected);
    } else {
        CpuCumsumFloat16(input, dim, shape, &expected);
    }
    CheckFloat16(result, expected, atol, rtol, name);

    if (wsSize > 0) aclrtFree(ws);
    aclDestroyTensor(selfT); aclDestroyTensor(outT);
    aclrtFree(selfDev); aclrtFree(outDev);
    return 0;
}

// ========== Test suites ==========

int RunCumsumV2Int32(const char* name, const std::vector<int32_t>& input,
                     const std::vector<int64_t>& shape, int64_t dim,
                     bool exclusive, bool reverse, aclrtStream stream)
{
    LOG_PRINT("  %s ... ", name);
    int64_t n = GetShapeSize(shape);
    if (n == 0) {
        LOG_PRINT("  [SKIP] empty tensor\n");
        return 0;
    }

    void* selfDev = nullptr; void* outDev = nullptr;
    aclTensor* selfT = nullptr; aclTensor* outT = nullptr;
    std::vector<int32_t> initOut(n, 0);

    int ret = CreateAclTensor(input, shape, &selfDev, ACL_INT32, &selfT);
    if (ret != 0) { gFailed++; return ret; }
    ret = CreateAclTensor(initOut, shape, &outDev, ACL_INT32, &outT);
    if (ret != 0) {
        aclDestroyTensor(selfT); aclrtFree(selfDev);
        gFailed++; return ret;
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exe = nullptr;
    ret = aclnnCumsumV2GetWorkspaceSize(selfT, dim, exclusive, reverse, outT, &wsSize, &exe);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWS failed: %d\n", ret);
        gFailed++;
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] alloc ws failed: %d\n", ret);
            gFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(outT);
            aclrtFree(selfDev); aclrtFree(outDev);
            return ret;
        }
    }

    ret = aclnnCumsumV2(ws, wsSize, exe, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] aclnnCumsumV2 failed: %d\n", ret);
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        gFailed++;
        if (wsSize > 0) aclrtFree(ws);
        aclDestroyTensor(selfT); aclDestroyTensor(outT);
        aclrtFree(selfDev); aclrtFree(outDev);
        return ret;
    }

    std::vector<int32_t> result(n, 0);
    aclrtMemcpy(result.data(), n * sizeof(int32_t), outDev, n * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<int64_t> expected;
    if (exclusive && reverse) {
        CpuCumsumExclusiveReverseInt32(input, dim, shape, &expected);
    } else if (exclusive) {
        CpuCumsumExclusiveInt32(input, dim, shape, &expected);
    } else if (reverse) {
        CpuCumsumReverseInt32(input, dim, shape, &expected);
    } else {
        CpuCumsumInt32(input, dim, shape, &expected);
    }
    CheckInt32(result, expected, name);

    if (wsSize > 0) aclrtFree(ws);
    aclDestroyTensor(selfT); aclDestroyTensor(outT);
    aclrtFree(selfDev); aclrtFree(outDev);
    return 0;
}

void TestSuite_Float32_Basic(aclrtStream stream)
{
    LOG_PRINT("\n=== Float32 Basic ===\n");
    std::vector<float> d = {1,2,3,4,5,6,7,8};
    RunCumsumFloat32("float32 {2,4} dim=0", d, {2,4}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,4} dim=1", d, {2,4}, 1, stream, 1e-5, 1e-5);
}

void TestSuite_Float16_Basic(aclrtStream stream)
{
    LOG_PRINT("\n=== Float16 Basic ===\n");
    std::vector<uint16_t> d;
    for (int i = 1; i <= 8; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i), &bits);
        d.push_back(bits);
    }
    RunCumsumFloat16("float16 {2,4} dim=0", d, {2,4}, 0, stream, 1e-3, 1e-3);
    RunCumsumFloat16("float16 {2,4} dim=1", d, {2,4}, 1, stream, 1e-3, 1e-3);
}

void TestSuite_Int32_Basic(aclrtStream stream)
{
    LOG_PRINT("\n=== Int32 Basic ===\n");
    std::vector<int32_t> d = {1,2,3,4,5,6,7,8};
    RunCumsumInt32("int32 {2,4} dim=0", d, {2,4}, 0, stream);
    RunCumsumInt32("int32 {2,4} dim=1", d, {2,4}, 1, stream);
}

void TestSuite_SequenceLengths(aclrtStream stream)
{
    LOG_PRINT("\n=== Sequence Lengths ===\n");
    std::vector<float> d10(10);
    for (int i = 0; i < 10; i++) d10[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=10 dim=0", d10, {10}, 0, stream, 1e-5, 1e-5);

    std::vector<float> d50(50);
    for (int i = 0; i < 50; i++) d50[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=50 dim=0", d50, {50}, 0, stream, 1e-5, 1e-5);

    std::vector<float> d100(100);
    for (int i = 0; i < 100; i++) d100[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=100 dim=0", d100, {100}, 0, stream, 1e-5, 1e-5);

    std::vector<float> d1000(1000);
    for (int i = 0; i < 1000; i++) d1000[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=1000 dim=0", d1000, {1000}, 0, stream, 1e-4, 1e-5);

    std::vector<float> d5000(5000);
    for (int i = 0; i < 5000; i++) d5000[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=5000 dim=0", d5000, {5000}, 0, stream, 1e-3, 1e-5);

    std::vector<float> d10000(10000);
    for (int i = 0; i < 10000; i++) d10000[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=10000 dim=0", d10000, {10000}, 0, stream, 1e-3, 1e-5);
}

void TestSuite_NegativeAndMixed(aclrtStream stream)
{
    LOG_PRINT("\n=== Negative & Mixed ===\n");
    std::vector<float> neg(10);
    for (int i = 0; i < 10; i++) neg[i] = static_cast<float>(-i - 1);
    RunCumsumFloat32("float32 negative values", neg, {10}, 0, stream, 1e-5, 1e-5);

    std::vector<float> mixed(10);
    for (int i = 0; i < 10; i++) mixed[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : static_cast<float>(-(i + 1));
    RunCumsumFloat32("float32 pos/neg mixed", mixed, {10}, 0, stream, 1e-5, 1e-5);

    std::vector<float> zeros(10, 0.0f);
    RunCumsumFloat32("float32 all zeros", zeros, {10}, 0, stream, 1e-5, 1e-5);

    std::vector<float> ones(10000, 1.0f);
    RunCumsumFloat32("float32 1.0 x10000 (error accum)", ones, {10000}, 0, stream, 1e-3, 1e-5);

    std::vector<float> mixedMag(20);
    for (int i = 0; i < 20; i++) mixedMag[i] = (i % 2 == 0) ? 1e8f : 1e-6f;
    RunCumsumFloat32("float32 [1e8,1e-6]x10", mixedMag, {20}, 0, stream, 1e-2, 1e-3);

    std::vector<float> d01(100, 0.1f);
    RunCumsumFloat32("float32 [0.1]x100", d01, {100}, 0, stream, 1e-3, 1e-3);
}

void TestSuite_MultiDim(aclrtStream stream)
{
    LOG_PRINT("\n=== Multi-Dimension Tensors ===\n");
    std::vector<float> d24(24);
    for (int i = 0; i < 24; i++) d24[i] = static_cast<float>(i + 1);

    RunCumsumFloat32("float32 {2,3,4} dim=0", d24, {2,3,4}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,3,4} dim=1", d24, {2,3,4}, 1, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,3,4} dim=2", d24, {2,3,4}, 2, stream, 1e-5, 1e-5);

    std::vector<float> d16(16);
    for (int i = 0; i < 16; i++) d16[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {2,2,2,2} dim=1", d16, {2,2,2,2}, 1, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,2,2,2} dim=3", d16, {2,2,2,2}, 3, stream, 1e-5, 1e-5);
}

void TestSuite_TilingShapes(aclrtStream stream)
{
    LOG_PRINT("\n=== Tiling Shape Coverage ===\n");
    std::vector<float> d128(128);
    for (int i = 0; i < 128; i++) d128[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=128 dim=0", d128, {128}, 0, stream, 1e-5, 1e-5);

    std::vector<float> d256(256);
    for (int i = 0; i < 256; i++) d256[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=256 dim=0", d256, {256}, 0, stream, 1e-5, 1e-5);

    std::vector<float> d512(512);
    for (int i = 0; i < 512; i++) d512[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=512 dim=0", d512, {512}, 0, stream, 1e-5, 1e-5);

    std::vector<float> d2048(2048);
    for (int i = 0; i < 2048; i++) d2048[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=2048 dim=0", d2048, {2048}, 0, stream, 1e-4, 1e-5);

    std::vector<float> d64_16(64*16);
    for (int i = 0; i < 64*16; i++) d64_16[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {64,16} dim=0", d64_16, {64,16}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {64,16} dim=1", d64_16, {64,16}, 1, stream, 1e-5, 1e-5);

    std::vector<float> d16_64(16*64);
    for (int i = 0; i < 16*64; i++) d16_64[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {16,64} dim=0", d16_64, {16,64}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {16,64} dim=1", d16_64, {16,64}, 1, stream, 1e-5, 1e-5);

    std::vector<float> d4_128(4*128);
    for (int i = 0; i < 4*128; i++) d4_128[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {4,128} dim=0", d4_128, {4,128}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {4,128} dim=1", d4_128, {4,128}, 1, stream, 1e-5, 1e-5);

    std::vector<float> d1_1000(1000);
    for (int i = 0; i < 1000; i++) d1_1000[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {1,1000} dim=0", d1_1000, {1,1000}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {1,1000} dim=1", d1_1000, {1,1000}, 1, stream, 1e-5, 1e-5);

    std::vector<float> d1000_1(1000);
    for (int i = 0; i < 1000; i++) d1000_1[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {1000,1} dim=0", d1000_1, {1000,1}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {1000,1} dim=1", d1000_1, {1000,1}, 1, stream, 1e-5, 1e-5);

    std::vector<float> d2_512(2*512);
    for (int i = 0; i < 2*512; i++) d2_512[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {2,512} dim=0", d2_512, {2,512}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,512} dim=1", d2_512, {2,512}, 1, stream, 1e-5, 1e-5);

    std::vector<float> d256_2(256*2);
    for (int i = 0; i < 256*2; i++) d256_2[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {256,2} dim=0", d256_2, {256,2}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {256,2} dim=1", d256_2, {256,2}, 1, stream, 1e-5, 1e-5);

    std::vector<float> d8_8_8(8*8*8);
    for (int i = 0; i < 8*8*8; i++) d8_8_8[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {8,8,8} dim=0", d8_8_8, {8,8,8}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {8,8,8} dim=2", d8_8_8, {8,8,8}, 2, stream, 1e-5, 1e-5);
}

void TestSuite_Int32_Shapes(aclrtStream stream)
{
    LOG_PRINT("\n=== Int32 Various Shapes ===\n");
    std::vector<int32_t> d128(128);
    for (int i = 0; i < 128; i++) d128[i] = i + 1;
    RunCumsumInt32("int32 len=128 dim=0", d128, {128}, 0, stream);
    RunCumsumInt32("int32 len=256 dim=0", d128, {256}, 0, stream);

    std::vector<int32_t> d500(500);
    for (int i = 0; i < 500; i++) d500[i] = i + 1;
    RunCumsumInt32("int32 len=500 dim=0", d500, {500}, 0, stream);

    std::vector<int32_t> d4_64(4*64);
    for (int i = 0; i < 4*64; i++) d4_64[i] = i + 1;
    RunCumsumInt32("int32 {4,64} dim=0", d4_64, {4,64}, 0, stream);
    RunCumsumInt32("int32 {4,64} dim=1", d4_64, {4,64}, 1, stream);

    std::vector<int32_t> d64_4(64*4);
    for (int i = 0; i < 64*4; i++) d64_4[i] = i + 1;
    RunCumsumInt32("int32 {64,4} dim=0", d64_4, {64,4}, 0, stream);
    RunCumsumInt32("int32 {64,4} dim=1", d64_4, {64,4}, 1, stream);

    std::vector<int32_t> d2_2_64(2*2*64);
    for (int i = 0; i < 2*2*64; i++) d2_2_64[i] = i + 1;
    RunCumsumInt32("int32 {2,2,64} dim=0", d2_2_64, {2,2,64}, 0, stream);
    RunCumsumInt32("int32 {2,2,64} dim=2", d2_2_64, {2,2,64}, 2, stream);
}

void TestSuite_V2_Basic(aclrtStream stream)
{
    LOG_PRINT("\n=== CumsumV2 Basic ===\n");
    std::vector<float> d = {1,2,3,4,5,6,7,8};
    RunCumsumV2Float32("V2 {2,4} excl=T rev=F dim=0", d, {2,4}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} excl=F rev=T dim=0", d, {2,4}, 0, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} excl=T rev=T dim=0", d, {2,4}, 0, true, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} excl=F rev=F dim=1", d, {2,4}, 1, false, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} excl=T rev=F dim=1", d, {2,4}, 1, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} excl=F rev=T dim=1", d, {2,4}, 1, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} excl=T rev=T dim=1", d, {2,4}, 1, true, true, stream, 1e-5, 1e-5);
}

void TestSuite_V2_LongSequence(aclrtStream stream)
{
    LOG_PRINT("\n=== CumsumV2 Long Sequences ===\n");
    std::vector<float> d1000(1000);
    for (int i = 0; i < 1000; i++) d1000[i] = static_cast<float>(i + 1);
    RunCumsumV2Float32("V2 len=1000 excl=T dim=0", d1000, {1000}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 len=1000 rev=T dim=0", d1000, {1000}, 0, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 len=1000 excl=T rev=T dim=0", d1000, {1000}, 0, true, true, stream, 1e-5, 1e-5);

    std::vector<uint16_t> d256_f16;
    for (int i = 0; i < 256; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i + 1), &bits);
        d256_f16.push_back(bits);
    }
    RunCumsumV2Float16("V2 float16 len=256 excl=T", d256_f16, {256}, 0, true, false, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 float16 len=256 rev=T", d256_f16, {256}, 0, false, true, stream, 1e-3, 1e-3);
}

void TestSuite_V2_Shapes(aclrtStream stream)
{
    LOG_PRINT("\n=== CumsumV2 Various Shapes ===\n");
    std::vector<float> d4_64(4*64);
    for (int i = 0; i < 4*64; i++) d4_64[i] = static_cast<float>(i + 1);
    RunCumsumV2Float32("V2 {4,64} excl=T dim=0", d4_64, {4,64}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {4,64} excl=T dim=1", d4_64, {4,64}, 1, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {4,64} rev=T dim=0", d4_64, {4,64}, 0, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {4,64} rev=T dim=1", d4_64, {4,64}, 1, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {4,64} excl=T rev=T dim=0", d4_64, {4,64}, 0, true, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {4,64} excl=T rev=T dim=1", d4_64, {4,64}, 1, true, true, stream, 1e-5, 1e-5);

    std::vector<float> d2_128(2*128);
    for (int i = 0; i < 2*128; i++) d2_128[i] = static_cast<float>(i + 1);
    RunCumsumV2Float32("V2 {2,128} excl=T dim=0", d2_128, {2,128}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,128} rev=T dim=0", d2_128, {2,128}, 0, false, true, stream, 1e-5, 1e-5);
}

void TestSuite_NegativeDim(aclrtStream stream)
{
    LOG_PRINT("\n=== Negative Dimension ===\n");
    std::vector<float> d = {1,2,3,4,5,6,7,8};
    RunCumsumFloat32("float32 {2,4} dim=-2", d, {2,4}, -2, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,4} dim=-1", d, {2,4}, -1, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} dim=-2 excl=T", d, {2,4}, -2, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,4} dim=-1 rev=T", d, {2,4}, -1, false, true, stream, 1e-5, 1e-5);
}

void TestSuite_DtypeConversion(aclrtStream stream)
{
    LOG_PRINT("\n=== Dtype Conversion (via Cast) ===\n");
    std::vector<float> d = {1,2,3,4,5,6,7,8};
    // Float32 in, Float16 out - handled by aclnnCumsum Cast path
    std::vector<uint16_t> dF16;
    for (int i = 0; i < 8; i++) {
        uint16_t bits; FloatToF16(d[i], &bits);
        dF16.push_back(bits);
    }
    RunCumsumFloat16("float32->float16 {2,4} dim=0", dF16, {2,4}, 0, stream, 1e-3, 1e-3);
    RunCumsumFloat16("float32->float16 {2,4} dim=1", dF16, {2,4}, 1, stream, 1e-3, 1e-3);
}

void TestSuite_EmptyTensor(aclrtStream stream)
{
    LOG_PRINT("\n=== Empty Tensor ===\n");
    std::vector<float> emptyF;
    std::vector<uint16_t> emptyF16;
    std::vector<int32_t> emptyI;

    RunCumsumFloat32("float32 empty dim=0", emptyF, {0}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat16("float16 empty dim=0", emptyF16, {0}, 0, stream, 1e-3, 1e-3);
    RunCumsumInt32("int32 empty dim=0", emptyI, {0}, 0, stream);

    std::vector<float> shape20;
    RunCumsumFloat32("float32 {2,0} dim=0", shape20, {2,0}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,0} dim=1", shape20, {2,0}, 1, stream, 1e-5, 1e-5);

    RunCumsumV2Float32("V2 float32 empty excl=F", emptyF, {0}, 0, false, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 float32 empty excl=T", emptyF, {0}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 float32 empty rev=T", emptyF, {0}, 0, false, true, stream, 1e-5, 1e-5);
}

void TestSuite_Overflow(aclrtStream stream)
{
    LOG_PRINT("\n=== Integer Overflow ===\n");
    std::vector<int32_t> d10(10);
    d10[0] = INT32_MAX - 9;
    for (int i = 1; i < 10; i++) d10[i] = 1;
    RunCumsumInt32("int32 near INT_MAX", d10, {10}, 0, stream);
}

void TestSuite_EdgePrecision(aclrtStream stream)
{
    LOG_PRINT("\n=== Edge Precision Cases ===\n");
    std::vector<float> d5(5, 0.0f);
    d5[2] = 1.0f;
    RunCumsumFloat32("float32 single non-zero at middle", d5, {5}, 0, stream, 1e-5, 1e-5);

    std::vector<float> d4(4);
    d4[0] = 1e10f; d4[1] = -1e10f; d4[2] = 1e10f; d4[3] = -1e10f;
    RunCumsumFloat32("float32 cancel-out [1e10,-1e10]x2", d4, {4}, 0, stream, 1e-3, 1e-3);

    std::vector<float> d10000_tens(10000, 10.0f);
    RunCumsumFloat32("float32 10.0 x10000", d10000_tens, {10000}, 0, stream, 1e-3, 1e-5);

    std::vector<uint16_t> d10000_f16(10000);
    for (int i = 0; i < 10000; i++) {
        uint16_t bits; FloatToF16(1.0f, &bits);
        d10000_f16[i] = bits;
    }
    RunCumsumFloat16("float16 1.0 x10000 (fast accum)", d10000_f16, {10000}, 0, stream, 1e-2, 1e-3);
}

// ========== Additional suites for tiling / branch coverage ==========

void TestSuite_CubePath(aclrtStream stream)
{
    LOG_PRINT("\n=== CUBE Path (batch>=12800, dim>=512) ===\n");
    // CUMSUM_CUBE_MIN_SUPPORT_BATCH = 12800, CUMSUM_CUBE_MIN_SUPPORT_DIM = 512
    std::vector<float> d_cube1(12800 * 512);
    for (int i = 0; i < 12800 * 512; i++) d_cube1[i] = static_cast<float>(i % 10 + 1);
    RunCumsumFloat32("CUBE {12800,512} dim=1", d_cube1, {12800, 512}, 1, stream, 1e-4, 1e-5);

    std::vector<uint16_t> d_cube_f16(12800 * 256);
    for (int i = 0; i < 12800 * 256; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i % 10 + 1), &bits);
        d_cube_f16[i] = bits;
    }
    RunCumsumFloat16("CUBE float16 {12800,256} dim=1", d_cube_f16, {12800, 256}, 1, stream, 1e-2, 1e-3);

    std::vector<float> d_cube_nb(12799 * 512);
    for (int i = 0; i < 12799 * 512; i++) d_cube_nb[i] = static_cast<float>(i % 5 + 1);
    RunCumsumFloat32("near-CUBE {12799,512} dim=1", d_cube_nb, {12799, 512}, 1, stream, 1e-4, 1e-5);

    std::vector<float> d_cube_nc(12800 * 511);
    for (int i = 0; i < 12800 * 511; i++) d_cube_nc[i] = static_cast<float>(i % 5 + 1);
    RunCumsumFloat32("near-CUBE {12800,511} dim=1", d_cube_nc, {12800, 511}, 1, stream, 1e-4, 1e-5);
}

void TestSuite_TilingBorrowAxis(aclrtStream stream)
{
    LOG_PRINT("\n=== Tiling Borrow Axis Branches ===\n");
    // borrow N: M < coreNum/2, N can be borrowed to split cores
    // {4, 256}: M=4 small, N=256 large -> borrows N axis for core splitting
    std::vector<float> d_bn1(4 * 256);
    for (int i = 0; i < 4 * 256; i++) d_bn1[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{4,256} dim=0 (borrow N)", d_bn1, {4, 256}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("{4,256} dim=1 (borrow N)", d_bn1, {4, 256}, 1, stream, 1e-5, 1e-5);

    // {2, 512}: M=2 very small, N=512 -> borrows heavily
    std::vector<float> d_bn2(2 * 512);
    for (int i = 0; i < 2 * 512; i++) d_bn2[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{2,512} dim=0 (heavy borrow N)", d_bn2, {2, 512}, 0, stream, 1e-5, 1e-5);

    // {1, 256}: M=1 minimal, borrows N maximally
    std::vector<float> d_bn3(1 * 256);
    for (int i = 0; i < 1 * 256; i++) d_bn3[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{1,256} dim=0 (M=1 borrow N)", d_bn3, {1, 256}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("{1,256} dim=1 (M=1 full)", d_bn3, {1, 256}, 1, stream, 1e-5, 1e-5);

    // {8, 128}: M=8 moderate
    std::vector<float> d_bn4(8 * 128);
    for (int i = 0; i < 8 * 128; i++) d_bn4[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{8,128} dim=0", d_bn4, {8, 128}, 0, stream, 1e-5, 1e-5);

    // {16, 64}: M=16 > coreNum/2 -> no borrow
    std::vector<float> d_bn5(16 * 64);
    for (int i = 0; i < 16 * 64; i++) d_bn5[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{16,64} dim=0 (no borrow)", d_bn5, {16, 64}, 0, stream, 1e-5, 1e-5);

    // 3D borrow scenarios
    std::vector<float> d_b3d(4 * 8 * 128);
    for (int i = 0; i < 4 * 8 * 128; i++) d_b3d[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{4,8,128} dim=1", d_b3d, {4, 8, 128}, 1, stream, 1e-5, 1e-5);
    RunCumsumFloat32("{4,8,128} dim=2", d_b3d, {4, 8, 128}, 2, stream, 1e-5, 1e-5);
    RunCumsumFloat32("{4,8,128} dim=0", d_b3d, {4, 8, 128}, 0, stream, 1e-5, 1e-5);

    // {2, 2, 256}: very small M, borrows
    std::vector<float> d_b3d2(2 * 2 * 256);
    for (int i = 0; i < 2 * 2 * 256; i++) d_b3d2[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{2,2,256} dim=2 (borrow)", d_b3d2, {2, 2, 256}, 2, stream, 1e-5, 1e-5);
}

void TestSuite_TilingTWOWAY(aclrtStream stream)
{
    LOG_PRINT("\n=== Tiling TWOWAY Branches ===\n");
    // TWOWAY (SklanskyPattern::SS_TWOWAY): alignN <= vRegSize/4, lenR > foldLen
    // Need lenN * dtSizeCast < clSize (goes to RNGreaterCl)
    // and lenR large enough for fold > 1
    // {1, 256}: M=1, N=1 (clSize > 256*4), R=256 -> lenR*clSize > ubSize -> NGreaterClRNotFullLoad -> borrowM path -> TWOWAY possible
    std::vector<float> d_tw1(1 * 256);
    for (int i = 0; i < 1 * 256; i++) d_tw1[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{1,256} dim=0 TWOWAY candidate", d_tw1, {1, 256}, 0, stream, 1e-5, 1e-5);

    // {1, 512}: even larger R
    std::vector<float> d_tw2(1 * 512);
    for (int i = 0; i < 1 * 512; i++) d_tw2[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{1,512} dim=0", d_tw2, {1, 512}, 0, stream, 1e-5, 1e-5);

    // V2 with reverse triggers TWOWAY
    RunCumsumV2Float32("{1,256} V2 rev=T", d_tw1, {1, 256}, 0, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("{1,256} V2 excl=T rev=T", d_tw1, {1, 256}, 0, true, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("{1,512} V2 excl=T rev=T", d_tw2, {1, 512}, 0, true, true, stream, 1e-5, 1e-5);

    // {1, 1024}: very long single row
    std::vector<float> d_tw3(1 * 1024);
    for (int i = 0; i < 1 * 1024; i++) d_tw3[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{1,1024} dim=0", d_tw3, {1, 1024}, 0, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("{1,1024} V2 excl=T", d_tw3, {1, 1024}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("{1,1024} V2 rev=T", d_tw3, {1, 1024}, 0, false, true, stream, 1e-5, 1e-5);
}

void TestSuite_IntTilingAll(aclrtStream stream)
{
    LOG_PRINT("\n=== Int Tiling (all shapes) ===\n");
    std::vector<int32_t> di32_256(256);
    for (int i = 0; i < 256; i++) di32_256[i] = i + 1;
    RunCumsumInt32("int32 len=256", di32_256, {256}, 0, stream);
    RunCumsumInt32("int32 len=256 dim=0", di32_256, {256}, 0, stream);

    // Various 2D shapes for int tiling
    std::vector<int32_t> di32_4x64(4*64);
    for (int i = 0; i < 4*64; i++) di32_4x64[i] = i + 1;
    RunCumsumInt32("int32 {4,64} dim=0", di32_4x64, {4,64}, 0, stream);
    RunCumsumInt32("int32 {4,64} dim=1", di32_4x64, {4,64}, 1, stream);

    std::vector<int32_t> di32_64x4(64*4);
    for (int i = 0; i < 64*4; i++) di32_64x4[i] = i + 1;
    RunCumsumInt32("int32 {64,4} dim=0", di32_64x4, {64,4}, 0, stream);
    RunCumsumInt32("int32 {64,4} dim=1", di32_64x4, {64,4}, 1, stream);

    std::vector<int32_t> di32_2x2x64(2*2*64);
    for (int i = 0; i < 2*2*64; i++) di32_2x2x64[i] = i + 1;
    RunCumsumInt32("int32 {2,2,64} dim=0", di32_2x2x64, {2,2,64}, 0, stream);
    RunCumsumInt32("int32 {2,2,64} dim=2", di32_2x2x64, {2,2,64}, 2, stream);

    std::vector<int32_t> di32_8x8x4(8*8*4);
    for (int i = 0; i < 8*8*4; i++) di32_8x8x4[i] = i + 1;
    RunCumsumInt32("int32 {8,8,4} dim=0", di32_8x8x4, {8,8,4}, 0, stream);
    RunCumsumInt32("int32 {8,8,4} dim=2", di32_8x8x4, {8,8,4}, 2, stream);

    // INT32 with various shapes for int tiling
    std::vector<int32_t> di32_500(500);
    for (int i = 0; i < 500; i++) di32_500[i] = i + 1;
    RunCumsumInt32("int32 len=500", di32_500, {500}, 0, stream);

    std::vector<int32_t> di32_10x50(10*50);
    for (int i = 0; i < 10*50; i++) di32_10x50[i] = i + 1;
    RunCumsumInt32("int32 {10,50} dim=0", di32_10x50, {10,50}, 0, stream);
    RunCumsumInt32("int32 {10,50} dim=1", di32_10x50, {10,50}, 1, stream);

    std::vector<int32_t> di32_50x10(50*10);
    for (int i = 0; i < 50*10; i++) di32_50x10[i] = i + 1;
    RunCumsumInt32("int32 {50,10} dim=0", di32_50x10, {50,10}, 0, stream);
    RunCumsumInt32("int32 {50,10} dim=1", di32_50x10, {50,10}, 1, stream);

    std::vector<int32_t> di32_1x500(500);
    for (int i = 0; i < 500; i++) di32_1x500[i] = i + 1;
    RunCumsumInt32("int32 {1,500} dim=0", di32_1x500, {1,500}, 0, stream);
    RunCumsumInt32("int32 {1,500} dim=1", di32_1x500, {1,500}, 1, stream);

    std::vector<int32_t> di32_500x1(500);
    for (int i = 0; i < 500; i++) di32_500x1[i] = i + 1;
    RunCumsumInt32("int32 {500,1} dim=0", di32_500x1, {500,1}, 0, stream);
    RunCumsumInt32("int32 {500,1} dim=1", di32_500x1, {500,1}, 1, stream);

    // V2 with int32 (use RunCumsumV2Int32 for correct dtype)
    RunCumsumV2Int32("V2 int32 {10,50} excl=T", di32_10x50, {10,50}, 0, true, false, stream);
    RunCumsumV2Int32("V2 int32 {10,50} rev=T", di32_10x50, {10,50}, 0, false, true, stream);
    RunCumsumV2Int32("V2 int32 {10,50} excl=T rev=T", di32_10x50, {10,50}, 0, true, true, stream);

    // V2 with exclusive/reverse on 2D shapes
    std::vector<int32_t> di32_4x64_v2(4*64);
    for (int i = 0; i < 4*64; i++) di32_4x64_v2[i] = i + 1;
    RunCumsumV2Int32("V2 int32 {4,64} excl=T dim=0", di32_4x64_v2, {4,64}, 0, true, false, stream);
    RunCumsumV2Int32("V2 int32 {4,64} excl=T dim=1", di32_4x64_v2, {4,64}, 1, true, false, stream);
    RunCumsumV2Int32("V2 int32 {4,64} rev=T dim=0", di32_4x64_v2, {4,64}, 0, false, true, stream);
    RunCumsumV2Int32("V2 int32 {4,64} rev=T dim=1", di32_4x64_v2, {4,64}, 1, false, true, stream);
    RunCumsumV2Int32("V2 int32 {4,64} excl=T rev=T dim=0", di32_4x64_v2, {4,64}, 0, true, true, stream);
    RunCumsumV2Int32("V2 int32 {4,64} excl=T rev=T dim=1", di32_4x64_v2, {4,64}, 1, true, true, stream);

    // V2 with larger int32
    RunCumsumV2Int32("V2 int32 len=256 excl=T", di32_256, {256}, 0, true, false, stream);
    RunCumsumV2Int32("V2 int32 len=256 rev=T", di32_256, {256}, 0, false, true, stream);
    RunCumsumV2Int32("V2 int32 len=256 excl=T rev=T", di32_256, {256}, 0, true, true, stream);
}

void TestSuite_V2AllCombos(aclrtStream stream)
{
    LOG_PRINT("\n=== V2 All Combos + dim branching ===\n");
    // V2 on 3D/4D tensors
    std::vector<float> d3d(2*3*4);
    for (int i = 0; i < 2*3*4; i++) d3d[i] = static_cast<float>(i + 1);

    RunCumsumV2Float32("V2 {2,3,4} excl=T dim=0", d3d, {2,3,4}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} excl=T dim=1", d3d, {2,3,4}, 1, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} excl=T dim=2", d3d, {2,3,4}, 2, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} rev=T dim=0", d3d, {2,3,4}, 0, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} rev=T dim=1", d3d, {2,3,4}, 1, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} rev=T dim=2", d3d, {2,3,4}, 2, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} excl=T rev=T dim=0", d3d, {2,3,4}, 0, true, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} excl=T rev=T dim=1", d3d, {2,3,4}, 1, true, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,3,4} excl=T rev=T dim=2", d3d, {2,3,4}, 2, true, true, stream, 1e-5, 1e-5);

    // dim > INT32_MAX (triggers DT_INT64 branch in dimTensor creation)
    // This is tricky in a test - we can't easily create that big a tensor,
    // but we can test with dim=1 on small tensor to hit different dimTensor branches
    std::vector<float> d_small(10);
    for (int i = 0; i < 10; i++) d_small[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("dim=1 small tensor", d_small, {10}, 1, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 dim=1 excl=T", d_small, {10}, 1, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 dim=1 rev=T", d_small, {10}, 1, false, true, stream, 1e-5, 1e-5);

    // 2D with various dims
    std::vector<float> d2d_10x5(10*5);
    for (int i = 0; i < 10*5; i++) d2d_10x5[i] = static_cast<float>(i + 1);
    RunCumsumV2Float32("V2 {10,5} excl=T dim=0", d2d_10x5, {10,5}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {10,5} rev=T dim=0", d2d_10x5, {10,5}, 0, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {10,5} excl=T rev=T dim=0", d2d_10x5, {10,5}, 0, true, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {10,5} excl=T dim=1", d2d_10x5, {10,5}, 1, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {10,5} rev=T dim=1", d2d_10x5, {10,5}, 1, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {10,5} excl=T rev=T dim=1", d2d_10x5, {10,5}, 1, true, true, stream, 1e-5, 1e-5);

    // Float16 V2 on multi-dim
    std::vector<uint16_t> d3d_f16;
    for (int i = 0; i < 2*3*4; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i + 1), &bits);
        d3d_f16.push_back(bits);
    }
    RunCumsumV2Float16("V2 float16 {2,3,4} excl=T dim=0", d3d_f16, {2,3,4}, 0, true, false, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 float16 {2,3,4} rev=T dim=2", d3d_f16, {2,3,4}, 2, false, true, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 float16 {2,3,4} excl=T rev=T dim=1", d3d_f16, {2,3,4}, 1, true, true, stream, 1e-3, 1e-3);

    // Large int32 V2
    std::vector<int32_t> di32_256(256);
    for (int i = 0; i < 256; i++) di32_256[i] = i + 1;
    RunCumsumV2Int32("V2 int32 len=256 excl=T", di32_256, {256}, 0, true, false, stream);
    RunCumsumV2Int32("V2 int32 len=256 rev=T", di32_256, {256}, 0, false, true, stream);
    RunCumsumV2Int32("V2 int32 len=256 excl=T rev=T", di32_256, {256}, 0, true, true, stream);
}

void TestSuite_LongAndLarge(aclrtStream stream)
{
    LOG_PRINT("\n=== Long & Large Tensors ===\n");
    std::vector<float> d_20000(20000);
    for (int i = 0; i < 20000; i++) d_20000[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=20000", d_20000, {20000}, 0, stream, 1e-3, 1e-5);

    std::vector<float> d_50000(50000);
    for (int i = 0; i < 50000; i++) d_50000[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 len=50000", d_50000, {50000}, 0, stream, 1e-3, 1e-5);

    std::vector<float> d_100000(100000);
    for (int i = 0; i < 100000; i++) d_100000[i] = static_cast<float>(i % 100 + 1);
    RunCumsumFloat32("float32 len=100000", d_100000, {100000}, 0, stream, 1e-3, 1e-5);

    std::vector<float> d_1000x100(1000*100);
    for (int i = 0; i < 1000*100; i++) d_1000x100[i] = static_cast<float>(i % 50 + 1);
    RunCumsumFloat32("float32 {1000,100} dim=0", d_1000x100, {1000,100}, 0, stream, 1e-4, 1e-5);
    RunCumsumFloat32("float32 {1000,100} dim=1", d_1000x100, {1000,100}, 1, stream, 1e-4, 1e-5);

    std::vector<int32_t> di32_10k(10000);
    for (int i = 0; i < 10000; i++) di32_10k[i] = i + 1;
    RunCumsumInt32("int32 len=10000", di32_10k, {10000}, 0, stream);
    RunCumsumInt32("int32 {100,100} dim=0", di32_10k, {100,100}, 0, stream);
    RunCumsumInt32("int32 {100,100} dim=1", di32_10k, {100,100}, 1, stream);
}

// ========== Additional suites for deeper tiling/branch coverage ==========

void TestSuite_CubePathV2(aclrtStream stream)
{
    LOG_PRINT("\n=== CUBE Path V2 (trigger CUBE via V2 path) ===\n");
    std::vector<float> d_cube1(12800 * 512);
    for (int i = 0; i < 12800 * 512; i++) d_cube1[i] = static_cast<float>(i % 10 + 1);
    RunCumsumV2Float32("CUBE V2 {12800,512} dim=1", d_cube1, {12800, 512}, 1, false, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("CUBE V2 {12800,512} excl=T dim=1", d_cube1, {12800, 512}, 1, true, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("CUBE V2 {12800,512} rev=T dim=1", d_cube1, {12800, 512}, 1, false, true, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("CUBE V2 {12800,512} excl=T rev=T dim=1", d_cube1, {12800, 512}, 1, true, true, stream, 1e-4, 1e-5);

    std::vector<uint16_t> d_cube_f16(12800 * 256);
    for (int i = 0; i < 12800 * 256; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i % 10 + 1), &bits);
        d_cube_f16[i] = bits;
    }
    RunCumsumV2Float16("CUBE V2 float16 {12800,256} excl=T dim=1", d_cube_f16, {12800, 256}, 1, true, false, stream, 1e-2, 1e-3);
    RunCumsumV2Float16("CUBE V2 float16 {12800,256} rev=T dim=1", d_cube_f16, {12800, 256}, 1, false, true, stream, 1e-2, 1e-3);
}

void TestSuite_IntTilingAxisBranches(aclrtStream stream)
{
    LOG_PRINT("\n=== Int Tiling Axis Branches ===\n");
    // axis=0: rightA = N (split RA), leftA = 1
    // triggers GetInputDims axis==0u branch
    std::vector<int32_t> di_axis0(10 * 64);
    for (int i = 0; i < 10 * 64; i++) di_axis0[i] = i + 1;
    RunCumsumInt32("int32 {10,64} dim=0 (axis=0)", di_axis0, {10, 64}, 0, stream);
    RunCumsumV2Int32("V2 int32 {10,64} excl=T dim=0", di_axis0, {10, 64}, 0, true, false, stream);
    RunCumsumV2Int32("V2 int32 {10,64} rev=T dim=0", di_axis0, {10, 64}, 0, false, true, stream);

    // axis=last (dim==dimNum-1): leftA = M, rightA = 1
    // triggers GetInputDims axis==dimNum-1 branch
    std::vector<int32_t> di_axis_last(10 * 64);
    for (int i = 0; i < 10 * 64; i++) di_axis_last[i] = i + 1;
    RunCumsumInt32("int32 {10,64} dim=1 (axis=last)", di_axis_last, {10, 64}, 1, stream);
    RunCumsumV2Int32("V2 int32 {10,64} excl=T dim=1", di_axis_last, {10, 64}, 1, true, false, stream);
    RunCumsumV2Int32("V2 int32 {10,64} rev=T dim=1", di_axis_last, {10, 64}, 1, false, true, stream);

    // middle axis: leftA = M, rightA = N (splits both LA and RA)
    // triggers GetInputDims else branch
    std::vector<int32_t> di_mid(2 * 32 * 8);
    for (int i = 0; i < 2 * 32 * 8; i++) di_mid[i] = i + 1;
    RunCumsumInt32("int32 {2,32,8} dim=1 (middle axis)", di_mid, {2, 32, 8}, 1, stream);
    RunCumsumV2Int32("V2 int32 {2,32,8} excl=T dim=1", di_mid, {2, 32, 8}, 1, true, false, stream);
    RunCumsumV2Int32("V2 int32 {2,32,8} rev=T dim=1", di_mid, {2, 32, 8}, 1, false, true, stream);
    RunCumsumV2Int32("V2 int32 {2,32,8} excl=T rev=T dim=1", di_mid, {2, 32, 8}, 1, true, true, stream);

    // TDRA path triggers: rightA * dtypeSize > vlSize/2
    // For ascend910_93, vlSize is typically 2048 (256B), so rightA needs to be > 256
    std::vector<int32_t> di_tdra(8 * 512);
    for (int i = 0; i < 8 * 512; i++) di_tdra[i] = i + 1;
    RunCumsumInt32("int32 {8,512} dim=0 (TDRA path)", di_tdra, {8, 512}, 0, stream);
    RunCumsumV2Int32("V2 int32 {8,512} dim=0 (TDRA path)", di_tdra, {8, 512}, 0, false, false, stream);

    // {4, 1024}: rightA large, triggers TDRA
    std::vector<int32_t> di_tdra2(4 * 1024);
    for (int i = 0; i < 4 * 1024; i++) di_tdra2[i] = i + 1;
    RunCumsumInt32("int32 {4,1024} dim=0 (TDRA heavy)", di_tdra2, {4, 1024}, 0, stream);
    RunCumsumV2Int32("V2 int32 {4,1024} rev=T (TDRA heavy)", di_tdra2, {4, 1024}, 0, false, true, stream);
}

void TestSuite_TilingOuterTD(aclrtStream stream)
{
    LOG_PRINT("\n=== Tiling Outer TD Branches (CORE_SS paths) ===\n");
    // RNGreaterClRNotFullLoadBorrowR: M very small, borrows both N and R
    // {2, 256, 4}: M=2 (tiny), N=4, R=256 -> triggers borrowN + borrowR path
    std::vector<float> d_outer1(2 * 256 * 4);
    for (int i = 0; i < 2 * 256 * 4; i++) d_outer1[i] = static_cast<float>(i % 10 + 1);
    RunCumsumFloat32("{2,256,4} dim=2 (outer-TD)", d_outer1, {2, 256, 4}, 2, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {2,256,4} excl=T dim=2", d_outer1, {2, 256, 4}, 2, true, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {2,256,4} rev=T dim=2", d_outer1, {2, 256, 4}, 2, false, true, stream, 1e-4, 1e-5);

    // {1, 512, 2}: M=1, N=2, R=512 -> extreme borrow
    std::vector<float> d_outer2(1 * 512 * 2);
    for (int i = 0; i < 1 * 512 * 2; i++) d_outer2[i] = static_cast<float>(i % 5 + 1);
    RunCumsumFloat32("{1,512,2} dim=2 (outer-TD extreme)", d_outer2, {1, 512, 2}, 2, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {1,512,2} excl=T dim=2", d_outer2, {1, 512, 2}, 2, true, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {1,512,2} excl=T rev=T dim=2", d_outer2, {1, 512, 2}, 2, true, true, stream, 1e-4, 1e-5);

    // CORE_SS path: lenM*lenN > clSize, lenM < coreNum/2
    // {2, 512}: M=2, N=1 (after dim=0 cumsum axis), R=512 -> borrows R
    std::vector<float> d_outer3(2 * 512);
    for (int i = 0; i < 2 * 512; i++) d_outer3[i] = static_cast<float>(i + 1);
    RunCumsumV2Float32("{2,512} V2 rev=T (borrowR)", d_outer3, {2, 512}, 0, false, true, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("{2,512} V2 excl=T rev=T (borrowR)", d_outer3, {2, 512}, 0, true, true, stream, 1e-4, 1e-5);
}

void TestSuite_TilingBoundaryShapes(aclrtStream stream)
{
    LOG_PRINT("\n=== Tiling Boundary Shapes ===\n");
    // UB full load boundary: lenR * alignN == ubSize
    // Float32 dtSize=4, blockSize=32, clSize=128 on ascend910_93
    std::vector<float> d_ub_boundary(16 * 128);
    for (int i = 0; i < 16 * 128; i++) d_ub_boundary[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{16,128} dim=0 (UB boundary)", d_ub_boundary, {16, 128}, 0, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {16,128} rev=T (UB boundary)", d_ub_boundary, {16, 128}, 0, false, true, stream, 1e-5, 1e-5);

    // lenM exactly == coreNum boundary: coreNum=64 on ascend910_93
    // {64, 128}: M=64 exactly, N=1, R=128 -> M >= coreNum branch
    std::vector<float> d_m_eq_core(64 * 128);
    for (int i = 0; i < 64 * 128; i++) d_m_eq_core[i] = static_cast<float>(i % 20 + 1);
    RunCumsumFloat32("{64,128} dim=0 (M==coreNum)", d_m_eq_core, {64, 128}, 0, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {64,128} excl=T dim=0", d_m_eq_core, {64, 128}, 0, true, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {64,128} rev=T dim=0", d_m_eq_core, {64, 128}, 0, false, true, stream, 1e-4, 1e-5);

    // {63, 128}: M just below coreNum
    std::vector<float> d_m_lt_core(63 * 128);
    for (int i = 0; i < 63 * 128; i++) d_m_lt_core[i] = static_cast<float>(i % 20 + 1);
    RunCumsumFloat32("{63,128} dim=0 (M<coreNum)", d_m_lt_core, {63, 128}, 0, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {63,128} excl=T dim=0", d_m_lt_core, {63, 128}, 0, true, false, stream, 1e-4, 1e-5);

    // {128, 64}: M=128 > coreNum, N=64
    std::vector<float> d_m_gt_core(128 * 64);
    for (int i = 0; i < 128 * 64; i++) d_m_gt_core[i] = static_cast<float>(i % 20 + 1);
    RunCumsumFloat32("{128,64} dim=0 (M>coreNum)", d_m_gt_core, {128, 64}, 0, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {128,64} excl=T dim=0", d_m_gt_core, {128, 64}, 0, true, false, stream, 1e-4, 1e-5);

    // lenN * dtSize == clSize boundary
    // clSize=128, dtSize=4 -> lenN=32
    std::vector<float> d_n_eq_cl(8 * 32);
    for (int i = 0; i < 8 * 32; i++) d_n_eq_cl[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{8,32} dim=0 (N*dtSize==clSize)", d_n_eq_cl, {8, 32}, 0, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {8,32} rev=T", d_n_eq_cl, {8, 32}, 0, false, true, stream, 1e-5, 1e-5);

    // lenN * dtSize < clSize (N lesser)
    std::vector<float> d_n_lt_cl(8 * 16);
    for (int i = 0; i < 8 * 16; i++) d_n_lt_cl[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{8,16} dim=0 (N*dtSize<clSize)", d_n_lt_cl, {8, 16}, 0, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {8,16} excl=T", d_n_lt_cl, {8, 16}, 0, true, false, stream, 1e-5, 1e-5);

    // {1, 128}: M=1 minimal, N=1, R=128
    std::vector<float> d_min_m(1 * 128);
    for (int i = 0; i < 1 * 128; i++) d_min_m[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("{1,128} dim=0 (M=1 minimal)", d_min_m, {1, 128}, 0, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {1,128} excl=T dim=0", d_min_m, {1, 128}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {1,128} rev=T dim=0", d_min_m, {1, 128}, 0, false, true, stream, 1e-5, 1e-5);
}

void TestSuite_IntTilingBoundary(aclrtStream stream)
{
    LOG_PRINT("\n=== Int Tiling Boundary Shapes ===\n");
    // TDLA path: leftA > coreNum/8 and rightA * dtypeSize <= vlSize/2
    // {64, 16}: leftA=64, rightA=16, axis=0
    std::vector<int32_t> di_tdla(64 * 16);
    for (int i = 0; i < 64 * 16; i++) di_tdla[i] = i + 1;
    RunCumsumInt32("int32 {64,16} dim=0 (TDLA path)", di_tdla, {64, 16}, 0, stream);
    RunCumsumV2Int32("V2 int32 {64,16} excl=T dim=0", di_tdla, {64, 16}, 0, true, false, stream);

    // {32, 32}: mid-sized for various tiling paths
    std::vector<int32_t> di_mid_int(32 * 32);
    for (int i = 0; i < 32 * 32; i++) di_mid_int[i] = i + 1;
    RunCumsumInt32("int32 {32,32} dim=0", di_mid_int, {32, 32}, 0, stream);
    RunCumsumInt32("int32 {32,32} dim=1", di_mid_int, {32, 32}, 1, stream);
    RunCumsumV2Int32("V2 int32 {32,32} excl=T dim=0", di_mid_int, {32, 32}, 0, true, false, stream);
    RunCumsumV2Int32("V2 int32 {32,32} rev=T dim=1", di_mid_int, {32, 32}, 1, false, true, stream);
    RunCumsumV2Int32("V2 int32 {32,32} excl=T rev=T dim=0", di_mid_int, {32, 32}, 0, true, true, stream);

    // {8, 256}: rightA large for TDRA
    std::vector<int32_t> di_tdra_int(8 * 256);
    for (int i = 0; i < 8 * 256; i++) di_tdra_int[i] = i + 1;
    RunCumsumInt32("int32 {8,256} dim=0 (TDRA)", di_tdra_int, {8, 256}, 0, stream);
    RunCumsumV2Int32("V2 int32 {8,256} rev=T (TDRA)", di_tdra_int, {8, 256}, 0, false, true, stream);

    // 3D int tiling: various axis positions
    std::vector<int32_t> di_3d_int(4 * 32 * 16);
    for (int i = 0; i < 4 * 32 * 16; i++) di_3d_int[i] = i + 1;
    RunCumsumInt32("int32 {4,32,16} dim=0", di_3d_int, {4, 32, 16}, 0, stream);
    RunCumsumInt32("int32 {4,32,16} dim=1", di_3d_int, {4, 32, 16}, 1, stream);
    RunCumsumInt32("int32 {4,32,16} dim=2", di_3d_int, {4, 32, 16}, 2, stream);
    RunCumsumV2Int32("V2 int32 {4,32,16} excl=T dim=1", di_3d_int, {4, 32, 16}, 1, true, false, stream);
    RunCumsumV2Int32("V2 int32 {4,32,16} rev=T dim=2", di_3d_int, {4, 32, 16}, 2, false, true, stream);
}

void TestSuite_Float16V2AndExclusiveReverse(aclrtStream stream)
{
    LOG_PRINT("\n=== Float16 V2 & Exclusive/Reverse Combos ===\n");
    std::vector<uint16_t> d8_f16;
    for (int i = 1; i <= 8; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i), &bits);
        d8_f16.push_back(bits);
    }
    // Float16 V2 all combos on 2D
    RunCumsumV2Float16("V2 FP16 {2,4} excl=T rev=F dim=0", d8_f16, {2, 4}, 0, true, false, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,4} excl=F rev=T dim=0", d8_f16, {2, 4}, 0, false, true, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,4} excl=T rev=T dim=0", d8_f16, {2, 4}, 0, true, true, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,4} excl=F rev=F dim=1", d8_f16, {2, 4}, 1, false, false, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,4} excl=T rev=F dim=1", d8_f16, {2, 4}, 1, true, false, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,4} excl=F rev=T dim=1", d8_f16, {2, 4}, 1, false, true, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,4} excl=T rev=T dim=1", d8_f16, {2, 4}, 1, true, true, stream, 1e-3, 1e-3);

    // Float16 V2 on longer sequence
    std::vector<uint16_t> d256_f16_long;
    for (int i = 0; i < 256; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i + 1), &bits);
        d256_f16_long.push_back(bits);
    }
    RunCumsumV2Float16("V2 FP16 len=256 excl=T rev=T dim=0", d256_f16_long, {256}, 0, true, true, stream, 1e-2, 1e-3);

    // Float16 V2 on 3D
    std::vector<uint16_t> d24_f16;
    for (int i = 0; i < 2 * 3 * 4; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i + 1), &bits);
        d24_f16.push_back(bits);
    }
    RunCumsumV2Float16("V2 FP16 {2,3,4} excl=T dim=0", d24_f16, {2, 3, 4}, 0, true, false, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,3,4} rev=T dim=2", d24_f16, {2, 3, 4}, 2, false, true, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 FP16 {2,3,4} excl=T rev=T dim=1", d24_f16, {2, 3, 4}, 1, true, true, stream, 1e-3, 1e-3);
}

void TestSuite_DimBranchesAndBoundaryDims(aclrtStream stream)
{
    LOG_PRINT("\n=== Dim Tensor Type Branches ===\n");
    // dim=0 -> DT_INT64 branch in both V1 and V2
    std::vector<float> d_for_dim0(32);
    for (int i = 0; i < 32; i++) d_for_dim0[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {4,8} dim=0 (INT64 dimTensor)", d_for_dim0, {4, 8}, 0, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 float32 {4,8} dim=0 (INT64 dimTensor)", d_for_dim0, {4, 8}, 0, false, false, stream, 1e-5, 1e-5);

    // dim=1 on small tensor -> DT_INT32 branch (dim > 0 && dim <= INT32_MAX)
    std::vector<float> d_for_dim1(32);
    for (int i = 0; i < 32; i++) d_for_dim1[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {4,8} dim=1 (INT32 dimTensor)", d_for_dim1, {4, 8}, 1, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 float32 {4,8} dim=1 (INT32 dimTensor)", d_for_dim1, {4, 8}, 1, true, false, stream, 1e-5, 1e-5);

    // 3D: dim=0, dim=1, dim=2 all covered for different dimTensor paths
    std::vector<float> d3d_dim(2 * 8 * 16);
    for (int i = 0; i < 2 * 8 * 16; i++) d3d_dim[i] = static_cast<float>(i + 1);
    RunCumsumFloat32("float32 {2,8,16} dim=0 (INT64)", d3d_dim, {2, 8, 16}, 0, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,8,16} dim=1 (INT32)", d3d_dim, {2, 8, 16}, 1, stream, 1e-5, 1e-5);
    RunCumsumFloat32("float32 {2,8,16} dim=2 (INT32)", d3d_dim, {2, 8, 16}, 2, stream, 1e-5, 1e-5);

    // Float16 and INT32 for dim branches
    std::vector<uint16_t> d3d_f16;
    for (int i = 0; i < 2 * 8 * 16; i++) {
        uint16_t bits; FloatToF16(static_cast<float>(i + 1), &bits);
        d3d_f16.push_back(bits);
    }
    RunCumsumFloat16("float16 {2,8,16} dim=0", d3d_f16, {2, 8, 16}, 0, stream, 1e-3, 1e-3);
    RunCumsumFloat16("float16 {2,8,16} dim=1", d3d_f16, {2, 8, 16}, 1, stream, 1e-3, 1e-3);

    std::vector<int32_t> d3d_i32;
    for (int i = 0; i < 2 * 8 * 16; i++) d3d_i32.push_back(i + 1);
    RunCumsumInt32("int32 {2,8,16} dim=0", d3d_i32, {2, 8, 16}, 0, stream);
    RunCumsumInt32("int32 {2,8,16} dim=1", d3d_i32, {2, 8, 16}, 1, stream);
    RunCumsumInt32("int32 {2,8,16} dim=2", d3d_i32, {2, 8, 16}, 2, stream);

    // negative dim on 3D
    RunCumsumV2Float32("V2 {2,8,16} dim=-1 excl=T", d3d_dim, {2, 8, 16}, -1, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 {2,8,16} dim=-2 rev=T", d3d_dim, {2, 8, 16}, -2, false, true, stream, 1e-5, 1e-5);
}

void TestSuite_V2Float32LongSeq(aclrtStream stream)
{
    LOG_PRINT("\n=== V2 Float32 Long Sequences ===\n");
    std::vector<float> d5000(5000);
    for (int i = 0; i < 5000; i++) d5000[i] = static_cast<float>(i + 1);
    RunCumsumV2Float32("V2 float32 len=5000 excl=T dim=0", d5000, {5000}, 0, true, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 float32 len=5000 rev=T dim=0", d5000, {5000}, 0, false, true, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 float32 len=5000 excl=T rev=T dim=0", d5000, {5000}, 0, true, true, stream, 1e-4, 1e-5);

    std::vector<float> d2000(2000);
    for (int i = 0; i < 2000; i++) d2000[i] = static_cast<float>(i + 1);
    RunCumsumV2Float32("V2 float32 len=2000 excl=T dim=0", d2000, {2000}, 0, true, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 float32 len=2000 rev=T dim=0", d2000, {2000}, 0, false, true, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 float32 len=2000 excl=T rev=T dim=0", d2000, {2000}, 0, true, true, stream, 1e-4, 1e-5);

    // 2D large V2
    std::vector<float> d2000_10(2000 * 10);
    for (int i = 0; i < 2000 * 10; i++) d2000_10[i] = static_cast<float>(i % 50 + 1);
    RunCumsumV2Float32("V2 {2000,10} excl=T dim=0", d2000_10, {2000, 10}, 0, true, false, stream, 1e-4, 1e-5);
    RunCumsumV2Float32("V2 {2000,10} rev=T dim=1", d2000_10, {2000, 10}, 1, false, true, stream, 1e-4, 1e-5);
}

void TestSuite_EmptyV2AllModes(aclrtStream stream)
{
    LOG_PRINT("\n=== Empty Tensor V2 All Modes ===\n");
    std::vector<float> emptyF;
    std::vector<uint16_t> emptyF16;
    std::vector<int32_t> emptyI;

    // V2 float32 all mode combinations on empty
    RunCumsumV2Float32("V2 empty excl=F rev=F", emptyF, {0}, 0, false, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 empty excl=T rev=F", emptyF, {0}, 0, true, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 empty excl=F rev=T", emptyF, {0}, 0, false, true, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 empty excl=T rev=T", emptyF, {0}, 0, true, true, stream, 1e-5, 1e-5);

    // {2,0} shape
    std::vector<float> shape20;
    RunCumsumV2Float32("V2 float32 {2,0} excl=F", shape20, {2, 0}, 0, false, false, stream, 1e-5, 1e-5);
    RunCumsumV2Float32("V2 float32 {2,0} excl=T rev=T", shape20, {2, 0}, 0, true, true, stream, 1e-5, 1e-5);

    // Float16 empty V2
    RunCumsumV2Float16("V2 float16 empty excl=F rev=F", emptyF16, {0}, 0, false, false, stream, 1e-3, 1e-3);
    RunCumsumV2Float16("V2 float16 empty excl=T rev=T", emptyF16, {0}, 0, true, true, stream, 1e-3, 1e-3);

    // Int32 empty V2
    RunCumsumV2Int32("V2 int32 empty excl=F rev=F", emptyI, {0}, 0, false, false, stream);
    RunCumsumV2Int32("V2 int32 empty excl=T rev=T", emptyI, {0}, 0, true, true, stream);
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed: %d\n", ret); return ret);

    TestSuite_Float32_Basic(stream);
    TestSuite_Float16_Basic(stream);
    TestSuite_Int32_Basic(stream);
    TestSuite_SequenceLengths(stream);
    TestSuite_NegativeAndMixed(stream);
    TestSuite_MultiDim(stream);
    TestSuite_TilingShapes(stream);
    TestSuite_CubePath(stream);
    TestSuite_CubePathV2(stream);
    TestSuite_Int32_Shapes(stream);
    TestSuite_V2_Basic(stream);
    TestSuite_V2_LongSequence(stream);
    TestSuite_V2_Shapes(stream);
    TestSuite_NegativeDim(stream);
    TestSuite_DtypeConversion(stream);
    TestSuite_EmptyTensor(stream);
    TestSuite_EmptyV2AllModes(stream);
    TestSuite_Overflow(stream);
    TestSuite_EdgePrecision(stream);
    TestSuite_TilingBorrowAxis(stream);
    TestSuite_TilingTWOWAY(stream);
    TestSuite_IntTilingAll(stream);
    TestSuite_IntTilingAxisBranches(stream);
    TestSuite_IntTilingBoundary(stream);
    TestSuite_TilingOuterTD(stream);
    TestSuite_TilingBoundaryShapes(stream);
    TestSuite_V2AllCombos(stream);
    TestSuite_DimBranchesAndBoundaryDims(stream);
    TestSuite_V2Float32LongSeq(stream);
    TestSuite_Float16V2AndExclusiveReverse(stream);
    TestSuite_LongAndLarge(stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n========== Summary ==========\n");
    LOG_PRINT("  Passed: %d\n", gPassed);
    LOG_PRINT("  Failed: %d\n", gFailed);
    LOG_PRINT("============================\n");
    return (gFailed > 0) ? gFailed : 0;
}
