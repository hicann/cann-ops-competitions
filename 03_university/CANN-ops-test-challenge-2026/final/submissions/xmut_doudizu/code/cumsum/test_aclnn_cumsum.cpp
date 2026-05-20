/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * ... (版权声明保持不变)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
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

static int32_t g_deviceId = 0;
static aclrtStream g_stream = nullptr;

int Init()
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(g_deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&g_stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize()
{
    aclrtDestroyStream(g_stream);
    aclrtResetDevice(g_deviceId);
    aclFinalize();
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty()) return 0;
    int64_t size = 1;
    for (auto d : shape) size *= d;
    return size;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(T);
    if (size == 0) {
        *deviceAddr = nullptr;
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                                  ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        return (*tensor != nullptr) ? ACL_SUCCESS : -1;
    }
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return (*tensor != nullptr) ? ACL_SUCCESS : -1;
}

int CreateAclTensorNoMalloc(const std::vector<int64_t>& shape, aclDataType dataType, aclTensor** tensor)
{
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    return (*tensor != nullptr) ? ACL_SUCCESS : -1;
}

void DestroyAclTensor(aclTensor* tensor, void* deviceAddr)
{
    if (tensor) aclDestroyTensor(tensor);
    if (deviceAddr) aclrtFree(deviceAddr);
}

// ==================== CPU 参考实现 ====================
template<typename T>
std::vector<T> CpuCumsum(const std::vector<T>& input, const std::vector<int64_t>& shape,
                         int64_t dim, bool exclusive, bool reverse)
{
    if (input.empty()) return {};
    int64_t rank = shape.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) return {};

    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    int64_t dimSize = shape[dim];
    for (int64_t i = dim + 1; i < rank; ++i) inner *= shape[i];

    std::vector<T> output(input.size());
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t d = 0; d < dimSize; ++d) {
            for (int64_t i = 0; i < inner; ++i) {
                int64_t idx = o * dimSize * inner + d * inner + i;
                T sum = 0;
                if (!reverse) {
                    for (int64_t k = 0; k <= d; ++k) {
                        if (!exclusive || k < d) {
                            int64_t srcIdx = o * dimSize * inner + k * inner + i;
                            sum += input[srcIdx];
                        }
                    }
                } else {
                    for (int64_t k = d; k < dimSize; ++k) {
                        if (!exclusive || k > d) {
                            int64_t srcIdx = o * dimSize * inner + k * inner + i;
                            sum += input[srcIdx];
                        }
                    }
                }
                output[idx] = sum;
            }
        }
    }
    return output;
}

template<typename T>
bool CompareWithTolerance(const std::vector<T>& expected, const std::vector<T>& actual,
                          double rtol = 1e-5, double atol = 1e-6)
{
    if (expected.size() != actual.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        double e = static_cast<double>(expected[i]);
        double a = static_cast<double>(actual[i]);
        double diff = std::abs(e - a);
        if (diff > atol && diff > rtol * std::abs(e)) {
            LOG_PRINT("Mismatch at %zu: expected %.12g, got %.12g, diff %.12g\n", i, e, a, diff);
            return false;
        }
    }
    return true;
}
template<> bool CompareWithTolerance(const std::vector<int32_t>& e, const std::vector<int32_t>& a, double, double) { return e == a; }
template<> bool CompareWithTolerance(const std::vector<int64_t>& e, const std::vector<int64_t>& a, double, double) { return e == a; }
template<> bool CompareWithTolerance(const std::vector<int8_t>& e, const std::vector<int8_t>& a, double, double) { return e == a; }
template<> bool CompareWithTolerance(const std::vector<uint8_t>& e, const std::vector<uint8_t>& a, double, double) { return e == a; }
template<> bool CompareWithTolerance(const std::vector<double>& e, const std::vector<double>& a, double rtol, double atol) {
    if (e.size() != a.size()) return false;
    for (size_t i = 0; i < e.size(); ++i) {
        double diff = std::abs(e[i] - a[i]);
        if (diff > atol && diff > rtol * std::abs(e[i])) return false;
    }
    return true;
}

template<typename T>
bool RunCumsumTestTemplate(const std::vector<T>& selfData, const std::vector<int64_t>& shape,
                           int64_t dim, aclDataType dtype, bool exclusive, bool reverse,
                           const std::vector<T>& expected = {})
{
    bool useV2 = (exclusive || reverse);
    std::vector<T> expectedResult = expected.empty() ?
        CpuCumsum(selfData, shape, dim, exclusive, reverse) : expected;
    if (expectedResult.empty() && !selfData.empty()) {
        LOG_PRINT("Reference calculation failed\n");
        return false;
    }

    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;
    int64_t outSize = GetShapeSize(shape);
    std::vector<T> actual(outSize, 0);
    bool success = false;

    int ret = CreateAclTensor(selfData, shape, &selfDev, dtype, &selfTensor);
    if (ret != ACL_SUCCESS) goto cleanup;
    ret = CreateAclTensor(actual, shape, &outDev, dtype, &outTensor);
    if (ret != ACL_SUCCESS) goto cleanup;

    if (useV2) {
        ret = aclnnCumsumV2GetWorkspaceSize(selfTensor, dim, exclusive, reverse, outTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) goto cleanup;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) goto cleanup;
        }
        ret = aclnnCumsumV2(workspace, workspaceSize, executor, g_stream);
    } else {
        ret = aclnnCumsumGetWorkspaceSize(selfTensor, dim, dtype, outTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) goto cleanup;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) goto cleanup;
        }
        ret = aclnnCumsum(workspace, workspaceSize, executor, g_stream);
    }
    if (ret != ACL_SUCCESS) goto cleanup;

    ret = aclrtSynchronizeStream(g_stream);
    if (ret != ACL_SUCCESS) goto cleanup;

    ret = aclrtMemcpy(actual.data(), outSize * sizeof(T), outDev,
                      outSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) goto cleanup;

    success = CompareWithTolerance(expectedResult, actual, 1e-5, 1e-6);
    if (!success) {
        for (int64_t i = 0; i < std::min(10L, outSize); ++i) {
            LOG_PRINT("  idx %ld: expected %.6f, actual %.6f\n", i, (double)expectedResult[i], (double)actual[i]);
        }
    }

cleanup:
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    if (workspace) aclrtFree(workspace);
    return success;
}

template<typename T>
bool RunCumsumTest(const std::vector<T>& self, const std::vector<int64_t>& shape, int64_t dim, aclDataType dtype)
{
    return RunCumsumTestTemplate(self, shape, dim, dtype, false, false);
}

template<typename T>
bool RunCumsumV2Test(const std::vector<T>& self, const std::vector<int64_t>& shape, int64_t dim,
                     aclDataType dtype, bool exclusive, bool reverse)
{
    return RunCumsumTestTemplate(self, shape, dim, dtype, exclusive, reverse);
}

// ==================== 原有测试 ====================
bool TestBasic()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> expected = {1, 2, 4, 6};
    return RunCumsumTestTemplate(self, shape, 0, ACL_FLOAT, false, false, expected);
}

bool TestDimLast()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    return RunCumsumTest(self, shape, 1, ACL_FLOAT);
}

bool TestNegativeDim()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    return RunCumsumTest(self, shape, -1, ACL_FLOAT);
}

bool TestNegativeDimOutOfRange()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret) return false;
    ret = CreateAclTensor(self, shape, &outDev, ACL_FLOAT, &outTensor);
    if (ret) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, -3, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

bool TestExclusive()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> expected = {0, 0, 1, 2};
    return RunCumsumV2Test(self, shape, 0, ACL_FLOAT, true, false);
}

bool TestReverse()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> expected = {4, 6, 3, 4};
    return RunCumsumV2Test(self, shape, 0, ACL_FLOAT, false, true);
}

bool TestExclusiveReverse()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> expected = {3, 4, 0, 0};
    return RunCumsumV2Test(self, shape, 0, ACL_FLOAT, true, true);
}

bool TestEmptyTensor()
{
    std::vector<float> self;
    std::vector<int64_t> shape = {0, 2};
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret) return false;
    ret = CreateAclTensor(self, shape, &outDev, ACL_FLOAT, &outTensor);
    if (ret) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS && workspaceSize == 0);
    if (ok) {
        ret = aclnnCumsum(nullptr, 0, executor, g_stream);
        if (ret == ACL_SUCCESS) aclrtSynchronizeStream(g_stream);
        else ok = false;
    }
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

bool TestV2EmptyTensor()
{
    std::vector<float> self;
    std::vector<int64_t> shape = {0, 2};
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret) return false;
    ret = CreateAclTensor(self, shape, &outDev, ACL_FLOAT, &outTensor);
    if (ret) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumV2GetWorkspaceSize(selfTensor, 0, false, false, outTensor, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS && workspaceSize == 0);
    if (ok) {
        ret = aclnnCumsumV2(nullptr, 0, executor, g_stream);
        if (ret == ACL_SUCCESS) aclrtSynchronizeStream(g_stream);
        else ok = false;
    }
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

bool TestInvalidDim()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> shape = {2, 2};
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret) return false;
    ret = CreateAclTensor(self, shape, &outDev, ACL_FLOAT, &outTensor);
    if (ret) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, 2, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

bool TestShapeMismatch()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int64_t> selfShape = {2, 2};
    std::vector<int64_t> outShape = {1, 4};
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, selfShape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret) return false;
    ret = CreateAclTensor(self, outShape, &outDev, ACL_FLOAT, &outTensor);
    if (ret) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

bool TestDtypeMismatch()
{
    std::vector<float> self = {1, 2, 3, 4};
    std::vector<int32_t> outData = {0, 0, 0, 0};
    std::vector<int64_t> shape = {2, 2};
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret) return false;
    ret = CreateAclTensor(outData, shape, &outDev, ACL_INT32, &outTensor);
    if (ret) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

bool TestDouble()
{
    std::vector<double> self = {1.0, 2.0, 3.0, 4.0};
    std::vector<int64_t> shape = {2, 2};
    std::vector<double> expected = {1.0, 2.0, 4.0, 6.0};
    return RunCumsumTestTemplate(self, shape, 0, ACL_DOUBLE, false, false, expected);
}

bool TestHighDim()
{
    std::vector<float> self(27, 1.0f);
    std::vector<int64_t> shape = {3, 3, 3};
    std::vector<float> expected(27);
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 3; ++j)
            for (int64_t k = 0; k < 3; ++k) {
                int64_t idx = i*9 + j*3 + k;
                expected[idx] = static_cast<float>(j+1);
            }
    return RunCumsumTestTemplate(self, shape, 1, ACL_FLOAT, false, false, expected);
}

bool TestLargeDimZero()
{
    std::vector<float> self(100, 1.0f);
    std::vector<int64_t> shape = {100};
    std::vector<float> expected(100);
    float sum = 0;
    for (int64_t i = 0; i < 100; ++i) { sum += 1.0f; expected[i] = sum; }
    return RunCumsumTestTemplate(self, shape, 0, ACL_FLOAT, false, false, expected);
}

bool TestLongSequence()
{
    const int64_t N = 1500;
    std::vector<float> self(N);
    for (int64_t i = 0; i < N; ++i) self[i] = i + 1.0f;
    std::vector<int64_t> shape = {N};
    std::vector<float> expected(N);
    double sum = 0;
    for (int64_t i = 0; i < N; ++i) { sum += self[i]; expected[i] = static_cast<float>(sum); }
    return RunCumsumTestTemplate(self, shape, 0, ACL_FLOAT, false, false, expected);
}

bool TestV2Variants()
{
    std::vector<float> self = {1,2,3,4,5,6,7,8,9,10};
    std::vector<int64_t> shape = {10};
    std::vector<float> exp1(10);
    float s=0;
    for(int i=0;i<10;++i){ s+=self[i]; exp1[i]=s; }
    if(!RunCumsumV2Test(self, shape, 0, ACL_FLOAT, false, false)) return false;

    std::vector<float> exp2(10);
    s=0;
    for(int i=0;i<10;++i){ exp2[i]=s; s+=self[i]; }
    if(!RunCumsumV2Test(self, shape, 0, ACL_FLOAT, true, false)) return false;

    std::vector<float> exp3(10);
    s=0;
    for(int i=9;i>=0;--i){ s+=self[i]; exp3[i]=s; }
    if(!RunCumsumV2Test(self, shape, 0, ACL_FLOAT, false, true)) return false;

    std::vector<float> exp4(10);
    s=0;
    for(int i=9;i>=0;--i){ exp4[i]=s; s+=self[i]; }
    return RunCumsumV2Test(self, shape, 0, ACL_FLOAT, true, true);
}

bool TestIntAxisMiddle()
{
    std::vector<int32_t> self(2*3*4, 1);
    std::vector<int64_t> shape = {2,3,4};
    std::vector<int32_t> expected(24);
    for (int64_t o = 0; o < 2; ++o)
        for (int64_t d = 0; d < 3; ++d)
            for (int64_t i = 0; i < 4; ++i) {
                int64_t idx = o*12 + d*4 + i;
                expected[idx] = static_cast<int32_t>(d+1);
            }
    return RunCumsumTestTemplate(self, shape, 1, ACL_INT32, false, false, expected);
}

bool TestLargeRightAxisLen()
{
    const int64_t N = 5000;
    std::vector<int32_t> self(N, 1);
    std::vector<int64_t> shape = {N};
    std::vector<int32_t> expected(N);
    int32_t sum = 0;
    for (int64_t i = 0; i < N; ++i) { sum += 1; expected[i] = sum; }
    return RunCumsumTestTemplate(self, shape, 0, ACL_INT32, false, false, expected);
}

bool TestLargeLeftAxisLen()
{
    const int64_t left = 2000, mid = 10, right = 10;
    std::vector<int64_t> shape = {left, mid, right};
    int64_t total = left * mid * right;
    std::vector<int32_t> self(total, 1);
    std::vector<int32_t> expected(total);
    for (int64_t o = 0; o < left; ++o) {
        for (int64_t d = 0; d < mid; ++d) {
            for (int64_t i = 0; i < right; ++i) {
                int64_t idx = o*mid*right + d*right + i;
                expected[idx] = static_cast<int32_t>(d+1);
            }
        }
    }
    return RunCumsumTestTemplate(self, shape, 1, ACL_INT32, false, false, expected);
}

bool TestRaAxisWeightMax()
{
    std::vector<int64_t> shape = {2, 3, 5000};
    int64_t total = 2 * 3 * 5000;
    std::vector<int32_t> self(total, 1);
    std::vector<int32_t> expected(total);
    for (int64_t o = 0; o < 2; ++o) {
        for (int64_t d = 0; d < 3; ++d) {
            for (int64_t i = 0; i < 5000; ++i) {
                int64_t idx = o*3*5000 + d*5000 + i;
                expected[idx] = static_cast<int32_t>(d+1);
            }
        }
    }
    return RunCumsumTestTemplate(self, shape, 1, ACL_INT32, false, false, expected);
}

bool TestRAxisWeightMax()
{
    std::vector<int64_t> shape = {2, 1024, 2};
    int64_t total = 2 * 1024 * 2;
    std::vector<int32_t> self(total, 1);
    std::vector<int32_t> expected(total);
    for (int64_t o = 0; o < 2; ++o) {
        for (int64_t d = 0; d < 1024; ++d) {
            for (int64_t i = 0; i < 2; ++i) {
                int64_t idx = o*1024*2 + d*2 + i;
                expected[idx] = static_cast<int32_t>(d+1);
            }
        }
    }
    return RunCumsumTestTemplate(self, shape, 1, ACL_INT32, false, false, expected);
}

bool TestLargeRightAxisLenForTilingKey()
{
    const int64_t N = 5000;
    std::vector<int32_t> self(N, 1);
    std::vector<int64_t> shape = {N};
    std::vector<int32_t> expected(N);
    int32_t sum = 0;
    for (int64_t i = 0; i < N; ++i) { sum += 1; expected[i] = sum; }
    return RunCumsumTestTemplate(self, shape, 0, ACL_INT32, false, false, expected);
}

bool TestIntNegativeDim()
{
    std::vector<int32_t> self = {1,2,3,4,5,6,7,8};
    std::vector<int64_t> shape = {2,2,2};
    std::vector<int32_t> expected(8);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            int32_t sum = 0;
            for (int64_t k = 0; k < 2; ++k) {
                sum += self[i*4 + j*2 + k];
                expected[i*4 + j*2 + k] = sum;
            }
        }
    }
    return RunCumsumTestTemplate(self, shape, -1, ACL_INT32, false, false, expected);
}

bool TestIntLastDim()
{
    std::vector<int32_t> self = {1,2,3,4,5,6,7,8};
    std::vector<int64_t> shape = {2,2,2};
    std::vector<int32_t> expected(8);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 2; ++j) {
            int32_t sum = 0;
            for (int64_t k = 0; k < 2; ++k) {
                sum += self[i*4 + j*2 + k];
                expected[i*4 + j*2 + k] = sum;
            }
        }
    }
    return RunCumsumTestTemplate(self, shape, 2, ACL_INT32, false, false, expected);
}

bool TestInt32()
{
    std::vector<int32_t> self = {1,2,3,4};
    std::vector<int64_t> shape = {2,2};
    std::vector<int32_t> expected = {1,2,4,6};
    return RunCumsumTestTemplate(self, shape, 0, ACL_INT32, false, false, expected);
}

bool TestInt64()
{
    std::vector<int64_t> self = {1,2,3,4};
    std::vector<int64_t> shape = {2,2};
    std::vector<int64_t> expected = {1,2,4,6};
    return RunCumsumTestTemplate(self, shape, 0, ACL_INT64, false, false, expected);
}

bool TestVariousDtypesFixed()
{
    bool ok = true;
    ok &= TestInt32();
    ok &= TestInt64();
    ok &= TestDouble();
    return ok;
}

// ==================== 覆盖率增强测试 ====================
static bool IsCubeSupported()
{
    const char* env = std::getenv("TEST_CUBE_ENABLE");
    return (env != nullptr && std::string(env) == "1");
}

// 1. Cube 正向路径（仅当芯片支持且环境变量设置）
bool TestCubePath()
{
    if (!IsCubeSupported()) {
        LOG_PRINT("Cube test skipped (set TEST_CUBE_ENABLE=1 to enable)\n");
        return true;
    }
    const int64_t batch = 12800;
    const int64_t channel = 512;
    std::vector<int64_t> shape = {batch, channel};
    int64_t total = batch * channel;
    std::vector<float> self(total, 1.0f);
    std::vector<float> expected(total);
    for (int64_t i = 0; i < batch; ++i) {
        float sum = 0;
        for (int64_t j = 0; j < channel; ++j) {
            sum += 1.0f;
            expected[i * channel + j] = sum;
        }
    }
    return RunCumsumTestTemplate(self, shape, 1, ACL_FLOAT, false, false, expected);
}

// 2. Cube 条件但 dtype 不支持（int32），触发 dtype 不匹配分支
bool TestCubeDtypeUnsupported()
{
    if (!IsCubeSupported()) {
        LOG_PRINT("Cube dtype unsupported test skipped (set TEST_CUBE_ENABLE=1)\n");
        return true;
    }
    const int64_t batch = 12800;
    const int64_t channel = 512;
    std::vector<int64_t> shape = {batch, channel};
    int64_t total = batch * channel;
    std::vector<int32_t> self(total, 1);
    std::vector<int32_t> expected(total);
    for (int64_t i = 0; i < batch; ++i) {
        int32_t sum = 0;
        for (int64_t j = 0; j < channel; ++j) {
            sum += 1;
            expected[i * channel + j] = sum;
        }
    }
    // 即使 dtype 不支持 cube，普通 cumsum 仍应正确工作
    return RunCumsumTestTemplate(self, shape, 1, ACL_INT32, false, false, expected);
}

// 3. Cube 条件满足但 dim 不为最后一维，触发 CheckShapeIsSupport 中的 false 分支
bool TestCubeNotLastDim()
{
    if (!IsCubeSupported()) {
        LOG_PRINT("Cube not last dim test skipped (set TEST_CUBE_ENABLE=1)\n");
        return true;
    }
    const int64_t batch = 12800;
    const int64_t channel = 512;
    const int64_t extra = 2;
    std::vector<int64_t> shape = {batch, channel, extra};
    int64_t total = batch * channel * extra;
    std::vector<float> self(total, 1.0f);
    std::vector<float> expected(total);
    // dim=0 沿着 batch 方向累加
    for (int64_t o = 0; o < batch; ++o) {
        for (int64_t d = 0; d < channel; ++d) {
            for (int64_t i = 0; i < extra; ++i) {
                float sum = 0;
                for (int64_t k = 0; k <= o; ++k) {
                    sum += self[k * channel * extra + d * extra + i];
                }
                expected[o * channel * extra + d * extra + i] = sum;
            }
        }
    }
    return RunCumsumTestTemplate(self, shape, 0, ACL_FLOAT, false, false, expected);
}

// 4. Cube 条件但某一维超过 50000000，触发 size 超限分支
bool TestCubeDimTooLarge()
{
    if (!IsCubeSupported()) {
        LOG_PRINT("Cube dim too large test skipped (set TEST_CUBE_ENABLE=1)\n");
        return true;
    }
    const int64_t batch = 12800;
    const int64_t channel = 60000000; // > CUMSUM_CUBE_MAX_SUPPORT_SIZE
    std::vector<int64_t> shape = {batch, channel};
    // 由于内存极大，只校验 GetWorkspaceSize 成功（普通路径），不实际执行 kernel
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 这里需要分配实际内存，但 channel 达 60M，batch 12800，总内存超限，改用 no malloc 方式
    int ret = CreateAclTensorNoMalloc(shape, ACL_FLOAT, &selfTensor);
    if (ret != ACL_SUCCESS) return false;
    ret = CreateAclTensorNoMalloc(shape, ACL_FLOAT, &outTensor);
    if (ret != ACL_SUCCESS) {
        aclDestroyTensor(selfTensor);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, 1, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret == ACL_SUCCESS);
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(outTensor);
    return ok;
}

// 5. 0 维张量（标量）触发 CheckShapeIsSupport 中的 zero-dim 分支
bool TestZeroDimTensor()
{
    std::vector<float> self = {42.0f};
    std::vector<int64_t> shape = {};
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret != ACL_SUCCESS) return false;
    ret = CreateAclTensor(self, shape, &outDev, ACL_FLOAT, &outTensor);
    if (ret != ACL_SUCCESS) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS); // 标量不支持，应失败
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

// 6. 高维（9 维）触发最大维度检查
bool TestHighDimOver8()
{
    std::vector<int64_t> shape = {1,1,1,1,1,1,1,1,1};
    std::vector<float> self(1, 1.0f);
    void* selfDev = nullptr;
    void* outDev = nullptr;
    aclTensor* selfTensor = nullptr;
    aclTensor* outTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfTensor);
    if (ret) return false;
    ret = CreateAclTensor(self, shape, &outDev, ACL_FLOAT, &outTensor);
    if (ret) {
        DestroyAclTensor(selfTensor, selfDev);
        return false;
    }
    ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_FLOAT, outTensor, &workspaceSize, &executor);
    bool ok = (ret != ACL_SUCCESS);
    DestroyAclTensor(selfTensor, selfDev);
    DestroyAclTensor(outTensor, outDev);
    return ok;
}

int main()
{
    if (Init() != 0) {
        LOG_PRINT("Init failed\n");
        return 1;
    }

    int passed = 0, total = 0;
    auto run = [&](bool result, const char* name) {
        total++;
        if (result) passed++;
        LOG_PRINT("[%s] %s\n", result ? "PASS" : "FAIL", name);
    };

    run(TestBasic(), "Basic");
    run(TestDimLast(), "DimLast");
    run(TestNegativeDim(), "NegativeDim");
    run(TestNegativeDimOutOfRange(), "NegativeDimOutOfRange");
    run(TestExclusive(), "Exclusive");
    run(TestReverse(), "Reverse");
    run(TestExclusiveReverse(), "ExclusiveReverse");
    run(TestEmptyTensor(), "EmptyTensor");
    run(TestV2EmptyTensor(), "V2EmptyTensor");
    run(TestInvalidDim(), "InvalidDim");
    run(TestShapeMismatch(), "ShapeMismatch");
    run(TestDtypeMismatch(), "DtypeMismatch");
    run(TestDouble(), "Double");
    run(TestHighDim(), "HighDim");
    run(TestLargeDimZero(), "LargeDimZero");
    run(TestLongSequence(), "LongSequence");
    run(TestV2Variants(), "V2Variants");
    run(TestIntAxisMiddle(), "IntAxisMiddle");
    run(TestLargeRightAxisLen(), "LargeRightAxisLen");
    run(TestLargeLeftAxisLen(), "LargeLeftAxisLen");
    run(TestRaAxisWeightMax(), "RaAxisWeightMax");
    run(TestRAxisWeightMax(), "RAxisWeightMax");
    run(TestLargeRightAxisLenForTilingKey(), "LargeRightAxisLenForTilingKey");
    run(TestIntNegativeDim(), "IntNegativeDim");
    run(TestIntLastDim(), "IntLastDim");
    run(TestVariousDtypesFixed(), "VariousDtypesFixed");

    run(TestCubePath(), "CubePath");
    run(TestCubeDtypeUnsupported(), "CubeDtypeUnsupported");
    run(TestCubeNotLastDim(), "CubeNotLastDim");
    run(TestCubeDimTooLarge(), "CubeDimTooLarge");
    run(TestZeroDimTensor(), "ZeroDimTensor");
    run(TestHighDimOver8(), "HighDimOver8");

#ifdef ENABLE_BF16
    // run(TestBF16(), "BF16");
#endif

    LOG_PRINT("Summary: %d/%d passed\n", passed, total);
    Finalize();
    return (passed == total) ? 0 : 1;
}