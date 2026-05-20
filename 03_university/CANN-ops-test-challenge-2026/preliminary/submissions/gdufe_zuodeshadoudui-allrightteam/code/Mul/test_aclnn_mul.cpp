#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

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

// 测试统计
static int g_total_tests = 0;
static int g_passed_tests = 0;
static int g_failed_tests = 0;

#define TEST_PASS()              \
    do {                         \
        g_total_tests++;         \
        g_passed_tests++;        \
        LOG_PRINT("  [PASS]\n"); \
    } while (0)
#define TEST_FAIL()              \
    do {                         \
        g_total_tests++;         \
        g_failed_tests++;        \
        LOG_PRINT("  [FAIL]\n"); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 创建 aclScalar
template <typename T>
aclScalar* CreateAclScalar(T value, aclDataType dataType)
{
    return aclCreateScalar(&value, dataType);
}

// 计算广播后的索引
size_t GetBroadcastIndex(size_t linearIdx, const std::vector<int64_t>& shape, const std::vector<int64_t>& targetShape)
{
    std::vector<int64_t> coord(shape.size(), 0);
    size_t tmp = linearIdx;
    for (int64_t i = shape.size() - 1; i >= 0; i--) {
        coord[i] = tmp % shape[i];
        tmp /= shape[i];
    }

    size_t targetIdx = 0;
    size_t stride = 1;
    for (int64_t i = targetShape.size() - 1; i >= 0; i--) {
        int64_t srcIdx = i - (targetShape.size() - shape.size());
        int64_t val = (srcIdx >= 0 && shape[srcIdx] > 1) ? coord[srcIdx] : 0;
        targetIdx += val * stride;
        stride *= targetShape[i];
    }
    return targetIdx;
}

// 整数验证
template <typename T>
bool VerifyResult(
    const std::vector<T>& result, const std::vector<T>& selfData, const std::vector<T>& otherData,
    const std::vector<int64_t>& outShape, const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape)
{
    for (size_t i = 0; i < result.size(); i++) {
        size_t selfIdx = GetBroadcastIndex(i, outShape, selfShape);
        size_t otherIdx = GetBroadcastIndex(i, outShape, otherShape);
        T expected = selfData[selfIdx] * otherData[otherIdx];
        if (result[i] != expected) {
            LOG_PRINT("    Mismatch at %zu: result=%d, expected=%d\n", i, (int)result[i], (int)expected);
            return false;
        }
    }
    return true;
}

// float 验证
template <>
bool VerifyResult<float>(
    const std::vector<float>& result, const std::vector<float>& selfData, const std::vector<float>& otherData,
    const std::vector<int64_t>& outShape, const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape)
{
    for (size_t i = 0; i < result.size(); i++) {
        size_t selfIdx = GetBroadcastIndex(i, outShape, selfShape);
        size_t otherIdx = GetBroadcastIndex(i, outShape, otherShape);
        float expected = selfData[selfIdx] * otherData[otherIdx];
        float diff = std::abs(result[i] - expected);
        if (diff > 1e-5 * std::abs(expected) + 1e-5) {
            LOG_PRINT("    Mismatch at %zu: result=%f, expected=%f\n", i, result[i], expected);
            return false;
        }
    }
    return true;
}

// double 验证
template <>
bool VerifyResult<double>(
    const std::vector<double>& result, const std::vector<double>& selfData, const std::vector<double>& otherData,
    const std::vector<int64_t>& outShape, const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape)
{
    for (size_t i = 0; i < result.size(); i++) {
        size_t selfIdx = GetBroadcastIndex(i, outShape, selfShape);
        size_t otherIdx = GetBroadcastIndex(i, outShape, otherShape);
        double expected = selfData[selfIdx] * otherData[otherIdx];
        double diff = std::abs(result[i] - expected);
        if (diff > 1e-9 * std::abs(expected) + 1e-9) {
            LOG_PRINT("    Mismatch at %zu: result=%f, expected=%f\n", i, result[i], expected);
            return false;
        }
    }
    return true;
}

// 简化版：同 shape 验证
template <typename T>
bool VerifySameShape(const std::vector<T>& result, const std::vector<T>& selfData, const std::vector<T>& otherData)
{
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != selfData[i] * otherData[i])
            return false;
    }
    return true;
}

template <>
bool VerifySameShape<float>(
    const std::vector<float>& result, const std::vector<float>& selfData, const std::vector<float>& otherData)
{
    for (size_t i = 0; i < result.size(); i++) {
        float expected = selfData[i] * otherData[i];
        float diff = std::abs(result[i] - expected);
        if (diff > 1e-5 * std::abs(expected) + 1e-5)
            return false;
    }
    return true;
}

template <>
bool VerifySameShape<double>(
    const std::vector<double>& result, const std::vector<double>& selfData, const std::vector<double>& otherData)
{
    for (size_t i = 0; i < result.size(); i++) {
        double expected = selfData[i] * otherData[i];
        double diff = std::abs(result[i] - expected);
        if (diff > 1e-9 * std::abs(expected) + 1e-9)
            return false;
    }
    return true;
}

// 标量乘法验证
template <typename T>
bool VerifyMulsResult(const std::vector<T>& result, const std::vector<T>& selfData, T scalar)
{
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != selfData[i] * scalar)
            return false;
    }
    return true;
}

template <>
bool VerifyMulsResult<float>(const std::vector<float>& result, const std::vector<float>& selfData, float scalar)
{
    for (size_t i = 0; i < result.size(); i++) {
        float expected = selfData[i] * scalar;
        float diff = std::abs(result[i] - expected);
        if (diff > 1e-5 * std::abs(expected) + 1e-5)
            return false;
    }
    return true;
}

// ============== aclnnMul 测试 ==============
template <typename T>
void RunSingleTest(
    const std::vector<T>& selfData, const std::vector<int64_t>& selfShape, const std::vector<T>& otherData,
    const std::vector<int64_t>& otherShape, aclDataType dataType, aclrtStream stream, const std::string& testName,
    bool expectSuccess = true)
{
    LOG_PRINT("  Test: %s\n", testName.c_str());

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;

    // 广播 shape 推导
    std::vector<int64_t> outShape;
    size_t maxDims = std::max(selfShape.size(), otherShape.size());
    outShape.resize(maxDims, 1);
    bool broadcastValid = true;
    for (size_t i = 0; i < maxDims; i++) {
        int64_t selfDim = (i < selfShape.size()) ? selfShape[selfShape.size() - 1 - i] : 1;
        int64_t otherDim = (i < otherShape.size()) ? otherShape[otherShape.size() - 1 - i] : 1;
        if (selfDim != otherDim && selfDim != 1 && otherDim != 1) {
            broadcastValid = false;
            break;
        }
        outShape[maxDims - 1 - i] = std::max(selfDim, otherDim);
    }
    if (!broadcastValid) {
        if (!expectSuccess) {
            TEST_PASS();
            return;
        }
        LOG_PRINT("    Incompatible broadcast shape\n");
        TEST_FAIL();
        return;
    }

    auto ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dataType, &self);
    if (ret != 0) {
        if (!expectSuccess) {
            TEST_PASS();
            return;
        }
        TEST_FAIL();
        return;
    }
    ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dataType, &other);
    if (ret != 0) {
        if (!expectSuccess) {
            TEST_PASS();
            goto cleanup_self;
        }
        TEST_FAIL();
        goto cleanup_self;
    }
    ret = CreateEmptyAclTensor<T>(outShape, &outDeviceAddr, dataType, &out);
    if (ret != 0) {
        if (!expectSuccess) {
            TEST_PASS();
            goto cleanup_other;
        }
        TEST_FAIL();
        goto cleanup_other;
    }

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            if (!expectSuccess) {
                TEST_PASS();
                goto cleanup_out;
            }
            TEST_FAIL();
            goto cleanup_out;
        }

        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                TEST_FAIL();
                goto cleanup_out;
            }
        }

        ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            if (!expectSuccess) {
                TEST_PASS();
                if (workspaceSize > 0)
                    aclrtFree(workspaceAddr);
                goto cleanup_out;
            }
            TEST_FAIL();
            if (workspaceSize > 0)
                aclrtFree(workspaceAddr);
            goto cleanup_out;
        }

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
            if (workspaceSize > 0)
                aclrtFree(workspaceAddr);
            goto cleanup_out;
        }

        int64_t outSize = GetShapeSize(outShape);
        std::vector<T> resultData(outSize);
        ret = aclrtMemcpy(
            resultData.data(), outSize * sizeof(T), outDeviceAddr, outSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
        } else {
            bool pass;
            if (selfShape == otherShape) {
                pass = VerifySameShape(resultData, selfData, otherData);
            } else {
                pass = VerifyResult(resultData, selfData, otherData, outShape, selfShape, otherShape);
            }
            if (pass) {
                TEST_PASS();
            } else {
                TEST_FAIL();
            }
        }
        if (workspaceSize > 0)
            aclrtFree(workspaceAddr);
    }

cleanup_out:
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
cleanup_other:
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
cleanup_self:
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
}

// ============== aclnnMuls 测试 ==============
template <typename T>
void RunMulsTest(
    const std::vector<T>& selfData, const std::vector<int64_t>& selfShape, T scalarValue, aclDataType dataType,
    aclrtStream stream, const std::string& testName, bool expectSuccess = true)
{
    LOG_PRINT("  Test: %s\n", testName.c_str());

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dataType, &self);
    if (ret != 0) {
        if (!expectSuccess) {
            TEST_PASS();
            return;
        }
        TEST_FAIL();
        return;
    }
    ret = CreateEmptyAclTensor<T>(selfShape, &outDeviceAddr, dataType, &out);
    if (ret != 0) {
        if (!expectSuccess) {
            TEST_PASS();
            goto cleanup_self;
        }
        TEST_FAIL();
        goto cleanup_self;
    }

    {
        aclScalar* scalar = CreateAclScalar(scalarValue, dataType);
        if (!scalar) {
            if (!expectSuccess) {
                TEST_PASS();
                goto cleanup_out;
            }
            TEST_FAIL();
            goto cleanup_out;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            if (!expectSuccess) {
                TEST_PASS();
                aclDestroyScalar(scalar);
                goto cleanup_out;
            }
            TEST_FAIL();
            aclDestroyScalar(scalar);
            goto cleanup_out;
        }

        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                TEST_FAIL();
                aclDestroyScalar(scalar);
                goto cleanup_out;
            }
        }

        ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            if (!expectSuccess) {
                TEST_PASS();
                if (workspaceSize > 0)
                    aclrtFree(workspaceAddr);
                aclDestroyScalar(scalar);
                goto cleanup_out;
            }
            TEST_FAIL();
            if (workspaceSize > 0)
                aclrtFree(workspaceAddr);
            aclDestroyScalar(scalar);
            goto cleanup_out;
        }

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
            if (workspaceSize > 0)
                aclrtFree(workspaceAddr);
            aclDestroyScalar(scalar);
            goto cleanup_out;
        }

        int64_t outSize = GetShapeSize(selfShape);
        std::vector<T> resultData(outSize);
        ret = aclrtMemcpy(
            resultData.data(), outSize * sizeof(T), outDeviceAddr, outSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
        } else {
            if (VerifyMulsResult(resultData, selfData, scalarValue)) {
                TEST_PASS();
            } else {
                TEST_FAIL();
            }
        }
        if (workspaceSize > 0)
            aclrtFree(workspaceAddr);
        aclDestroyScalar(scalar);
    }

cleanup_out:
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
cleanup_self:
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
}

// ============== aclnnInplaceMuls 测试 ==============
template <typename T>
void RunInplaceMulsTest(
    std::vector<T>& selfData, const std::vector<int64_t>& selfShape, T scalarValue, aclDataType dataType,
    aclrtStream stream, const std::string& testName)
{
    LOG_PRINT("  Test: %s\n", testName.c_str());

    void* selfDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;

    auto ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dataType, &selfRef);
    if (ret != 0) {
        TEST_FAIL();
        return;
    }

    {
        aclScalar* scalar = CreateAclScalar(scalarValue, dataType);
        if (!scalar) {
            TEST_FAIL();
            goto cleanup_self;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        ret = aclnnInplaceMulsGetWorkspaceSize(selfRef, scalar, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
            aclDestroyScalar(scalar);
            goto cleanup_self;
        }

        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                TEST_FAIL();
                aclDestroyScalar(scalar);
                goto cleanup_self;
            }
        }

        ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
            if (workspaceSize > 0)
                aclrtFree(workspaceAddr);
            aclDestroyScalar(scalar);
            goto cleanup_self;
        }

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
            if (workspaceSize > 0)
                aclrtFree(workspaceAddr);
            aclDestroyScalar(scalar);
            goto cleanup_self;
        }

        int64_t outSize = GetShapeSize(selfShape);
        std::vector<T> resultData(outSize);
        ret = aclrtMemcpy(
            resultData.data(), outSize * sizeof(T), selfDeviceAddr, outSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            TEST_FAIL();
        } else {
            bool pass = true;
            for (size_t i = 0; i < resultData.size(); i++) {
                if (resultData[i] != selfData[i] * scalarValue) {
                    pass = false;
                    break;
                }
            }
            if (pass) {
                TEST_PASS();
            } else {
                TEST_FAIL();
            }
        }
        if (workspaceSize > 0)
            aclrtFree(workspaceAddr);
        aclDestroyScalar(scalar);
    }

cleanup_self:
    aclDestroyTensor(selfRef);
    aclrtFree(selfDeviceAddr);
}

// ============== 主函数 ==============
int main()
{
    LOG_PRINT("\n========== Mul Operator Full Test Suite ==========\n\n");

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 1. 原始 float32
    LOG_PRINT("--- 1. Original float32 ---\n");
    {
        std::vector<int64_t> shape = {4, 2};
        std::vector<float> selfData = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<float> otherData = {1, 1, 1, 2, 2, 2, 3, 3};
        RunSingleTest<float>(selfData, shape, otherData, shape, ACL_FLOAT, stream, "float32_original");
    }

    // 2. 不同数据类型
    LOG_PRINT("\n--- 2. Different dtypes ---\n");
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int32_t> selfData = {1, 2, 3, 4};
        std::vector<int32_t> otherData = {5, 6, 7, 8};
        RunSingleTest<int32_t>(selfData, shape, otherData, shape, ACL_INT32, stream, "int32");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int64_t> selfData = {10, 20, 30, 40};
        std::vector<int64_t> otherData = {2, 3, 4, 5};
        RunSingleTest<int64_t>(selfData, shape, otherData, shape, ACL_INT64, stream, "int64");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int8_t> selfData = {1, 2, 3, 4};
        std::vector<int8_t> otherData = {2, 2, 2, 2};
        RunSingleTest<int8_t>(selfData, shape, otherData, shape, ACL_INT8, stream, "int8");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<uint8_t> selfData = {10, 20, 30, 40};
        std::vector<uint8_t> otherData = {2, 2, 2, 2};
        RunSingleTest<uint8_t>(selfData, shape, otherData, shape, ACL_UINT8, stream, "uint8");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int16_t> selfData = {100, 200, 300, 400};
        std::vector<int16_t> otherData = {2, 3, 4, 5};
        RunSingleTest<int16_t>(selfData, shape, otherData, shape, ACL_INT16, stream, "int16");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<double> selfData = {1.5, 2.5, 3.5, 4.5};
        std::vector<double> otherData = {2.0, 2.0, 2.0, 2.0};
        RunSingleTest<double>(selfData, shape, otherData, shape, ACL_DOUBLE, stream, "double");
    }

    // 3. 广播 Shape
    LOG_PRINT("\n--- 3. Broadcast shapes ---\n");
    {
        std::vector<int64_t> selfShape = {2, 3};
        std::vector<int64_t> otherShape = {3};
        std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
        std::vector<float> otherData = {10, 20, 30};
        RunSingleTest<float>(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "broadcast_2x3_x_3");
    }
    {
        std::vector<int64_t> selfShape = {2, 1};
        std::vector<int64_t> otherShape = {1, 3};
        std::vector<float> selfData = {1, 2};
        std::vector<float> otherData = {10, 20, 30};
        RunSingleTest<float>(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "broadcast_2x1_x_1x3");
    }
    {
        // 标量广播
        std::vector<int64_t> selfShape = {};
        std::vector<int64_t> otherShape = {2, 3};
        std::vector<float> selfData = {5.0f};
        std::vector<float> otherData = {1, 2, 3, 4, 5, 6};
        RunSingleTest<float>(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "scalar_broadcast");
    }
    {
        // 三维广播
        std::vector<int64_t> selfShape = {2, 1, 3};
        std::vector<int64_t> otherShape = {1, 4, 3};
        std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
        std::vector<float> otherData(12, 2.0f);
        RunSingleTest<float>(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "broadcast_3d");
    }
    {
        // 高维广播 (5维)
        std::vector<int64_t> selfShape = {1, 2, 1, 3, 4};
        std::vector<int64_t> otherShape = {2, 3, 1};
        std::vector<float> selfData(24, 3.0f);
        std::vector<float> otherData(6, 2.0f);
        RunSingleTest<float>(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "broadcast_5d");
    }

    // 4. aclnnMuls API
    LOG_PRINT("\n--- 4. aclnnMuls API ---\n");
    {
        std::vector<int64_t> shape = {3, 2};
        std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        RunMulsTest<float>(selfData, shape, 2.5f, ACL_FLOAT, stream, "muls_float32");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int32_t> selfData = {10, 20, 30, 40};
        RunMulsTest<int32_t>(selfData, shape, 3, ACL_INT32, stream, "muls_int32");
    }

    // 5. aclnnInplaceMuls API
    LOG_PRINT("\n--- 5. aclnnInplaceMuls API ---\n");
    {
        std::vector<int64_t> shape = {3, 2};
        std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        RunInplaceMulsTest<float>(selfData, shape, 3.0f, ACL_FLOAT, stream, "inplace_muls_float32");
    }

    // 6. 边界值
    LOG_PRINT("\n--- 6. Boundary values ---\n");
    {
        std::vector<int64_t> shape = {3};
        std::vector<float> selfData = {0.0f, -1.0f, 100.0f};
        std::vector<float> otherData = {5.0f, -2.0f, 0.0f};
        RunSingleTest<float>(selfData, shape, otherData, shape, ACL_FLOAT, stream, "zero_negative");
    }

    // 7. FLOAT16 测试（如果模拟器支持）
    LOG_PRINT("\n--- 7. FLOAT16 tests (may fail if unsupported) ---\n");
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<aclFloat16> selfData = {aclFloat16(1.0f), aclFloat16(2.0f), aclFloat16(3.0f), aclFloat16(4.0f)};
        std::vector<aclFloat16> otherData = {aclFloat16(2.0f), aclFloat16(2.0f), aclFloat16(2.0f), aclFloat16(2.0f)};
        RunSingleTest<aclFloat16>(selfData, shape, otherData, shape, ACL_FLOAT16, stream, "float16_mul");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<aclFloat16> selfData = {aclFloat16(1.0f), aclFloat16(2.0f), aclFloat16(3.0f), aclFloat16(4.0f)};
        RunMulsTest<aclFloat16>(selfData, shape, aclFloat16(2.0f), ACL_FLOAT16, stream, "float16_muls");
    }

    // 8. 异常输入（覆盖错误分支）
    LOG_PRINT("\n--- 8. Exception cases ---\n");
    {
        // 不可广播的 shape
        std::vector<int64_t> selfShape = {2, 3};
        std::vector<int64_t> otherShape = {4};
        std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
        std::vector<float> otherData = {1, 2, 3, 4};
        RunSingleTest<float>(
            selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "incompatible_shape", false);
    }

    LOG_PRINT("\n========== Test Summary ==========\n");
    LOG_PRINT("Total: %d, Passed: %d, Failed: %d\n", g_total_tests, g_passed_tests, g_failed_tests);
    LOG_PRINT("===================================\n\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (g_failed_tests == 0) ? 0 : 1;
}
