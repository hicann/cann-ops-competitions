#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

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
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// ============ 测试用例1: Basic Add (float32) ============
// 测试目标: 验证 float32 下的基本加法 + alpha 缩放功能
// 覆盖维度: dtype=float32, alpha=1.0, 同 shape, 正常数值范围
bool TestBasicAddFloat32(aclrtStream stream)
{
    LOG_PRINT("\n========== Test case 1: Basic Add (float32) ==========\n");

    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    float alphaValue = 1.0f;
    std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> otherHostData = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
    std::vector<float> outHostData(8, 0.0f);

    auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    if (ret != 0) return false;
    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    if (ret != 0) return false;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (alpha == nullptr) return false;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    if (ret != 0) return false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);
        return false;
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
            return false;
        }
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAdd failed. ERROR: %d\n", ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return false;
    }

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                      outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  copy result from device to host failed. ERROR: %d\n", ret);
        return false;
    }

    // CPU 参考计算 + 验证
    constexpr double atol = 1e-6;
    constexpr double rtol = 1e-6;
    bool allPass = true;
    std::vector<double> expected(size);
    LOG_PRINT("  Data type: float32, alpha: %.1f\n", alphaValue);
    LOG_PRINT("  Shape: [%ld, %ld]\n", selfShape[0], selfShape[1]);
    LOG_PRINT("  Expected (self + alpha*other):\n    ");
    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<double>(selfHostData[i]) + alphaValue * static_cast<double>(otherHostData[i]);
        LOG_PRINT("%.6f ", expected[i]);
    }
    LOG_PRINT("\n  Actual:\n    ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%.6f ", resultData[i]);
    }
    LOG_PRINT("\n  Error:\n    ");
    for (int64_t i = 0; i < size; i++) {
        double absErr = std::fabs(static_cast<double>(resultData[i]) - expected[i]);
        double relErr = std::fabs(expected[i]) > 1e-10 ? absErr / std::fabs(expected[i]) : absErr;
        LOG_PRINT("%.6e ", absErr);
        if (absErr > atol + rtol * std::fabs(expected[i])) {
            allPass = false;
        }
    }
    LOG_PRINT("\n");

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
    } else {
        LOG_PRINT("  [FAIL]\n");
    }
    return allPass;
}

// ============ 测试用例2: Alpha Scale + Negative (float32) ============
// 测试目标: 验证 alpha 缩放因子为负数和浮点值时的正确性
// 覆盖维度: dtype=float32, alpha=-2.5, 同 shape, 含负数输入
bool TestAlphaScaleFloat32(aclrtStream stream)
{
    LOG_PRINT("\n========== Test case 2: Alpha Scale + Negative (float32) ==========\n");

    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    float alphaValue = -2.5f;
    std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> otherHostData = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> outHostData(8, 0.0f);

    auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    if (ret != 0) return false;
    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    if (ret != 0) return false;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (alpha == nullptr) return false;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    if (ret != 0) return false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);
        return false;
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
            return false;
        }
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAdd failed. ERROR: %d\n", ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return false;
    }

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                      outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  copy result from device to host failed. ERROR: %d\n", ret);
        return false;
    }

    // CPU 参考计算 + 验证
    constexpr double atol = 1e-6;
    constexpr double rtol = 1e-6;
    bool allPass = true;
    std::vector<double> expected(size);
    LOG_PRINT("  Data type: float32, alpha: %.1f\n", alphaValue);
    LOG_PRINT("  Shape: [%ld, %ld]\n", selfShape[0], selfShape[1]);
    LOG_PRINT("  Expected (self + alpha*other):\n    ");
    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<double>(selfHostData[i]) + alphaValue * static_cast<double>(otherHostData[i]);
        LOG_PRINT("%.6f ", expected[i]);
    }
    LOG_PRINT("\n  Actual:\n    ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%.6f ", resultData[i]);
    }
    LOG_PRINT("\n  Error:\n    ");
    for (int64_t i = 0; i < size; i++) {
        double absErr = std::fabs(static_cast<double>(resultData[i]) - expected[i]);
        double relErr = std::fabs(expected[i]) > 1e-10 ? absErr / std::fabs(expected[i]) : absErr;
        LOG_PRINT("%.6e ", absErr);
        if (absErr > atol + rtol * std::fabs(expected[i])) {
            allPass = false;
        }
    }
    LOG_PRINT("\n");

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
    } else {
        LOG_PRINT("  [FAIL]\n");
    }
    return allPass;
}

// ============ 测试用例3: Int32 整数类型 ============
// 测试目标: 验证 int32 整数加法的精确匹配
// 覆盖维度: dtype=int32, alpha=1.0, 同 shape, 精确匹配（无浮点容差）
bool TestInt32(aclrtStream stream)
{
    LOG_PRINT("\n========== Test case 3: Int32 Data Type ==========\n");

    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    float alphaValue = 1.0f;
    std::vector<int32_t> selfHostData = {100, 200, -300, 400, -500, 600, -700, 800};
    std::vector<int32_t> otherHostData = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<int32_t> outHostData(8, 0);

    auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    if (ret != 0) return false;
    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT32, &other);
    if (ret != 0) return false;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (alpha == nullptr) return false;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    if (ret != 0) return false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);
        return false;
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
            return false;
        }
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAdd failed. ERROR: %d\n", ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return false;
    }

    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                      outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  copy result from device to host failed. ERROR: %d\n", ret);
        return false;
    }

    // CPU 参考: 精确匹配，int32 无容差
    bool allPass = true;
    std::vector<int32_t> expected(size);
    LOG_PRINT("  Data type: int32, alpha: %.1f\n", alphaValue);
    LOG_PRINT("  Shape: [%ld, %ld]\n", selfShape[0], selfShape[1]);
    LOG_PRINT("  Expected (self + alpha*other):\n    ");
    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<int32_t>(selfHostData[i] + alphaValue * otherHostData[i]);
        LOG_PRINT("%d ", expected[i]);
    }
    LOG_PRINT("\n  Actual:\n    ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%d ", resultData[i]);
    }
    LOG_PRINT("\n  Error:\n    ");
    for (int64_t i = 0; i < size; i++) {
        int64_t absErr = std::llabs(static_cast<int64_t>(resultData[i]) - static_cast<int64_t>(expected[i]));
        LOG_PRINT("%ld ", static_cast<long>(absErr));
        if (resultData[i] != expected[i]) {
            allPass = false;
        }
    }
    LOG_PRINT("\n");

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
    } else {
        LOG_PRINT("  [FAIL]\n");
    }
    return allPass;
}

// ============ 测试用例4: aclnnInplaceAdd ============
// 测试目标: 验证原地加法（self += alpha*other）的正确性
// 覆盖维度: API=aclnnInplaceAdd, dtype=float32, 原地修改
bool TestInplaceAdd(aclrtStream stream)
{
    LOG_PRINT("\n========== Test case 4: aclnnInplaceAdd ==========\n");

    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;

    float alphaValue = 2.0f;
    std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> otherHostData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> outHostData(8, 0.0f);

    auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    if (ret != 0) return false;
    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    if (ret != 0) return false;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (alpha == nullptr) return false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret);
        return false;
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
            return false;
        }
    }
    ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnInplaceAdd failed. ERROR: %d\n", ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return false;
    }

    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                      selfDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  copy result from device to host failed. ERROR: %d\n", ret);
        return false;
    }

    // CPU 参考: self += alpha * other
    constexpr double atol = 1e-6;
    constexpr double rtol = 1e-6;
    bool allPass = true;
    std::vector<double> expected(size);
    LOG_PRINT("  API: aclnnInplaceAdd, Data type: float32, alpha: %.1f\n", alphaValue);
    LOG_PRINT("  Shape: [%ld, %ld]\n", selfShape[0], selfShape[1]);
    LOG_PRINT("  Expected (self += alpha*other):\n    ");
    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<double>(selfHostData[i]) + alphaValue * static_cast<double>(otherHostData[i]);
        LOG_PRINT("%.6f ", expected[i]);
    }
    LOG_PRINT("\n  Actual (modified in-place):\n    ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%.6f ", resultData[i]);
    }
    LOG_PRINT("\n  Error:\n    ");
    for (int64_t i = 0; i < size; i++) {
        double absErr = std::fabs(static_cast<double>(resultData[i]) - expected[i]);
        LOG_PRINT("%.6e ", absErr);
        if (absErr > atol + rtol * std::fabs(expected[i])) {
            allPass = false;
        }
    }
    LOG_PRINT("\n");

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
    } else {
        LOG_PRINT("  [FAIL]\n");
    }
    return allPass;
}

// ============ 测试用例5: aclnnAddV3 (scalar + tensor) ============
// 测试目标: 验证 V3 API（标量+张量）的正确性
// 覆盖维度: API=aclnnAddV3, dtype=float32, self为标量, other为张量
bool TestAddV3(aclrtStream stream)
{
    LOG_PRINT("\n========== Test case 5: aclnnAddV3 (scalar + tensor) ==========\n");

    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclScalar* selfScalar = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    float selfScalarValue = 10.0f;
    float alphaValue = 1.0f;
    std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> outHostData(8, 0.0f);

    selfScalar = aclCreateScalar(&selfScalarValue, aclDataType::ACL_FLOAT);
    if (selfScalar == nullptr) return false;

    auto ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    if (ret != 0) return false;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (alpha == nullptr) return false;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    if (ret != 0) return false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", ret);
        return false;
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
            return false;
        }
    }
    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnAddV3 failed. ERROR: %d\n", ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return false;
    }

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                      outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  copy result from device to host failed. ERROR: %d\n", ret);
        return false;
    }

    // CPU 参考: scalar + alpha * tensor
    constexpr double atol = 1e-6;
    constexpr double rtol = 1e-6;
    bool allPass = true;
    std::vector<double> expected(size);
    LOG_PRINT("  API: aclnnAddV3, Data type: float32, alpha: %.1f, scalar: %.1f\n", alphaValue, selfScalarValue);
    LOG_PRINT("  Shape: other=[%ld,%ld], out=[%ld,%ld]\n", otherShape[0], otherShape[1], outShape[0], outShape[1]);
    LOG_PRINT("  Expected (scalar + alpha*tensor):\n    ");
    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<double>(selfScalarValue) + alphaValue * static_cast<double>(otherHostData[i]);
        LOG_PRINT("%.6f ", expected[i]);
    }
    LOG_PRINT("\n  Actual:\n    ");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%.6f ", resultData[i]);
    }
    LOG_PRINT("\n  Error:\n    ");
    for (int64_t i = 0; i < size; i++) {
        double absErr = std::fabs(static_cast<double>(resultData[i]) - expected[i]);
        LOG_PRINT("%.6e ", absErr);
        if (absErr > atol + rtol * std::fabs(expected[i])) {
            allPass = false;
        }
    }
    LOG_PRINT("\n");

    aclDestroyScalar(selfScalar);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclDestroyTensor(out);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (allPass) {
        LOG_PRINT("  [PASS]\n");
    } else {
        LOG_PRINT("  [FAIL]\n");
    }
    return allPass;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 执行测试用例
    int passed = 0;
    int total = 0;

    // 测试用例1: Basic Add (float32)
    total++;
    if (TestBasicAddFloat32(stream)) passed++;

    // 测试用例2: Alpha Scale + Negative (float32)
    total++;
    if (TestAlphaScaleFloat32(stream)) passed++;

    // 测试用例3: Int32 整数类型
    total++;
    if (TestInt32(stream)) passed++;

    // 测试用例4: aclnnInplaceAdd
    total++;
    if (TestInplaceAdd(stream)) passed++;

    // 测试用例5: aclnnAddV3 (scalar + tensor)
    total++;
    if (TestAddV3(stream)) passed++;

    // 3. （固定写法）同步等待所有任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 4. 汇总输出
    LOG_PRINT("\n========================================\n");
    LOG_PRINT("Summary: %d passed, %d failed out of %d test cases\n", passed, total - passed, total);
    LOG_PRINT("========================================\n");

    // 5. （固定写法）释放Device资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (passed == total) ? 0 : 1;
}