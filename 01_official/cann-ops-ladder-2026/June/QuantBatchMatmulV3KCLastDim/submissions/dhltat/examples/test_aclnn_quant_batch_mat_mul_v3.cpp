#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_quant_batch_mat_mul_v3.h"

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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入 tensor
    aclTensor* x1 = nullptr;
    void* x1DeviceAddr = nullptr;
    aclTensor* x2 = nullptr;
    void* x2DeviceAddr = nullptr;
    aclTensor* scale = nullptr;
    void* scaleDeviceAddr = nullptr;
    aclTensor* pertokenScale = nullptr;
    void* pertokenScaleDeviceAddr = nullptr;
    std::vector<int64_t> x1Shape = {128, 65536};
    std::vector<int8_t> x1HostData(8388608, 1);
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> x2Shape = {65536, 96};
    std::vector<int8_t> x2HostData(6291456, 1);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> scaleShape = {96};
    std::vector<float> scaleHostData(96, 1);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> pertokenScaleShape = {128};
    std::vector<float> pertokenScaleHostData(128, 1);
    ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {128, 96};
    std::vector<float> outHostData(12288, 0);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用 aclnnQuantBatchMatMulV3 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnQuantBatchMatMulV3GetWorkspaceSize(x1, x2, scale, pertokenScale, false, false, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatMulV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnQuantBatchMatMulV3 第二段接口
    ret = aclnnQuantBatchMatMulV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatMulV3 failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 释放资源
    aclDestroyTensor(x1);
    aclrtFree(x1DeviceAddr);
    aclDestroyTensor(x2);
    aclrtFree(x2DeviceAddr);
    aclDestroyTensor(scale);
    aclrtFree(scaleDeviceAddr);
    aclDestroyTensor(pertokenScale);
    aclrtFree(pertokenScaleDeviceAddr);
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
