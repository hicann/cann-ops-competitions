// ============================================================================
// Mul算子测试代码
// 参赛者: sysu_ruangongxiaodui (中山大学软工小队)
// 赛事: CANN算子测试大赛预选赛
// 功能: 测试aclnnMul, aclnnMuls, aclnnInplaceMul, aclnnInplaceMuls算子
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {              \
            return_expr;            \
        }                           \
    } while (0)

#define LOG_PRINT(message, ...)    \
    do {                           \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// 计算shape的元素总数
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// AscendCL环境初始化函数
int Init(int32_t deviceId, aclrtStream* stream) {
    // 初始化ACL库
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // 设置计算设备
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    // 创建计算流
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 创建ACL Tensor的模板函数
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                     aclDataType dataType, aclTensor** tensor) {
    // 计算所需内存大小
    auto size = GetShapeSize(shape) * sizeof(T);
    // 申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 将Host侧数据拷贝到Device侧
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算tensor的strides信息
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 创建ACL Tensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ============================================================================
// 【新增功能】结果验证函数
// ============================================================================
// 参考文档: 题目A_Mul算子测试_1775911961492.md 第40-44行
// 要求: "为每个测试用例在CPU端独立计算期望值,并与算子输出进行数值比对"
// ============================================================================

/**
 * 【函数说明】比较单个结果值
 * 
 * 【为什么需要这个函数】
 * 1. 浮点数计算存在精度误差,不能直接用 == 比较
 * 2. 题目要求使用容差比较: |actual - expected| ≤ atol + rtol × |expected|
 * 
 * 【参数说明】
 * @param actual: NPU计算的实际结果
 * @param expected: CPU计算的期望结果
 * @param atol: 绝对容差(Absolute Tolerance)
 * @param rtol: 相对容差(Relative Tolerance)
 * 
 * 【容差设置参考】题目要求第44行:
 * - FLOAT32: 1e-5
 * - FLOAT16: 1e-3
 * - BF16: 1e-2
 * - 整数: 精确匹配
 */
template <typename T>
bool CompareResult(const T& actual, const T& expected, double atol, double rtol) {
  // 计算差值的绝对值
  double diff = std::abs(static_cast<double>(actual) - static_cast<double>(expected));
  // 计算容差: atol + rtol × |expected|
  double tolerance = atol + rtol * std::abs(static_cast<double>(expected));
  // 判断是否在容差范围内
  return diff <= tolerance;
}

/**
 * 【特化版本】int32_t精确比较
 * 
 * 【为什么需要特化】
 * 整数运算不存在浮点精度问题,应该精确匹配
 * 如果不匹配,说明计算逻辑错误
 */
template <>
bool CompareResult<int32_t>(const int32_t& actual, const int32_t& expected, double atol, double rtol) {
  return actual == expected;  // 整数必须完全相等
}

/**
 * 【特化版本】int64_t精确比较
 */
template <>
bool CompareResult<int64_t>(const int64_t& actual, const int64_t& expected, double atol, double rtol) {
  return actual == expected;
}

/**
 * 【函数说明】批量验证结果
 * 
 * 【为什么需要这个函数】
 * 1. 题目要求每个测试用例输出 [PASS] 或 [FAIL]
 * 2. 需要验证所有元素,发现第一个不匹配的位置
 * 
 * 【输出格式参考】题目要求第54行:
 * "每个测试用例输出 [PASS] 或 [FAIL]"
 */
template <typename T>
bool VerifyResult(const std::vector<T>& actual, const std::vector<T>& expected, 
                  const std::string& testName, double atol, double rtol) {
  // 检查大小是否一致
  if (actual.size() != expected.size()) {
    LOG_PRINT("[FAIL] %s: size mismatch (actual=%zu, expected=%zu)\n", 
              testName.c_str(), actual.size(), expected.size());
    return false;
  }
  
  // 逐个元素比较
  for (size_t i = 0; i < actual.size(); i++) {
    if (!CompareResult(actual[i], expected[i], atol, rtol)) {
      // 发现不匹配,输出详细信息
      LOG_PRINT("[FAIL] %s: mismatch at index %zu, actual=%.6f, expected=%.6f\n",
               testName.c_str(), i, static_cast<double>(actual[i]), static_cast<double>(expected[i]));
      return false;
    }
  }
  
  // 所有元素都匹配,输出成功
  LOG_PRINT("[PASS] %s\n", testName.c_str());
  return true;
}

// ============================================================================
// 【新增功能】广播(Broadcast)辅助函数
// ============================================================================
// 参考文档: 题目A_Mul算子测试_1775911961492.md 第7-8行
// 算子定义: "当两个输入的shape不一致时,按广播规则对齐后逐元素计算"
// 
// 【广播规则说明】
// 1. 从右向左对齐维度
// 2. 维度大小为1的可以广播为任意大小
// 3. 维度大小相同则直接使用
// 4. 其他情况无法广播
// 
// 【示例】
// [2, 3, 4] × [4]      → [2, 3, 4]  (尾部对齐)
// [3, 1] × [1, 4]      → [3, 4]     (中间维度广播)
// [5] × [1]            → [5]        (标量广播)
// ============================================================================

/**
 * 【函数说明】计算广播后的输出shape
 * 
 * 【实现思路】
 * 1. 从右向左逐维比较
 * 2. 取每维的最大值作为输出维度
 * 
 * 【示例】
 * BroadcastShape({2, 3}, {3}) = {2, 3}
 * BroadcastShape({3, 1}, {1, 4}) = {3, 4}
 */
std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& shape1, 
                                     const std::vector<int64_t>& shape2) {
  std::vector<int64_t> result;
  int64_t maxDim = std::max(shape1.size(), shape2.size());
  
  // 从右向左对齐计算
  for (int64_t i = 0; i < maxDim; i++) {
    // 获取两个shape在当前维度的值(超出范围视为1)
    int64_t dim1 = (i < static_cast<int64_t>(shape1.size())) ? 
                   shape1[shape1.size() - 1 - i] : 1;
    int64_t dim2 = (i < static_cast<int64_t>(shape2.size())) ? 
                   shape2[shape2.size() - 1 - i] : 1;
    // 取最大值作为输出维度
    int64_t outDim = std::max(dim1, dim2);
    result.insert(result.begin(), outDim);
  }
  return result;
}

/**
 * 【函数说明】计算广播索引映射
 * 
 * 【为什么需要这个函数】
 * 当shape不同时,需要将输出索引映射回输入索引
 * 
 * 【示例】
 * 输出shape=[2,3], 输入shape=[3]
 * 输出索引0 → 输入索引0
 * 输出索引1 → 输入索引1
 * 输出索引2 → 输入索引2
 * 输出索引3 → 输入索引0  (广播)
 * 输出索引4 → 输入索引1  (广播)
 * 输出索引5 → 输入索引2  (广播)
 * 
 * @param outIdx: 输出tensor的一维索引
 * @param outShape: 输出tensor的shape
 * @param inShape: 输入tensor的shape
 * @return: 对应输入tensor的一维索引
 */
int64_t BroadcastIndex(int64_t outIdx, const std::vector<int64_t>& outShape,
                       const std::vector<int64_t>& inShape) {
  if (inShape.size() == 0) return 0;
  
  // 步骤1: 将一维索引转换为多维坐标
  // 例如: outIdx=5, outShape=[2,3] → outCoords=[1,2]
  std::vector<int64_t> outCoords(outShape.size());
  int64_t temp = outIdx;
  for (int64_t i = outShape.size() - 1; i >= 0; i--) {
    outCoords[i] = temp % outShape[i];
    temp /= outShape[i];
  }
  
  // 步骤2: 映射到输入shape的坐标
  // 广播维度(大小为1)的坐标固定为0
  int64_t inIdx = 0;
  int64_t dimOffset = outShape.size() - inShape.size();
  for (size_t i = 0; i < inShape.size(); i++) {
    inIdx *= inShape[i];
    if (inShape[i] == 1) {
      // 广播维度,坐标为0
      inIdx += 0;
    } else {
      // 非广播维度,使用输出坐标
      inIdx += outCoords[i + dimOffset];
    }
  }
  return inIdx;
}

/**
 * 【函数说明】计算期望结果(支持广播)
 * 
 * 【实现思路】
 * 1. 遍历输出的每个元素
 * 2. 通过广播索引映射找到对应的输入元素
 * 3. 执行乘法计算
 * 
 * 【参考】题目要求第40行: "在CPU端独立计算期望值"
 */
template <typename T>
std::vector<T> ComputeExpectedMul(
    const std::vector<T>& selfData, const std::vector<int64_t>& selfShape,
    const std::vector<T>& otherData, const std::vector<int64_t>& otherShape,
    const std::vector<int64_t>& outShape) {
  
  std::vector<T> expected(GetShapeSize(outShape));
  
  // 遍历输出的每个元素
  for (int64_t i = 0; i < GetShapeSize(outShape); i++) {
    // 计算广播后的索引
    int64_t selfIdx = BroadcastIndex(i, outShape, selfShape);
    int64_t otherIdx = BroadcastIndex(i, outShape, otherShape);
    // 执行乘法: y = x1 × x2
    expected[i] = selfData[selfIdx] * otherData[otherIdx];
  }
  return expected;
}

// ============================================================================
// 【新增功能】测试用例封装函数
// ============================================================================
// 【为什么需要这个函数】
// 1. 避免重复代码,每个测试用例都需要相同的步骤
// 2. 统一错误处理和资源清理
// 3. 自动化测试流程
// 
// 【测试流程】
// 1. 创建输入tensor
// 2. 调用Mul算子
// 3. 获取计算结果
// 4. 计算期望值
// 5. 验证结果
// 6. 清理资源
// ============================================================================

template <typename T>
int TestMulCase(
    const std::string& testName,                           // 测试名称
    const std::vector<T>& selfData,                        // 第一个输入数据
    const std::vector<int64_t>& selfShape,                 // 第一个输入shape
    const std::vector<T>& otherData,                       // 第二个输入数据
    const std::vector<int64_t>& otherShape,                // 第二个输入shape
    aclDataType dataType,                                  // 数据类型
    double atol,                                           // 绝对容差
    double rtol,                                           // 相对容差
    aclrtStream stream) {                                  // 计算流

    // ===== 计算输出shape(广播机制) =====
    auto outShape = BroadcastShape(selfShape, otherShape);

    // ===== 本地变量定义 =====
    void* devAddrSelf = nullptr;
    void* devAddrOther = nullptr;
    void* devAddrOut = nullptr;
    aclTensor* tensorSelf = nullptr;
    aclTensor* tensorOther = nullptr;
    aclTensor* tensorOut = nullptr;
    uint64_t workspaceBytes = 0;
    void* workspaceBuf = nullptr;
    aclOpExecutor* executorHandle = nullptr;
    int64_t elemCount = 0;
    std::vector<T> hostResult;
    std::vector<T> expectedResult;
    bool testPassed = false;
    int funcStatus = 0;
    bool hasError = false;

    // ===== 创建输入输出tensor =====
    // 第1个输入: self
    funcStatus = CreateAclTensor(selfData, selfShape, &devAddrSelf, dataType, &tensorSelf);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create self tensor failed\n");
        return funcStatus;
    }

    // 第2个输入: other
    funcStatus = CreateAclTensor(otherData, otherShape, &devAddrOther, dataType, &tensorOther);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create other tensor failed\n");
        aclDestroyTensor(tensorSelf);
        aclrtFree(devAddrSelf);
        return funcStatus;
    }

    // 输出tensor: out
    std::vector<T> hostBuffer(GetShapeSize(outShape), 0);
    funcStatus = CreateAclTensor(hostBuffer, outShape, &devAddrOut, dataType, &tensorOut);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create out tensor failed\n");
        aclDestroyTensor(tensorSelf);
        aclDestroyTensor(tensorOther);
        aclrtFree(devAddrSelf);
        aclrtFree(devAddrOther);
        return funcStatus;
    }

    // ===== 执行Mul算子(两段式) =====
    // 步骤A: 查询所需workspace大小
    funcStatus = aclnnMulGetWorkspaceSize(tensorSelf, tensorOther, tensorOut, &workspaceBytes, &executorHandle);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", funcStatus);
        hasError = true;
    }

    // 步骤B: 申请workspace内存
    if (!hasError && workspaceBytes > 0) {
        funcStatus = aclrtMalloc(&workspaceBuf, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Allocate workspace failed, size=%lu\n", workspaceBytes);
            hasError = true;
        }
    }

    // 步骤C: 执行算子
    if (!hasError) {
        funcStatus = aclnnMul(workspaceBuf, workspaceBytes, executorHandle, stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] aclnnMul execute failed, ret=%d\n", funcStatus);
            hasError = true;
        }
    }

    // 步骤D: 等待算子执行完成
    if (!hasError) {
        funcStatus = aclrtSynchronizeStream(stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Stream sync failed, ret=%d\n", funcStatus);
            hasError = true;
        }
    }

    // 步骤E: 从设备拷贝结果到主机
    if (!hasError) {
        elemCount = GetShapeSize(outShape);
        hostResult.resize(elemCount, 0);
        funcStatus = aclrtMemcpy(hostResult.data(), hostResult.size() * sizeof(T), devAddrOut,
                                  elemCount * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Copy result to host failed\n");
            hasError = true;
        }
    }

    // ===== 验证结果 =====
    if (!hasError) {
        expectedResult = ComputeExpectedMul(selfData, selfShape, otherData, otherShape, outShape);
        testPassed = VerifyResult(hostResult, expectedResult, testName, atol, rtol);
    }

    // ===== 释放workspace =====
    if (workspaceBytes > 0 && workspaceBuf != nullptr) {
        aclrtFree(workspaceBuf);
    }

    // ===== 释放tensor和内存 =====
    if (tensorSelf != nullptr) aclDestroyTensor(tensorSelf);
    if (tensorOther != nullptr) aclDestroyTensor(tensorOther);
    if (tensorOut != nullptr) aclDestroyTensor(tensorOut);
    if (devAddrSelf != nullptr) aclrtFree(devAddrSelf);
    if (devAddrOther != nullptr) aclrtFree(devAddrOther);
    if (devAddrOut != nullptr) aclrtFree(devAddrOut);

    return testPassed ? 0 : 1;
}

// ============================================================================
// 【新增】aclnnMuls测试函数 (tensor × scalar)
// ============================================================================
template <typename T>
int TestMulsCase(
    const std::string& testName,
    const std::vector<T>& selfData,
    const std::vector<int64_t>& selfShape,
    T scalarValue,
    aclDataType dataType,
    double atol,
    double rtol,
    aclrtStream stream) {

// ===== 本地变量定义 =====
    void* devAddrSelf = nullptr;
    void* devAddrOut = nullptr;
    aclTensor* tensorSelf = nullptr;
    aclTensor* tensorOut = nullptr;
    uint64_t workspaceBytes = 0;
    void* workspaceBuf = nullptr;
    aclOpExecutor* executorHandle = nullptr;
    int64_t elemCount = 0;
    std::vector<T> hostResult;
    std::vector<T> expectedResult;
    bool testPassed = false;
    int funcStatus = 0;
    bool hasError = false;
    aclScalar* scalarObj = nullptr;

    // 创建输入tensor (self)
    funcStatus = CreateAclTensor(selfData, selfShape, &devAddrSelf, dataType, &tensorSelf);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create self tensor failed\n");
        return funcStatus;
    }

    // 创建输出tensor (out)
    std::vector<T> hostBuffer(selfData.size(), 0);
    funcStatus = CreateAclTensor(hostBuffer, selfShape, &devAddrOut, dataType, &tensorOut);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create out tensor failed\n");
        aclDestroyTensor(tensorSelf);
        aclrtFree(devAddrSelf);
        return funcStatus;
    }

    // 创建scalar标量对象
    scalarObj = aclCreateScalar(&scalarValue, dataType);
    if (scalarObj == nullptr) {
        LOG_PRINT("[ERROR] Create scalar failed\n");
        hasError = true;
    }

    // ===== 执行Muls算子(两段式) =====
    // 步骤A: 查询workspace
    if (!hasError) {
        funcStatus = aclnnMulsGetWorkspaceSize(tensorSelf, scalarObj, tensorOut, &workspaceBytes, &executorHandle);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", funcStatus);
            hasError = true;
        }
    }

    // 步骤B: 申请workspace
    if (!hasError && workspaceBytes > 0) {
        funcStatus = aclrtMalloc(&workspaceBuf, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Allocate workspace failed\n");
            hasError = true;
        }
    }

    // 步骤C: 执行算子
    if (!hasError) {
        funcStatus = aclnnMuls(workspaceBuf, workspaceBytes, executorHandle, stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] aclnnMuls execute failed, ret=%d\n", funcStatus);
            hasError = true;
        }
    }

    // 步骤D: 等待完成
    if (!hasError) {
        funcStatus = aclrtSynchronizeStream(stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Stream sync failed\n");
            hasError = true;
        }
    }

    // 步骤E: 拷贝结果
    if (!hasError) {
        elemCount = selfData.size();
        hostResult.resize(elemCount, 0);
        funcStatus = aclrtMemcpy(hostResult.data(), hostResult.size() * sizeof(T), devAddrOut,
                                  elemCount * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Copy result failed\n");
            hasError = true;
        }
    }

    // ===== 验证结果 =====
    if (!hasError) {
        expectedResult = ComputeExpectedMuls(selfData, selfShape, scalarValue, selfShape);
        testPassed = VerifyResult(hostResult, expectedResult, testName, atol, rtol);
    }

    // ===== 释放资源 =====
    if (workspaceBytes > 0 && workspaceBuf != nullptr) {
        aclrtFree(workspaceBuf);
    }
    if (scalarObj != nullptr) aclDestroyScalar(scalarObj);
    if (tensorSelf != nullptr) aclDestroyTensor(tensorSelf);
    if (tensorOut != nullptr) aclDestroyTensor(tensorOut);
    if (devAddrSelf != nullptr) aclrtFree(devAddrSelf);
    if (devAddrOut != nullptr) aclrtFree(devAddrOut);

    return testPassed ? 0 : 1;
}

// ============================================================================
// 【新增】aclnnInplaceMul测试函数 (原地乘tensor)
// ============================================================================
template <typename T>
int TestInplaceMulCase(
    const std::string& testName,
    const std::vector<T>& selfData,
    const std::vector<int64_t>& selfShape,
    const std::vector<T>& otherData,
    const std::vector<int64_t>& otherShape,
    aclDataType dataType,
    double atol,
    double rtol,
    aclrtStream stream) {

// ===== 本地变量定义 =====
    void* devAddrSelf = nullptr;
    void* devAddrOther = nullptr;
    aclTensor* tensorSelf = nullptr;
    aclTensor* tensorOther = nullptr;
    uint64_t workspaceBytes = 0;
    void* workspaceBuf = nullptr;
    aclOpExecutor* executorHandle = nullptr;
    int64_t elemCount = 0;
    std::vector<T> hostResult;
    std::vector<T> expectedResult;
    bool testPassed = false;
    int funcStatus = 0;
    bool hasError = false;

    // 计算输出shape（原地算子输出shape与输入相同）
    auto outShape = BroadcastShape(selfShape, otherShape);

    // 创建self tensor（原地算子会修改此tensor）
    funcStatus = CreateAclTensor(selfData, selfShape, &devAddrSelf, dataType, &tensorSelf);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create self tensor failed\n");
        return funcStatus;
    }

    // 创建other tensor
    funcStatus = CreateAclTensor(otherData, otherShape, &devAddrOther, dataType, &tensorOther);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create other tensor failed\n");
        aclDestroyTensor(tensorSelf);
        aclrtFree(devAddrSelf);
        return funcStatus;
    }

    // ===== 执行InplaceMul算子(两段式) =====
    // 步骤A: 查询workspace
    funcStatus = aclnnInplaceMulGetWorkspaceSize(tensorSelf, tensorOther, &workspaceBytes, &executorHandle);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", funcStatus);
        hasError = true;
    }

    // 步骤B: 申请workspace
    if (!hasError && workspaceBytes > 0) {
        funcStatus = aclrtMalloc(&workspaceBuf, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Allocate workspace failed\n");
            hasError = true;
        }
    }

    // 步骤C: 执行算子
    if (!hasError) {
        funcStatus = aclnnInplaceMul(workspaceBuf, workspaceBytes, executorHandle, stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] aclnnInplaceMul execute failed, ret=%d\n", funcStatus);
            hasError = true;
        }
    }

    // 步骤D: 等待完成
    if (!hasError) {
        funcStatus = aclrtSynchronizeStream(stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Stream sync failed\n");
            hasError = true;
        }
    }

    // ===== 获取结果(原地算子结果在self中) =====
    if (!hasError) {
        elemCount = GetShapeSize(selfShape);
        hostResult.resize(elemCount, 0);
        funcStatus = aclrtMemcpy(hostResult.data(), hostResult.size() * sizeof(T), devAddrSelf,
                                  elemCount * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Copy result failed\n");
            hasError = true;
        }
    }

    // ===== 验证结果 =====
    if (!hasError) {
        expectedResult = ComputeExpectedMul(selfData, selfShape, otherData, otherShape, outShape);
        testPassed = VerifyResult(hostResult, expectedResult, testName, atol, rtol);
    }

    // ===== 释放资源 =====
    if (workspaceBytes > 0 && workspaceBuf != nullptr) {
        aclrtFree(workspaceBuf);
    }
    if (tensorSelf != nullptr) aclDestroyTensor(tensorSelf);
    if (tensorOther != nullptr) aclDestroyTensor(tensorOther);
    if (devAddrSelf != nullptr) aclrtFree(devAddrSelf);
    if (devAddrOther != nullptr) aclrtFree(devAddrOther);

    return testPassed ? 0 : 1;
}

// ============================================================================
// 【新增】aclnnInplaceMuls测试函数 (原地乘scalar)
// ============================================================================
template <typename T>
int TestInplaceMulsCase(
    const std::string& testName,
    const std::vector<T>& selfData,
    const std::vector<int64_t>& selfShape,
    T scalarValue,
    aclDataType dataType,
    double atol,
    double rtol,
    aclrtStream stream) {

// ===== 本地变量定义 =====
    void* devAddrSelf = nullptr;
    aclTensor* tensorSelf = nullptr;
    uint64_t workspaceBytes = 0;
    void* workspaceBuf = nullptr;
    aclOpExecutor* executorHandle = nullptr;
    int64_t elemCount = 0;
    std::vector<T> hostResult;
    std::vector<T> expectedResult;
    bool testPassed = false;
    int funcStatus = 0;
    bool hasError = false;
    aclScalar* scalarObj = nullptr;

    // 创建self tensor（原地算子会修改此tensor）
    funcStatus = CreateAclTensor(selfData, selfShape, &devAddrSelf, dataType, &tensorSelf);
    if (funcStatus != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Create self tensor failed\n");
        return funcStatus;
    }

    // 创建scalar标量对象
    scalarObj = aclCreateScalar(&scalarValue, dataType);
    if (scalarObj == nullptr) {
        LOG_PRINT("[ERROR] Create scalar failed\n");
        hasError = true;
    }

    // ===== 执行InplaceMuls算子(两段式) =====
    // 步骤A: 查询workspace
    if (!hasError) {
        funcStatus = aclnnInplaceMulsGetWorkspaceSize(tensorSelf, scalarObj, &workspaceBytes, &executorHandle);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", funcStatus);
            hasError = true;
        }
    }

    // 步骤B: 申请workspace
    if (!hasError && workspaceBytes > 0) {
        funcStatus = aclrtMalloc(&workspaceBuf, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Allocate workspace failed\n");
            hasError = true;
        }
    }

    // 步骤C: 执行算子
    if (!hasError) {
        funcStatus = aclnnInplaceMuls(workspaceBuf, workspaceBytes, executorHandle, stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] aclnnInplaceMuls execute failed, ret=%d\n", funcStatus);
            hasError = true;
        }
    }

    // 步骤D: 等待完成
    if (!hasError) {
        funcStatus = aclrtSynchronizeStream(stream);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Stream sync failed\n");
            hasError = true;
        }
    }

    // ===== 获取结果(原地算子结果在self中) =====
    if (!hasError) {
        elemCount = selfData.size();
        hostResult.resize(elemCount, 0);
        funcStatus = aclrtMemcpy(hostResult.data(), hostResult.size() * sizeof(T), devAddrSelf,
                                  elemCount * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        if (funcStatus != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] Copy result failed\n");
            hasError = true;
        }
    }

    // ===== 计算期望值: self × scalar =====
    if (!hasError) {
        expectedResult.resize(elemCount);
        for (int64_t i = 0; i < elemCount; i++) {
            expectedResult[i] = selfData[i] * scalarValue;
        }
        testPassed = VerifyResult(hostResult, expectedResult, testName, atol, rtol);
    }

    // ===== 释放资源 =====
    if (workspaceBytes > 0 && workspaceBuf != nullptr) {
        aclrtFree(workspaceBuf);
    }
    if (scalarObj != nullptr) aclDestroyScalar(scalarObj);
    if (tensorSelf != nullptr) aclDestroyTensor(tensorSelf);
    if (devAddrSelf != nullptr) aclrtFree(devAddrSelf);

    return testPassed ? 0 : 1;
}
            LOG_PRINT("[ERROR] Allocate workspace failed\n");
            goto CLEANUP;
        }
    }

    // 阶段2: 执行计算
    ret = aclnnInplaceMuls(workBuff, workSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] aclnnInplaceMuls execute failed, ret=%d\n", ret);
        if (workSize > 0) aclrtFree(workBuff);
        goto CLEANUP;
    }

    // 等待完成
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Stream sync failed\n");
        if (workSize > 0) aclrtFree(workBuff);
        goto CLEANUP;
    }

    // ===== 获取结果(原地算子结果在self中) =====
    totalElems = selfData.size();
    outputData.resize(totalElems, 0);
    ret = aclrtMemcpy(outputData.data(), outputData.size() * sizeof(T), selfDeviceAddr,
                      totalElems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] Copy result failed\n");
        if (workSize > 0) aclrtFree(workBuff);
        goto CLEANUP;
    }

    // ===== 计算期望值: self × scalar =====
    goldenData.resize(totalElems);
    for (int64_t i = 0; i < totalElems; i++) {
        goldenData[i] = selfData[i] * scalarValue;
    }
    isPass = VerifyResult(outputData, goldenData, testName, atol, rtol);

    // 清理workspace
    if (workSize > 0) {
        aclrtFree(workBuff);
    }

CLEANUP:
    if (scalar != nullptr) aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);

    return isPass ? 0 : 1;
}

// ============================================================================
// 主测试函数
// ============================================================================

int main() {
  // 1. 初始化AscendCL环境
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  int failCount = 0;
  int totalTests = 0;
  
  LOG_PRINT("\n========== Mul算子测试开始 ==========\n\n");
  
  // ============================================================================
  // 【测试用例设计】参考题目要求第48-52行
  // 
  // 覆盖维度:
  // 1. 数据类型: FLOAT, FLOAT16, INT32, INT64
  // 2. Shape组合: 同shape, 广播, 标量
  // 3. 数值边界: 零值, 负数, 大数, 小数
  // 4. 特殊场景: 大tensor, 三维广播
  // ============================================================================
  
  // ========== 测试1: FLOAT同shape ==========
  // 【测试目的】验证最基本的FLOAT类型乘法
  // 【覆盖路径】op_api层的FLOAT类型处理路径
  totalTests++;
  failCount += TestMulCase<float>(
      "Test1_FLOAT_SameShape",                    // 测试名称
      {0, 1, 2, 3, 4, 5, 6, 7}, {4, 2},           // self: 4×2矩阵
      {1, 1, 1, 2, 2, 2, 3, 3}, {4, 2},           // other: 4×2矩阵
      aclDataType::ACL_FLOAT,                     // 数据类型: FLOAT
      1e-5, 1e-5,                                 // 容差: atol=1e-5, rtol=1e-5
      stream
  );
  
  // ========== 测试2: FLOAT广播 [2,3] × [3] ==========
  // 【测试目的】验证广播功能(尾部对齐)
  // 【广播规则】[2,3] × [3] → [2,3]
  //   [[a,b,c],    [x,y,z]   [[a*x, b*y, c*z],
  //    [d,e,f]] ×           →  [d*x, e*y, f*z]]
  // 【覆盖路径】op_host层的广播tiling策略
  totalTests++;
  failCount += TestMulCase<float>(
      "Test2_FLOAT_Broadcast_2x3_mul_3",
      {1, 2, 3, 4, 5, 6}, {2, 3},                 // self: 2×3矩阵
      {2, 3, 4}, {3},                            // other: 3维向量
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试3: FLOAT广播 [3,1] × [1,4] ==========
  // 【测试目的】验证中间维度广播
  // 【广播规则】[3,1] × [1,4] → [3,4]
  //   [[a],     [[x,y,z,w]]   [[a*x, a*y, a*z, a*w],
  //    [b],  ×              →  [b*x, b*y, b*z, b*w],
  //    [c]]                    [c*x, c*y, c*z, c*w]]
  totalTests++;
  failCount += TestMulCase<float>(
      "Test3_FLOAT_Broadcast_3x1_mul_1x4",
      {1, 2, 3}, {3, 1},                         // self: 3×1矩阵
      {1, 2, 3, 4}, {1, 4},                      // other: 1×4矩阵
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试4: FLOAT标量广播 [5] × [1] ==========
  // 【测试目的】验证标量广播
  // 【广播规则】[5] × [1] → [5]
  //   [a,b,c,d,e] × [s] → [a*s, b*s, c*s, d*s, e*s]
  totalTests++;
  failCount += TestMulCase<float>(
      "Test4_FLOAT_ScalarBroadcast",
      {1, 2, 3, 4, 5}, {5},                      // self: 5维向量
      {10}, {1},                                 // other: 标量(1维)
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试5: FLOAT零值测试 ==========
  // 【测试目的】验证零值乘法的正确性
  // 【数学性质】0 × x = 0
  totalTests++;
  failCount += TestMulCase<float>(
      "Test5_FLOAT_ZeroValue",
      {0, 0, 0, 1, 2, 3}, {2, 3},
      {100, 200, 300, 400, 500, 600}, {2, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试6: FLOAT负数测试 ==========
  // 【测试目的】验证负数乘法的正确性
  // 【数学性质】(-a) × (-b) = a×b, (-a) × b = -(a×b)
  totalTests++;
  failCount += TestMulCase<float>(
      "Test6_FLOAT_NegativeValue",
      {-1, -2, -3, 1, 2, 3}, {2, 3},
      {2, 2, 2, -2, -2, -2}, {2, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试7: FLOAT大数测试 ==========
  // 【测试目的】验证大数乘法的数值稳定性
  // 【注意事项】避免溢出,测试数值范围
  totalTests++;
  failCount += TestMulCase<float>(
      "Test7_FLOAT_LargeValue",
      {1e5f, 2e5f, 3e5f, 4e5f}, {2, 2},
      {1e3f, 2e3f, 3e3f, 4e3f}, {2, 2},
      aclDataType::ACL_FLOAT, 1e-2, 1e-5, stream
  );
  
  // ========== 测试8: FLOAT小数测试 ==========
  // 【测试目的】验证小数乘法的精度
  // 【注意事项】避免下溢,测试精度范围
  totalTests++;
  failCount += TestMulCase<float>(
      "Test8_FLOAT_SmallValue",
      {1e-5f, 2e-5f, 3e-5f, 4e-5f}, {2, 2},
      {1e-3f, 2e-3f, 3e-3f, 4e-3f}, {2, 2},
      aclDataType::ACL_FLOAT, 1e-10, 1e-5, stream
  );
  
  // ========== 测试9: INT32同shape ==========
  // 【测试目的】验证INT32类型乘法
  // 【覆盖路径】op_api层的INT32类型处理路径
  // 【验证方式】整数精确匹配,不容差
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test9_INT32_SameShape",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      {2, 3, 4, 5, 6, 7}, {2, 3},
      aclDataType::ACL_INT32, 0, 0, stream       // 整数容差为0
  );
  
  // ========== 测试10: INT32广播 ==========
  // 【测试目的】验证INT32类型的广播功能
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test10_INT32_Broadcast",
      {10, 20, 30, 40, 50, 60}, {2, 3},
      {2, 3, 4}, {3},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试11: INT32负数 ==========
  // 【测试目的】验证INT32负数乘法
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test11_INT32_Negative",
      {-10, -20, 10, 20}, {2, 2},
      {2, -3, -4, 5}, {2, 2},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试12: INT32零值 ==========
  // 【测试目的】验证INT32零值乘法
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test12_INT32_ZeroValue",
      {0, 1, 2, 3, 0, 5, 6, 0}, {2, 4},
      {100, 200, 300, 400, 500, 600, 700, 800}, {2, 4},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试13: FLOAT16同shape ==========
  // 【测试目的】验证FLOAT16类型乘法
  // 【覆盖路径】op_api层的FLOAT16类型处理路径
  // 【容差设置】FLOAT16精度较低,使用1e-3
  // 【参考】题目要求第44行: "FLOAT16取1e-3"
  totalTests++;
  failCount += TestMulCase<float>(
      "Test13_FLOAT16_SameShape",
      {1.5f, 2.5f, 3.5f, 4.5f}, {2, 2},
      {2.0f, 2.0f, 2.0f, 2.0f}, {2, 2},
      aclDataType::ACL_FLOAT16, 1e-3, 1e-3, stream  // FLOAT16容差1e-3
  );
  
  // ========== 测试14: FLOAT16广播 ==========
  // 【测试目的】验证FLOAT16类型的广播功能
  totalTests++;
  failCount += TestMulCase<float>(
      "Test14_FLOAT16_Broadcast",
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3},
      {2.0f, 3.0f, 4.0f}, {3},
      aclDataType::ACL_FLOAT16, 1e-3, 1e-3, stream
  );
  
  // ========== 测试15: FLOAT大tensor测试 ==========
  // 【测试目的】验证大tensor的性能和正确性
  // 【覆盖路径】op_host层的大tensor tiling策略
  totalTests++;
  std::vector<float> largeData1(100, 2.0f);  // 100个2.0
  std::vector<float> largeData2(100, 3.0f);  // 100个3.0
  failCount += TestMulCase<float>(
      "Test15_FLOAT_LargeTensor_100",
      largeData1, {100},
      largeData2, {100},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试16: INT64同shape ==========
  // 【测试目的】验证INT64类型乘法
  // 【覆盖路径】op_api层的INT64类型处理路径
  totalTests++;
  failCount += TestMulCase<int64_t>(
      "Test16_INT64_SameShape",
      {1000, 2000, 3000, 4000}, {2, 2},
      {2, 3, 4, 5}, {2, 2},
      aclDataType::ACL_INT64, 0, 0, stream
  );
  
  // ========== 测试17: FLOAT三维广播 [2,3,4] × [4] ==========
  // 【测试目的】验证三维tensor的广播功能
  // 【广播规则】[2,3,4] × [4] → [2,3,4]
  // 【覆盖路径】op_host层的高维广播tiling策略
  totalTests++;
  failCount += TestMulCase<float>(
      "Test17_FLOAT_3D_Broadcast",
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {2, 3, 4},
      {2, 2, 2, 2}, {4},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试18: FLOAT不同维度广播 [3,4] × [1] ==========
  // 【测试目的】验证不同维度的广播
  // 【广播规则】[3,4] × [1] → [3,4]
  totalTests++;
  failCount += TestMulCase<float>(
      "Test18_FLOAT_DimBroadcast",
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4},
      {5}, {1},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试19: FLOAT单位矩阵测试 ==========
  // 【测试目的】验证单位矩阵乘法
  // 【数学性质】I × A = A
  totalTests++;
  failCount += TestMulCase<float>(
      "Test19_FLOAT_Identity",
      {1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3},        // 单位矩阵
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试20: INT32大数测试 ==========
  // 【测试目的】验证INT32大数乘法
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test20_INT32_LargeValue",
      {10000, 20000, 30000, 40000}, {2, 2},
      {2, 3, 4, 5}, {2, 2},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ============================================================================
  // 【API变体测试】测试aclnnMuls, aclnnInplaceMul, aclnnInplaceMuls
  // ============================================================================
  
  // ========== 测试21: aclnnMuls FLOAT ==========
  // 【测试目的】验证aclnnMuls API (tensor × scalar)
  totalTests++;
  failCount += TestMulsCase<float>(
      "Test21_Muls_FLOAT",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      2.5f,  // scalar
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试22: aclnnMuls INT32 ==========
  totalTests++;
  failCount += TestMulsCase<int32_t>(
      "Test22_Muls_INT32",
      {10, 20, 30, 40}, {2, 2},
      3,  // scalar
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试23: aclnnMuls FLOAT16 ==========
  totalTests++;
  failCount += TestMulsCase<float>(
      "Test23_Muls_FLOAT16",
      {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2},
      0.5f,  // scalar
      aclDataType::ACL_FLOAT16, 1e-3, 1e-3, stream
  );
  
  // ========== 测试24: aclnnMuls负数 ==========
  totalTests++;
  failCount += TestMulsCase<float>(
      "Test24_Muls_Negative",
      {1, 2, 3, 4, 5}, {5},
      -2.0f,  // scalar
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试25: aclnnMuls零值 ==========
  totalTests++;
  failCount += TestMulsCase<float>(
      "Test25_Muls_Zero",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      0.0f,  // scalar
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试26: aclnnInplaceMul FLOAT ==========
  // 【测试目的】验证aclnnInplaceMul API (原地乘tensor)
  totalTests++;
  failCount += TestInplaceMulCase<float>(
      "Test26_InplaceMul_FLOAT",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      {2, 2, 2, 2, 2, 2}, {2, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试27: aclnnInplaceMul INT32 ==========
  totalTests++;
  failCount += TestInplaceMulCase<int32_t>(
      "Test27_InplaceMul_INT32",
      {10, 20, 30, 40}, {2, 2},
      {2, 3, 4, 5}, {2, 2},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试28: aclnnInplaceMul广播 ==========
  totalTests++;
  failCount += TestInplaceMulCase<float>(
      "Test28_InplaceMul_Broadcast",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      {2, 3, 4}, {3},  // 广播
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试29: aclnnInplaceMul负数 ==========
  totalTests++;
  failCount += TestInplaceMulCase<float>(
      "Test29_InplaceMul_Negative",
      {-1, -2, 3, 4}, {2, 2},
      {2, -3, -4, 5}, {2, 2},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试30: aclnnInplaceMuls FLOAT ==========
  // 【测试目的】验证aclnnInplaceMuls API (原地乘scalar)
  totalTests++;
  failCount += TestInplaceMulsCase<float>(
      "Test30_InplaceMuls_FLOAT",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      3.0f,  // scalar
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试31: aclnnInplaceMuls INT32 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<int32_t>(
      "Test31_InplaceMuls_INT32",
      {10, 20, 30, 40}, {2, 2},
      5,  // scalar
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试32: aclnnInplaceMuls FLOAT16 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<float>(
      "Test32_InplaceMuls_FLOAT16",
      {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2},
      0.5f,  // scalar
      aclDataType::ACL_FLOAT16, 1e-3, 1e-3, stream
  );
  
  // ========== 测试33: aclnnInplaceMuls负数 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<float>(
      "Test33_InplaceMuls_Negative",
      {1, 2, 3, 4, 5}, {5},
      -2.0f,  // scalar
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试34: aclnnInplaceMuls零值 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<float>(
      "Test34_InplaceMuls_Zero",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      0.0f,  // scalar
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试35: aclnnMuls INT64 ==========
  totalTests++;
  failCount += TestMulsCase<int64_t>(
      "Test35_Muls_INT64",
      {1000, 2000, 3000, 4000}, {2, 2},
      10,  // scalar
      aclDataType::ACL_INT64, 0, 0, stream
  );
  
  // ========== 测试36: aclnnInplaceMuls INT64 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<int64_t>(
      "Test36_InplaceMuls_INT64",
      {1000, 2000, 3000, 4000}, {2, 2},
      10,  // scalar
      aclDataType::ACL_INT64, 0, 0, stream
  );
  
  // ========== 测试37: aclnnInplaceMul FLOAT16 ==========
  totalTests++;
  failCount += TestInplaceMulCase<float>(
      "Test37_InplaceMul_FLOAT16",
      {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2},
      {2.0f, 2.0f, 2.0f, 2.0f}, {2, 2},
      aclDataType::ACL_FLOAT16, 1e-3, 1e-3, stream
  );
  
  // ========== 测试38: aclnnMuls大数 ==========
  totalTests++;
  failCount += TestMulsCase<float>(
      "Test38_Muls_LargeValue",
      {1e5f, 2e5f, 3e5f, 4e5f}, {2, 2},
      1e3f,  // scalar
      aclDataType::ACL_FLOAT, 1e-2, 1e-5, stream
  );
  
  // ========== 测试39: aclnnInplaceMuls大数 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<float>(
      "Test39_InplaceMuls_LargeValue",
      {1e5f, 2e5f, 3e5f, 4e5f}, {2, 2},
      1e3f,  // scalar
      aclDataType::ACL_FLOAT, 1e-2, 1e-5, stream
  );
  
  // ========== 测试40: aclnnInplaceMul广播 [3,4] × [4] ==========
  totalTests++;
  failCount += TestInplaceMulCase<float>(
      "Test40_InplaceMul_3D_Broadcast",
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4},
      {2, 2, 2, 2}, {4},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ============================================================================
  // 【新增】INT8/UINT8测试 (优化方案B)
  // ============================================================================
  
  // ========== 测试49: INT8同shape ==========
  // 【测试目的】验证INT8类型乘法
  totalTests++;
  failCount += TestMulCase<int8_t>(
      "Test49_INT8_SameShape",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      {2, 3, 4, 5, 6, 7}, {2, 3},
      aclDataType::ACL_INT8, 0, 0, stream
  );
  
  // ========== 测试50: INT8广播 ==========
  totalTests++;
  failCount += TestMulCase<int8_t>(
      "Test50_INT8_Broadcast",
      {10, 20, 30, 40, 50, 60}, {2, 3},
      {2, 3, 4}, {3},
      aclDataType::ACL_INT8, 0, 0, stream
  );
  
  // ========== 测试51: INT8负数 ==========
  totalTests++;
  failCount += TestMulCase<int8_t>(
      "Test51_INT8_Negative",
      {-10, -20, 10, 20}, {2, 2},
      {2, -3, -4, 5}, {2, 2},
      aclDataType::ACL_INT8, 0, 0, stream
  );
  
  // ========== 测试52: INT8零值 ==========
  totalTests++;
  failCount += TestMulCase<int8_t>(
      "Test52_INT8_ZeroValue",
      {0, 1, 2, 0, 4, 5}, {2, 3},
      {10, 20, 30, 40, 50, 60}, {2, 3},
      aclDataType::ACL_INT8, 0, 0, stream
  );
  
  // ========== 测试53: aclnnMuls INT8 ==========
  totalTests++;
  failCount += TestMulsCase<int8_t>(
      "Test53_Muls_INT8",
      {10, 20, 30, 40}, {2, 2},
      3,  // scalar
      aclDataType::ACL_INT8, 0, 0, stream
  );
  
  // ========== 测试54: aclnnInplaceMul INT8 ==========
  totalTests++;
  failCount += TestInplaceMulCase<int8_t>(
      "Test54_InplaceMul_INT8",
      {10, 20, 30, 40}, {2, 2},
      {2, 3, 4, 5}, {2, 2},
      aclDataType::ACL_INT8, 0, 0, stream
  );
  
  // ========== 测试55: aclnnInplaceMuls INT8 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<int8_t>(
      "Test55_InplaceMuls_INT8",
      {10, 20, 30, 40}, {2, 2},
      5,  // scalar
      aclDataType::ACL_INT8, 0, 0, stream
  );
  
  // ========== 测试56: UINT8同shape ==========
  // 【测试目的】验证UINT8类型乘法
  totalTests++;
  failCount += TestMulCase<uint8_t>(
      "Test56_UINT8_SameShape",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      {2, 3, 4, 5, 6, 7}, {2, 3},
      aclDataType::ACL_UINT8, 0, 0, stream
  );
  
  // ========== 测试57: UINT8广播 ==========
  totalTests++;
  failCount += TestMulCase<uint8_t>(
      "Test57_UINT8_Broadcast",
      {10, 20, 30, 40, 50, 60}, {2, 3},
      {2, 3, 4}, {3},
      aclDataType::ACL_UINT8, 0, 0, stream
  );
  
  // ========== 测试58: UINT8零值 ==========
  totalTests++;
  failCount += TestMulCase<uint8_t>(
      "Test58_UINT8_ZeroValue",
      {0, 1, 2, 0, 4, 5}, {2, 3},
      {10, 20, 30, 40, 50, 60}, {2, 3},
      aclDataType::ACL_UINT8, 0, 0, stream
  );
  
  // ========== 测试59: aclnnMuls UINT8 ==========
  totalTests++;
  failCount += TestMulsCase<uint8_t>(
      "Test59_Muls_UINT8",
      {10, 20, 30, 40}, {2, 2},
      3,  // scalar
      aclDataType::ACL_UINT8, 0, 0, stream
  );
  
  // ========== 测试60: aclnnInplaceMul UINT8 ==========
  totalTests++;
  failCount += TestInplaceMulCase<uint8_t>(
      "Test60_InplaceMul_UINT8",
      {10, 20, 30, 40}, {2, 2},
      {2, 3, 4, 5}, {2, 2},
      aclDataType::ACL_UINT8, 0, 0, stream
  );
  
  // ========== 测试61: aclnnInplaceMuls UINT8 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<uint8_t>(
      "Test61_InplaceMuls_UINT8",
      {10, 20, 30, 40}, {2, 2},
      5,  // scalar
      aclDataType::ACL_UINT8, 0, 0, stream
  );
  
  // ============================================================================
  // 【新增】INT16测试 (补充整数类型)
  // ============================================================================
  
  // ========== 测试62: INT16同shape ==========
  totalTests++;
  failCount += TestMulCase<int16_t>(
      "Test62_INT16_SameShape",
      {100, 200, 300, 400}, {2, 2},
      {2, 3, 4, 5}, {2, 2},
      aclDataType::ACL_INT16, 0, 0, stream
  );
  
  // ========== 测试63: INT16广播 ==========
  totalTests++;
  failCount += TestMulCase<int16_t>(
      "Test63_INT16_Broadcast",
      {10, 20, 30, 40, 50, 60}, {2, 3},
      {2, 3, 4}, {3},
      aclDataType::ACL_INT16, 0, 0, stream
  );
  
  // ========== 测试64: aclnnMuls INT16 ==========
  totalTests++;
  failCount += TestMulsCase<int16_t>(
      "Test64_Muls_INT16",
      {100, 200, 300, 400}, {2, 2},
      5,  // scalar
      aclDataType::ACL_INT16, 0, 0, stream
  );
  
  // ========== 测试65: aclnnInplaceMuls INT16 ==========
  totalTests++;
  failCount += TestInplaceMulsCase<int16_t>(
      "Test65_InplaceMuls_INT16",
      {100, 200, 300, 400}, {2, 2},
      3,  // scalar
      aclDataType::ACL_INT16, 0, 0, stream
  );
  
  // ============================================================================
  // 【新增】边界值测试
  // ============================================================================
  
  // ========== 测试66: FLOAT极大值测试 ==========
  // 【测试目的】验证大数乘法的数值稳定性
  totalTests++;
  failCount += TestMulCase<float>(
      "Test66_FLOAT_MaxValue",
      {1e10f, 2e10f, 3e10f, 4e10f}, {2, 2},
      {1e10f, 1e10f, 1e10f, 1e10f}, {2, 2},
      aclDataType::ACL_FLOAT, 1e5, 1e-5, stream
  );
  
  // ========== 测试67: FLOAT极小值测试 ==========
  // 【测试目的】验证小数乘法的精度
  totalTests++;
  failCount += TestMulCase<float>(
      "Test67_FLOAT_MinValue",
      {1e-10f, 2e-10f, 3e-10f, 4e-10f}, {2, 2},
      {1e-10f, 1e-10f, 1e-10f, 1e-10f}, {2, 2},
      aclDataType::ACL_FLOAT, 1e-20, 1e-5, stream
  );
  
  // ========== 测试68: INT32极大值测试 ==========
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test68_INT32_MaxValue",
      {100000, 200000, 300000, 400000}, {2, 2},
      {100, 100, 100, 100}, {2, 2},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试69: FLOAT全1测试 ==========
  // 【测试目的】验证单位元性质: 1 × x = x
  totalTests++;
  failCount += TestMulCase<float>(
      "Test69_FLOAT_AllOnes",
      {1, 1, 1, 1, 1, 1}, {2, 3},
      {1, 1, 1, 1, 1, 1}, {2, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试70: FLOAT全0测试 ==========
  // 【测试目的】验证零元性质: 0 × x = 0
  totalTests++;
  failCount += TestMulCase<float>(
      "Test70_FLOAT_AllZeros",
      {0, 0, 0, 0, 0, 0}, {2, 3},
      {1, 2, 3, 4, 5, 6}, {2, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试71: INT32全1测试 ==========
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test71_INT32_AllOnes",
      {1, 1, 1, 1}, {2, 2},
      {1, 1, 1, 1}, {2, 2},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试72: INT32全0测试 ==========
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test72_INT32_AllZeros",
      {0, 0, 0, 0}, {2, 2},
      {1, 2, 3, 4}, {2, 2},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试73: FLOAT正负交替测试 ==========
  totalTests++;
  failCount += TestMulCase<float>(
      "Test73_FLOAT_AlternatingSign",
      {1, -2, 3, -4, 5, -6}, {2, 3},
      {-1, 2, -3, 4, -5, 6}, {2, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试74: INT32正负交替测试 ==========
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test74_INT32_AlternatingSign",
      {10, -20, 30, -40}, {2, 2},
      {-1, 2, -3, 4}, {2, 2},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ============================================================================
  // 【新增】更多广播场景测试
  // ============================================================================
  
  // ========== 测试75: FLOAT三维广播 [4,1,3] × [1,3] ==========
  totalTests++;
  failCount += TestMulCase<float>(
      "Test75_FLOAT_3D_Broadcast2",
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 1, 3},
      {2, 3, 4}, {1, 3},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试76: FLOAT三维广播 [1,4,3] × [4,1] ==========
  totalTests++;
  failCount += TestMulCase<float>(
      "Test76_FLOAT_3D_Broadcast3",
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 4, 3},
      {2, 3, 4, 5}, {4, 1},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试77: INT32三维广播 [2,1,2] × [1,3,1] ==========
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test77_INT32_3D_Broadcast",
      {1, 2, 3, 4}, {2, 1, 2},
      {10, 20, 30}, {1, 3, 1},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试78: FLOAT四维广播 [2,1,2,3] × [1,2,1] ==========
  totalTests++;
  failCount += TestMulCase<float>(
      "Test78_FLOAT_4D_Broadcast",
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 1, 2, 3},
      {2, 3, 4}, {1, 2, 1},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ============================================================================
  // 【新增】单元素tensor测试
  // ============================================================================
  
  // ========== 测试79: FLOAT单元素tensor ==========
  totalTests++;
  failCount += TestMulCase<float>(
      "Test79_FLOAT_SingleElement",
      {5.0f}, {1},
      {3.0f}, {1},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ========== 测试80: INT32单元素tensor ==========
  totalTests++;
  failCount += TestMulCase<int32_t>(
      "Test80_INT32_SingleElement",
      {100}, {1},
      {50}, {1},
      aclDataType::ACL_INT32, 0, 0, stream
  );
  
  // ========== 测试81: FLOAT单元素广播 ==========
  totalTests++;
  failCount += TestMulCase<float>(
      "Test81_FLOAT_SingleElement_Broadcast",
      {1, 2, 3, 4, 5, 6}, {2, 3},
      {10.0f}, {1},
      aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
  );
  
  // ============================================================================
  // 【新增】大尺寸tensor测试
  // ============================================================================
  
  // ========== 测试82: FLOAT大尺寸tensor (1000元素) ==========
  totalTests++;
  {
    std::vector<float> largeData1(1000, 2.0f);
    std::vector<float> largeData2(1000, 3.0f);
    failCount += TestMulCase<float>(
        "Test82_FLOAT_LargeTensor_1000",
        largeData1, {1000},
        largeData2, {1000},
        aclDataType::ACL_FLOAT, 1e-5, 1e-5, stream
    );
  }
  
  // ========== 测试83: INT32大尺寸tensor (500元素) ==========
  totalTests++;
  {
    std::vector<int32_t> largeData1(500, 10);
    std::vector<int32_t> largeData2(500, 20);
    failCount += TestMulCase<int32_t>(
        "Test83_INT32_LargeTensor_500",
        largeData1, {500},
        largeData2, {500},
        aclDataType::ACL_INT32, 0, 0, stream
    );
  }
  
  // ============================================================================
  // 输出测试汇总
  // ============================================================================
  LOG_PRINT("\n========== 测试汇总 ==========\n");
  LOG_PRINT("总测试数: %d\n", totalTests);
  LOG_PRINT("通过: %d\n", totalTests - failCount);
  LOG_PRINT("失败: %d\n", failCount);
  LOG_PRINT("通过率: %.2f%%\n", (totalTests - failCount) * 100.0 / totalTests);
  
  // 清理资源
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  // 返回失败数量(题目要求第55行: "有失败用例返回非0值")
  return failCount;
}
