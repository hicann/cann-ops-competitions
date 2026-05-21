# aclnnApplyAdagradD


## 产品支持情况
|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |


## 功能说明

- **算子功能：** 实现adagradD优化器功能。

- **计算公式：**
$$
g_t = grad
$$

$$
accum_{t}=\begin{cases}
accum_{t-1}+g_{t}^{2}
& \text{ if } update\_slots = true\\
accum_{t-1}
& \text{ if } update\_slots = false
\end{cases}
$$

$$
\theta_{t}=\theta_{t-1}-\frac{\eta \cdot g_t}{\sqrt{accum_t}}
$$



## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“`aclnnApplyAdagradDGetWorkspaceSize`”接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用“`aclnnApplyAdagradD`”接口执行计算。`aclnnApplyAdagradD`：需预先创建输出张量 `var_out` 和 `accum_out`，分别用于存储更新后的参数和历史梯度平方累积量。

* `aclnnStatus aclnnApplyAdagradDGetWorkspaceSize(const aclTensor* var, const aclTensor* accum, const aclTensor* lr, const aclTensor* grad, bool update_slots, aclTensor *var_out, aclTensor *accum_out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnApplyAdagradD(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`


## aclnnApplyAdagradDGetWorkspaceSize

- **参数说明：**

  * `var`（aclTensor\*，计算输入）：待更新参数，公式中的 $\theta_{t-1}$，Device 侧的 aclTensor，shape 支持 1-8 维，数据类型支持 FLOAT16、BFLOAT16、FLOAT32。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。
  * `accum`（aclTensor\*，计算输入）：历史梯度平方累积量，公式中的 $accum_{t-1}$，Device 侧的 aclTensor，数据类型支持 FLOAT16、BFLOAT16、FLOAT32，shape、dtype 要求与 `var` 一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。
  * `lr`（aclTensor\*，计算输入）：学习率，公式中的 $\eta$，Device 侧的 aclTensor，数据类型支持 FLOAT16、BFLOAT16、FLOAT32，shape 要求为 `[1]`，dtype 要求与 `var` 一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。
  * `grad`（aclTensor\*，计算输入）：当前梯度，公式中的 $g_t$，Device 侧的 aclTensor，数据类型支持 FLOAT16、BFLOAT16、FLOAT32，shape、dtype 要求与 `var` 一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。
  * `update_slots`（bool，属性）：是否更新历史梯度平方累积量，数据类型为 BOOL。可选值为 `true` 或 `false`，默认值为 `true`。
  * `var_out`（aclTensor\*，计算输出）：更新后参数，公式中的 $\theta_t$，Device 侧的 aclTensor，数据类型支持 FLOAT16、BFLOAT16、FLOAT32，shape、dtype 要求与 `var` 一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。
  * `accum_out`（aclTensor\*，计算输出）：更新后历史梯度平方累积量，公式中的 $accum_t$，Device 侧的 aclTensor，数据类型支持 FLOAT16、BFLOAT16、FLOAT32，shape、dtype 要求与 `accum` 一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。
  * `workspaceSize`（uint64_t\*，出参）：返回需要在 Device 侧申请的 workspace 大小。
  * `executor`（aclOpExecutor\*\*，出参）：返回执行器地址。

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```text
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：传入的计算输入或输出参数是空指针时。
  161002 (ACLNN_ERR_PARAM_INVALID)：1. 传入的计算输入或计算输出的数据类型不在支持的范围内时。
                                  2. `var`、`accum`、`grad`、`var_out`、`accum_out` 的数据类型不一致时。
                                  3. `var`、`accum`、`grad`、`var_out`、`accum_out` 的 shape 不匹配时。
                                  4. `lr` 的 shape 大小不为 1 时。
  ```


## aclnnApplyAdagradD

- **参数说明：**

  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnApplyAdagradDGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 输入输出张量的数据类型应保持一致，数据类型支持FLOAT16、BFLOAT16、FLOAT32。
- `var`、`accum`、`grad`、`var_out`、`accum_out` 的 shape 应保持一致。
- 输入张量 `lr` 的 shape 大小应为 1，且数据类型应与 `var` 一致。

- 确定性计算： 
  aclnnApplyAdagradD默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include <cstdio>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adagrad_d.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d", ret); return ret);

  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d", ret); return ret);

  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d", ret); return ret);

  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);

  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);

  return 0;
}

int main() {
  // 1. 初始化 device / stream
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d", ret); return ret);

  // 2. 构造输入与输出
  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> accumShape = {2, 2};
  std::vector<int64_t> lrShape = {1};
  std::vector<int64_t> gradShape = {2, 2};
  std::vector<int64_t> varOutShape = {2, 2};
  std::vector<int64_t> accumOutShape = {2, 2};

  void* varDeviceAddr = nullptr;
  void* accumDeviceAddr = nullptr;
  void* lrDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  void* varOutDeviceAddr = nullptr;
  void* accumOutDeviceAddr = nullptr;

  aclTensor* var = nullptr;
  aclTensor* accum = nullptr;
  aclTensor* lr = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* var_out = nullptr;
  aclTensor* accum_out = nullptr;

  std::vector<float> varHostData = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> accumHostData = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> lrHostData = {0.001f};
  std::vector<float> gradHostData = {0.1f, 0.2f, 0.3f, 0.4f};

  // 输出 tensor 对应 device 内存也需要先分配
  std::vector<float> varOutHostInitData = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> accumOutHostInitData = {0.0f, 0.0f, 0.0f, 0.0f};

  bool update_slots = true;

  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(accumHostData, accumShape, &accumDeviceAddr, aclDataType::ACL_FLOAT, &accum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(varOutHostInitData, varOutShape, &varOutDeviceAddr, aclDataType::ACL_FLOAT, &var_out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(accumOutHostInitData, accumOutShape, &accumOutDeviceAddr, aclDataType::ACL_FLOAT, &accum_out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用 ApplyAdagradD 两段式接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnApplyAdagradDGetWorkspaceSize(
      var, accum, lr, grad, update_slots, var_out, accum_out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnApplyAdagradDGetWorkspaceSize failed. ERROR: %d", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d", ret); return ret);
  }

  ret = aclnnApplyAdagradD(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnApplyAdagradD failed. ERROR: %d", ret); return ret);

  // 4. 同步
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d", ret); return ret);

  // 5. 拷回输出结果
  auto varSize = GetShapeSize(varOutShape);
  auto accumSize = GetShapeSize(accumOutShape);

  std::vector<float> varOutResult(varSize, 0.0f);
  std::vector<float> accumOutResult(accumSize, 0.0f);

  ret = aclrtMemcpy(varOutResult.data(), varOutResult.size() * sizeof(float),
                    varOutDeviceAddr, varSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy var_out from device to host failed. ERROR: %d", ret); return ret);

  ret = aclrtMemcpy(accumOutResult.data(), accumOutResult.size() * sizeof(float),
                    accumOutDeviceAddr, accumSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy accum_out from device to host failed. ERROR: %d", ret); return ret);

  LOG_PRINT("var_out:\n");
  for (int64_t i = 0; i < varSize; ++i) {
    LOG_PRINT("var_out[%ld] = %f", i, varOutResult[i]);
  }

  LOG_PRINT("accum_out:\n");
  for (int64_t i = 0; i < accumSize; ++i) {
    LOG_PRINT("accum_out[%ld] = %f", i, accumOutResult[i]);
  }

  // 6. 释放 aclTensor
  aclDestroyTensor(var);
  aclDestroyTensor(accum);
  aclDestroyTensor(lr);
  aclDestroyTensor(grad);
  aclDestroyTensor(var_out);
  aclDestroyTensor(accum_out);

  // 7. 释放 device 资源
  aclrtFree(varDeviceAddr);
  aclrtFree(accumDeviceAddr);
  aclrtFree(lrDeviceAddr);
  aclrtFree(gradDeviceAddr);
  aclrtFree(varOutDeviceAddr);
  aclrtFree(accumOutDeviceAddr);

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
