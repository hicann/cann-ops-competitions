# aclnnMaxPool2dWithMaskBackward

## 接口定义

参考昇腾官方标准两阶段（Two-stage）ACLNN 调用范式，提供算子工作区大小获取接口和算子下发执行接口：

```cpp
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

// 第一阶段：获取算子执行所需的 Workspace 空间大小并构建执行器
aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* self,
    const aclTensor* indices,
    const aclIntArray* kernelSize,
    const aclIntArray* stride,
    const aclIntArray* padding,
    const aclIntArray* dilation,
    bool ceilMode,
    aclTensor* gradInput,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

// 第二阶段：下发算子任务到指定的计算流执行计算
aclnnStatus aclnnMaxPool2dWithMaskBackward(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## 参数说明

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor | 描述 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `gradOutput` | 输入 | FLOAT32, FLOAT16, BFLOAT16 | NCHW | 4维 [N, C, Ho, Wo] | √ | 反向传播上一步输出的梯度，与正向输出 shape 一致 |
| `self` | 输入 | FLOAT32, FLOAT16, BFLOAT16 | NCHW | 4维 [N, C, Hi, Wi] | √ | 正向输入 Tensor，提供输入 shape 与数据类型 |
| `indices` | 输入 | INT8 | NCHW | 4维 [N, C, kh*kw, maskW] | √ | 正向最大值索引位置组成的 Tensor，有效前缀为小端 INT32 argmax |
| `kernelSize` | 属性 | INT64 数组 | — | 长度 1 或 2 | — | 池化操作中使用的滑动窗口大小 |
| `stride` | 属性 | INT64 数组 | — | 长度 0, 1 或 2 | — | 窗口移动的步长，长度为 0 时等于 kernelSize |
| `padding` | 属性 | INT64 数组 | — | 长度 1 或 2 | — | 输入数据的边缘填充量 |
| `dilation` | 属性 | INT64 数组 | — | 长度 1 或 2 | — | 控制窗口中元素的步幅，当前仅支持值为 1 |
| `ceilMode` | 属性 | BOOL | — | 标量 | — | 计算输出形状时向上取整标志（True / False） |
| `gradInput` | 输出 | FLOAT32, FLOAT16, BFLOAT16 | NCHW | 4维 [N, C, Hi, Wi] | √ | 反向传播输出梯度，与 `self` 的 shape 和数据格式一致 |
| `workspaceSize` | 输出 | uint64_t* | — | 标量指针 | — | 返回算子在设备侧执行所需的 Workspace 大小（字节数） |
| `executor` | 输出 | aclOpExecutor** | — | 指针的指针 | — | 返回算子执行器指针，供第二阶段调用 |
| `workspace` | 输入 | void* | — | 设备内存指针 | — | 设备侧 Workspace 内存地址，由调用者依据第一阶段返回的大小分配 |
| `stream` | 输入 | aclrtStream | — | 昇腾计算流 | — | 指定算子任务执行的昇腾计算流 |

> **注**：
> 1. `BFLOAT16` 仅在 Atlas 800T A2 (Ascend 910B) 上生效。
> 2. `gradOutput`、`self`、`indices`、`gradInput` 均支持非连续 Tensor，由框架底层或 Host 适配层自动完成连续化处理。

## 约束限制

1. **确定性计算**：本算子采用属主独占与固定 Tile 空间切分，天然支持确定性计算（Deterministic Execution），多核执行结果 100% 逐位复现。
2. **NaN/-Inf 限制**：输入数据暂不支持 NaN、-Inf。
3. **FLOAT 类型精度**：在 Atlas 800T A2 上，单精度浮点（FLOAT32）采用片上单精度原生累加，与参考结果达到 0 误差完全对齐。
4. **dilation 限制**：dilation 数组长度支持 1 或 2，数值仅支持为 1。
5. **padding 约束**：padding 值大于等于 0 且小于等于 kernelSize。

## 调用示例

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool2d_with_mask_backward.h"

int main() {
    // 1. 初始化 ACL 环境
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 2. 构造输入输出 Tensor 与属性（示例伪代码）
    // ... aclCreateTensor, aclCreateIntArray ...
    aclTensor* gradOutput = nullptr;
    aclTensor* self = nullptr;
    aclTensor* indices = nullptr;
    aclIntArray* kernelSize = nullptr;
    aclIntArray* stride = nullptr;
    aclIntArray* padding = nullptr;
    aclIntArray* dilation = nullptr;
    bool ceilMode = false;
    aclTensor* gradInput = nullptr;

    // 3. 第一阶段：获取 Workspace 大小与构建执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus status = aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
        gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode,
        gradInput, &workspaceSize, &executor);
    if (status != ACL_SUCCESS) {
        std::cerr << "GetWorkspaceSize failed with code: " << status << std::endl;
        return -1;
    }

    // 4. 分配 Workspace 内存
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 5. 第二阶段：下发算子执行
    status = aclnnMaxPool2dWithMaskBackward(workspace, workspaceSize, executor, stream);
    if (status != ACL_SUCCESS) {
        std::cerr << "Execution failed with code: " << status << std::endl;
        return -1;
    }

    // 6. 同步等待流完成
    aclrtSynchronizeStream(stream);

    // 7. 释放资源
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "aclnnMaxPool2dWithMaskBackward executed successfully!" << std::endl;
    return 0;
}
```
