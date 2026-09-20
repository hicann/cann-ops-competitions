# aclnnMaxPool2dWithMask 接口合同核对

本文用于09-46前向任务设计评审，不创建新的公开接口。基准为[官方API说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMaxPool2dWithMask.md)及[ops-nn d9ba860头文件](https://gitcode.com/cann/ops-nn/blob/d9ba8606dd4e9ebaf02b14b7fbf78906d439a6b8/pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool2d_with_indices.h)。存在差异的indices/310P/1×1以设计文档所列待确认项为准。

## 函数原型

```cpp
aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode,
    aclTensor* out, aclTensor* indices,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnMaxPool2dWithMask(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

安装头文件为`aclnnop/aclnn_max_pool2d_with_indices.h`。不手写另一个同名ABI，不以内部kernel直调替代正式入口。

## 参数

| 参数 | 含义与约束 |
| --- | --- |
| self | 设备tensor，3/4维，CHW/NCHW逻辑布局，ND/NCHW格式；支持非连续；A2 FP16/FP32/BF16 |
| kernelSize | INT64数组，长度1/2，值正数 |
| stride | INT64数组，长度0/1/2；空时等于kernelSize，其他正数 |
| padding | INT64数组，长度1/2；0<=p<=k/2 |
| dilation | INT64数组，长度1/2；值均为1 |
| ceilMode | 是否使用ceil输出尺寸及最后窗口修正 |
| out | 设备tensor，与self相同dtype，shape按公式推导；支持非连续 |
| indices | INT8设备容器，shape见设计文档；附件golden为连续int32小端前缀+零尾部，当前310P/1×1语义冲突待确认 |
| workspaceSize | 第一段返回所需设备workspace字节数，第二段传同一值 |
| executor | 第一段返回本次执行计划，交给对应第二段；不假定可跨调用复用 |
| workspace | 非零workspaceSize时申请对应设备内存，零时可传nullptr |
| stream | 第二段使用调用者指定stream，不在算子内加入Host同步 |

## 返回值与约束

`ACLNN_SUCCESS`表示该段调用成功，设备异步错误还需在测试的stream同步阶段检查。现有参数检查使用`ACLNN_ERR_PARAM_NULLPTR=161001`及`ACLNN_ERR_PARAM_INVALID=161002`等状态；具体失败点按正式SDK实测，不把本地C++异常当作API错误码。

N=0为空batch，C/H/W不可为0。input不支持NaN/-Inf，dilation仅1；310P的dtype、布局和ceil限制单独核验。out/indices形状错误、空指针、错误属性、非连续storageOffset、workspace/stream行为均列入后续公开入口测试。

## 调用顺序示例

以下省略tensor/数组构造，完整用法以[官方示例](https://gitcode.com/cann/ops-nn/blob/d9ba8606dd4e9ebaf02b14b7fbf78906d439a6b8/pooling/max_pool3d_with_argmax_v2/examples/arch22/test_aclnn_max_pool2d_with_mask.cpp)为准：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus status = aclnnMaxPool2dWithMaskGetWorkspaceSize(
    self, kernelSize, stride, padding, dilation, ceilMode,
    out, indices, &workspaceSize, &executor);
// 必须先检查status；非零时停止，不能继续执行。
void* workspace = nullptr;
if (status == ACLNN_SUCCESS && workspaceSize > 0) {
    // 必须检查分配返回码；失败时停止。
    aclError allocationStatus = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (allocationStatus != ACL_SUCCESS) {
        // 进入调用者的错误处理和资源释放路径。
        return;
    }
}
if (status == ACLNN_SUCCESS) {
    status = aclnnMaxPool2dWithMask(workspace, workspaceSize, executor, stream);
}
// 调用者在需要结果时同步；同步完成后再释放workspace/tensor/输入数组。
```

该段是调用顺序说明，不是已编译的完整程序。实际测试应使用完整的错误检查和资源释放驱动，并记录动态库身份及实际kernel派发。

AI辅助：OpenAI Codex（GPT-6）。
