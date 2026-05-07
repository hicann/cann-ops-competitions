# Add 算子测试说明

## 1. 项目概况

本次提交围绕 `ops-math` 仓库中的 Add 算子示例程序进行扩展，形成一份单文件测试用例，并保留当前 `build/` 编译产物目录。

测试文件位置为：

- `math/add/examples/test_aclnn_add.cpp`

## 2. 测试程序说明

测试程序在官方示例基础上补充了多组端到端测试，主要包括：

- ACL 运行环境初始化与释放
- Tensor / Scalar 构造与释放
- 结果回读与逐元素校验
- 正常路径、异常路径和边界输入测试

测试覆盖的接口包括：

- `aclnnAdd`
- `aclnnAdds`
- `aclnnInplaceAdd`
- `aclnnInplaceAdds`
- `aclnnAddV3`
- `aclnnInplaceAddV3`

## 3. 覆盖率统计范围

当前 `build/` 目录中与 Add 直接相关的覆盖率统计对象包括：

- `math/add/op_api/aclnn_add.cpp`
- `math/add/op_api/aclnn_add_v3.cpp`
- `math/add/op_api/add.cpp`
- `math/add/op_graph/add_graph_infer.cpp`
- `math/add/op_host/add_infershape.cpp`
- `math/add/op_host/arch35/add_tiling_arch35.cpp`
- `math/add/op_host/add_def.cpp`

## 4. 当前覆盖情况

基于当前提交包内 `build/` 目录中的 `gcda` 文件，使用 `/usr/bin/gcov-12 -b` 统计到的行覆盖结果如下：

- `math/add/op_api/add.cpp`：`96.36%`
- `math/add/op_api/aclnn_add.cpp`：`80.14%`
- `math/add/op_api/aclnn_add_v3.cpp`：`77.63%`
- `math/add/op_host/arch35/add_tiling_arch35.cpp`：`100.00%`
- `math/add/op_host/add_def.cpp`：`100.00%`
- `math/add/op_host/add_infershape.cpp`：`0.00%`
- `math/add/op_graph/add_graph_infer.cpp`：`0.00%`

## 5. 已覆盖内容

当前测试已覆盖的主要内容包括：

- Add 基础逐元素加法
- 广播场景
- `alpha == 1` 与 `alpha != 1`
- 张量与标量组合
- 原地加法路径
- `AddV3` 与 `InplaceAddV3`
- 多种数据类型组合
- 混合类型提升路径
- `Bool` 特殊类型处理
- 空 tensor 处理
- 参数非法场景

## 6. 未覆盖内容及原因

当前未完全覆盖的部分主要集中在以下几类：

- `op_graph` 相关文件未覆盖
  原因：本次执行链路使用的是 eager 示例，未走 graph 示例流程，因此 `add_graph_infer.cpp` 未实际执行。

- `op_host/add_infershape.cpp` 未覆盖
  原因：当前测试主要通过 eager / op_api 路径驱动算子执行，没有单独触发该 host infer shape 逻辑的执行路径。

- `aclnn_add.cpp` 与 `aclnn_add_v3.cpp` 中仍有部分分支未覆盖
  原因：剩余分支中包含平台架构相关判断、特定 promote 分支、部分错误分支以及在当前模拟器环境下不稳定或不可达的路径。

- `add.cpp` 仍有极少量未覆盖分支
  原因：剩余分支主要与底层架构分支和特定 kernel 选择条件有关，在当前执行环境下无法稳定触发全部分支。

## 7. 场景设计

测试场景主要分为以下几类：

- 基础计算场景
- 广播与格式场景
- 数据类型场景
- 参数检查场景

具体覆盖了：

- 相同 shape 的逐元素加法
- 带 `alpha` 的加法
- 标量与张量组合
- 原地加法
- ND / NCHW 格式
- `FLOAT`、`FLOAT16`、`BF16`、`INT8`、`UINT8`、`INT32`、`INT64`、`BOOL`、`COMPLEX64`、`DOUBLE`
- 空指针、非法 shape、非法 rank、非法广播、不支持 dtype、非法 `alpha`

## 8. 结果校验方式

测试程序在执行完成后会把输出数据回读到主机侧，并进行逐元素校验。

校验方式包括：

- 实数类型按容差比较
- 复数类型分别比较实部与虚部
- 布尔类型按逻辑值比较

## 9. 提交内容

提交包目录结构如下：

```text
云上码术/
├── test_aclnn_add.cpp
├── build/
└── Add.md
```

测试文件保持为单文件形式，没有依赖额外自定义头文件或其他私有源码，便于在相同版本仓库中直接替换使用。
