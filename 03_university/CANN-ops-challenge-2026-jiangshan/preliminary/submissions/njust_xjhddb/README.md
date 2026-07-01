## 团队信息

- 团队名称：[xjhddb]
- 所属单位：[南京理工大学]
- 团队成员：
  - [季然]，[算子代码]
  - [王睿涵]，[算子代码]
- 联系人：[季然]
- 联系邮箱：[ranji@njust.edu.cn]

## 实现思路

host 侧在 `op_host/erf.cpp` 中完成算子描述和 tiling：

1. 读取输入张量的数据类型、元素总数和平台 UB 大小。
2. 根据输入长度动态选择 `blockDim` 和 `tileLength`。
3. 将 `length`、`blockLength`、`tileLength` 写入 `ErfTilingData`。
4. kernel 侧根据 tiling 信息确定每个 AICore 负责的数据区间。

kernel 侧在 `op_kernel/erf.cpp` 中完成实际计算：

1. 每个 AICore 根据 block 索引计算全局内存偏移。
2. 将输入从 GM 分片搬运到 UB。
3. 在 UB 中调用 `FastErf` 进行多项式近似计算。
4. 将结果从 UB 写回 GM。

对于尾块或非对齐长度，代码使用 `DataCopyPad` 处理，保证任意长度输入都能正常搬运和计算。

## 精度策略

默认实现使用 `FastErf` 多项式近似计算 Erf，并在计算结束后将结果限制在 `[-1, 1]` 区间，符合 Erf 函数值域。

如果需要切换为 AscendC 内置实现，可在编译时定义 `ERF_USE_ASCENDC_BUILTIN`，此时代码会调用：

```cpp
AscendC::Erf(y_local, x_local, calcCount);
```

## 性能优化策略

- **多核并行**：根据输入规模设置 `blockDim`，大规模输入由多个 AICore 并行处理。
- **分片计算**：通过 `tileLength` 控制单次进入 UB 的元素数量，降低 UB 占用压力。
- **双缓冲队列**：使用 `BUFFER_NUM = 2`，在处理当前 tile 时预取下一片数据。
- **向量化计算**：`FastErf` 使用 AscendC 向量算子完成乘法、加法、截断等操作。
- **尾块处理**：对非对齐数据使用 pad 搬运，减少边界分支和越界风险。