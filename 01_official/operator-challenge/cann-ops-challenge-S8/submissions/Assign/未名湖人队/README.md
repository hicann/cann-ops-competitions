# Assign 算子提交说明

## 团队信息

- 团队名称：未名湖人队
- GitCode：<https://gitcode.com/yangtianjian798>

## 作品信息

- 赛题算子：Assign
- 目标芯片：Ascend 910B
- CANN 版本：社区版 8.5.0

## 编译方式

在已配置 CANN 环境的 Linux 系统中执行：

```bash
bash build.sh
```

本次提交使用官方 PR CI 完成最终编译检查。

## 实现与优化策略

Host 侧按数据类型和总字节数选择 8 核或 32 核调度，并将每核数据量对齐到 32 个元素。Kernel 侧把输入 `value` 按核分片后拷贝到 `ref`，使用 192 KiB UB 和三级队列流水并行搬运；对不足 32 个元素的尾块使用 `DataCopyPad` 处理。

不同数据类型采用经过赛题测试的多核切换阈值：FP32/INT32、FP16/BF16/INT16 为 16 KiB，UINT8/INT8/BOOL 为 256 KiB。

## 性能结果

官方赛题基准为 1672 us。本版本首次官方测评 `prof_sum` 为 1668.4644 us；14 次重复测评中的最佳结果为 1665.980398 us，中位数为 1675.4631975 us。
