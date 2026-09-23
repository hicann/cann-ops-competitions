# 哈尔滨工业大学 + SuperScale：东北赛区初赛代码归档

- 赛事：2026 HCCL 通信库创新大赛（东北赛区），初赛。
- 学校：哈尔滨工业大学；队伍：SuperScale；参赛账号：[qq_36999329](https://gitcode.com/qq_36999329)。
- 题目：HcclScatter，AICPU + TS 编排。
- 归档版本：v14，队伍于 2026-09-11 15:44:53 手动提交的最终初赛版本。

## 目录与源码一致性

`code/` 保存最终冻结工程的全部 20 个文件，包括构建脚本、头文件、Host、AICPU 源码和原工程说明；没有构建产物。`SHA256SUMS` 记录 PR 中这些文件的 SHA-256。归档时仅将 `custom.h` 和 `exec_op.cc` 的 CRLF 行尾统一为原工程 `.gitattributes` 要求的 LF；与冻结 v14 比对，除这两处行尾转换外，全部文件逐字一致，算法内容未修改。

本目录按赛区 submission.md 的“学校+队伍名”要求命名。与最初比赛工程相比，仅 `include/custom.h` 和 `op_kernel_aicpu/exec_op.cc` 不同，其余 18 个文件保持一致；Host 文件 `op_host/scatter.cc` 未改动。

```bash
sha256sum -c SHA256SUMS
```

## 算法说明

针对初赛的 2 个 Server、每个 Server 8 个 NPU 拓扑：

- 小数据使用四组通知树：root 一次暂存，组长转发 ready 并汇总 consumed，各 rank 直接读取 root 对应切片到输出。
- 大数据结合 root 直达与配对中继，将约 4/11 前缀经中继转发；按角色打包双缓冲，块长上限 32 MiB，并根据可用 CCL 容量缩小。
- 原末块至少 4 MiB 时，按 FP32 对齐拆为约 3/4 和 1/4 两段，最多增加一轮。
- 保留显式本地线程启动、同一 root 通道由主线程等待、槽 credit 和末尾完成确认；非 16 rank 及容量不足场景使用回退路径。

接口为 `HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm, stream)`，其中 `recvCount` 是每个 rank 的接收元素数。实现细节见 `code/op_kernel_aicpu/exec_op.cc`。

## 构建

按比赛原工程要求使用 Linux 和 CANN Toolkit 9.1.0，安装来源及配置详见 [原工程 README](code/README.md)。在已安装匹配 SDK 的环境中：

```bash
source /usr/local/Ascend/cann/set_env.sh
cd code
bash build.sh
# 调试构建：bash build.sh --debug
```

若 CANN 安装在其他目录，请先 source 对应的 `set_env.sh`，确保 `ASCEND_HOME_PATH` 指向安装目录。

## 验证与比赛结果

队伍提供的最终判题结果：Pass，8 个功能用例全部通过，3 个性能用例均有 Pass 计时，榜单总分 143.89，显示第 4 名。功能分为 120，性能分 23.89 由总分减功能分得到。

| 性能用例 | v14 耗时 |
| --- | ---: |
| 9 | 37 μs |
| 10 | 1.05 ms |
| 11 | 844 μs |

以上来自队伍提供的 2026-09-11 15:44:53 榜单记录；本次归档未独立核实该次提交 ID，未将提交入口当作提交详情链接。带宽未提供。

开发期间，此源码通过 16,974 次本地异步功能模型调用（含专项边界、连续排队和两种调度，ASan/UBSan），另通过 200,000 组整数布局检查。这些是 CPU 模型验证，不等于官方 Checker、完整 SDK 构建或 NPU 性能测试。本次归档未重新运行官方环境；本机缺少匹配 CANN SDK，未声称本地完整工程构建通过。

v14 的大数据耗时相比 v12 的 1.04 ms / 825 μs 略高，本归档保留实际最终提交源码，不把本轮参数改动表述为已获性能提升。

## 许可

保留原工程所有版权声明及 [CANN Open Software License Agreement Version 2.0](code/LICENSE)。
