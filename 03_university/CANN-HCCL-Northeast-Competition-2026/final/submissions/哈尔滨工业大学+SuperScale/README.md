# 哈尔滨工业大学 + SuperScale：东北赛区决赛代码归档

- 赛事：2026 HCCL 通信库创新大赛（东北赛区），决赛。
- 学校：哈尔滨工业大学；队伍：SuperScale。
- 提交账号：[qq_36999329](https://gitcode.com/qq_36999329)；联系邮箱：1092897051@qq.com。
- 赛题：486，Scatter 集合通信，CCU 执行。
- 版本：队伍确认的最终版本 v18；比赛提交时间：2026-09-18 21:59:52。

## 源码与目录

`code/` 包含 v18 冻结源码包的完整 19 个文件，逐字节一致，未修改算法、行尾、构建脚本或许可文件。`SHA256SUMS` 记录这 19 个文件的 SHA-256；没有包含构建产物、访问令牌或本地开发资料。目录命名遵循本赛区 `final/submissions/submission.md` 的“学校+队伍名”要求。

原始源码包 SHA-256：`d665fdadeac69c657bd91ea7643146b4adf12b849122f337758e984ec107417b`。

```bash
sha256sum -c SHA256SUMS
```

## 算法说明

实现 `HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm, stream)`，`recvCount` 为每个 rank 接收的元素数。使用实际拓扑 API 获取成员、链路及本端 IO die，支持任意合法 root 和逻辑 rank 映射；同一 peer 复用 channel，不同本端 IO die 分别编排 CCU kernel。

- 小数据由接收者读取 root 输入切片，按调用代际区分元数据槽和通知位。
- 大数据结合直接读取、拓扑感知中继与分块传输，根据实际成员、数据尺寸和可用 CCL 容量选路。
- 面向决赛的 2×8、4×1、8+4、4×3 四种拓扑；源文件保留比赛最终提交行为。

## 构建与运行

使用比赛原工程要求的 Linux、CANN Toolkit 9.1.0 及匹配的 CCU/HCCL SDK。具体安装来源见 [原工程说明](code/README.md)。

```bash
source /usr/local/Ascend/cann/set_env.sh
cd code
bash build.sh
```

产物供比赛配套 Runner/Checker 或相应通信运行环境加载；本目录不是独立的命令行测试程序。可用 `bash build.sh --debug` 构建 Debug 版本。

## 成绩与验证范围

队伍记录的最终比赛结果为 **Pass / 381.93，18 个功能点全部通过**。四组性能耗时如下（单位 μs，每组依次为题面的 512KB、512MB、400MB+4B）：

| 拓扑 | 小数据 | 大数据 | 非整齐大数据 |
| --- | ---: | ---: | ---: |
| 2×8 | 10 | 942 | 739 |
| 4×1 | 9 | 2060 | 1610 |
| 8+4 | 10 | 923 | 723 |
| 4×3 | 11 | 1680 | 1310 |

以上是队伍提供的比赛记录，本次归档没有重新访问判题平台或复测性能。v18 的历史本地仿真验证为 12 场景、230 次调用，数值检查及 Checker 通过；该版本未重新跑完整矩阵。本次归档验证了源码包、19 文件清单、六个导出文件和 Git 提交内容的一致性；未新增完整 SDK 构建或硬件测试结论。

## 适用边界

这是比赛仿真环境的最终代码归档。小数据路径使用接收者自己的 output token 读取远端 root 输入，并允许 root 先于读者完成返回；比赛仿真通过不代表严格远端 token 校验或调用方提前复用输入缓冲时的生产环境安全性。移植到真实设备或其他运行环境前，需要重新验证远端内存授权、输入缓冲生命周期与跨调用同步。

## 参考与许可

- [赛事说明与代码归档要求](https://gitcode.com/cann/hccl/discussions/16)
- [初赛归档 PR #1522](https://gitcode.com/cann/cann-ops-competitions/pull/1522)
- 保留原工程 [CANN Open Software License Agreement Version 2.0](code/LICENSE)。
