# HCCL AllGather 初赛代码归档

## 团队信息

- 团队名称：GG
- 所属单位：北京科技大学
- 团队成员：段铸城（GitCode：`2402_87469259`）
- 参赛阶段：2026 HCCL 通信库创新大赛华北赛区初赛

## 作品信息

- 赛题：AllGather 集合通信算子
- 通信引擎：AICPU + TS
- 赛事平台提交编号：`110712`
- 代码目录：`code/hccl_allgather/`

本实现完成 AllGather 操作接口，将通信域内各 Rank 的输入按 Rank ID 顺序拼接，并将完整结果写入所有 Rank 的输出。代码针对初赛给定的 2 × 8 拓扑和不同数据规模组织通信与同步流程。

## 归档来源

- 原始归档：`hccl_allgather_submission_110712.zip`
- 原始归档 SHA-256：`1AB06C1257966C821AD4DBDAC1106D007B81B455D018769F6DCDB0561770D484`
- 工程文件数：20

本目录保留赛事平台下载包中的完整可构建工程结构。归档前已核对压缩包完整性、文件清单和路径安全性，未包含构建产物、缓存、日志或访问凭据。赛事成绩与提交状态以赛事平台的最终记录为准。

## 构建方法

配置 CANN Toolkit 9.1.0 环境后执行：

```bash
cd code/hccl_allgather
bash build.sh
```

如需格式化代码：

```bash
bash build.sh --format
```

## 验证说明

- 解压后的 20 个工程文件与原始 ZIP 文件清单一致。
- 已检查并排除路径穿越、符号链接、密钥、日志和二进制构建产物。
- 本地在 Alpine 3.24、GCC/G++ 15.2.0 与 CANN Toolkit 9.1.0 头文件环境中验证时，CMake 配置成功，Host 侧 `libhccl.so` 构建完成；Device 侧因当前 CANN 安装缺少 `aarch64-target-linux-gnu-gcc/g++` 工具链而无法继续。验证过程未修改源码。
- 在线功能与性能结果以赛事平台提交 `110712` 的最终记录为准。
