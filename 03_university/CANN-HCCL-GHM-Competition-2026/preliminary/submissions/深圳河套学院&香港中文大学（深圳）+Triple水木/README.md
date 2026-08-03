# Triple水木——初赛代码归档

## 团队信息

- 赛区：2026 HCCL通信库创新大赛粤港澳赛区
- 队伍：Triple水木
- 归档学校：深圳河套学院&香港中文大学（深圳）
- 参赛成员：
  - Menphina（深圳河套学院）
  - qq_40734045（深圳河套学院）
  - dubai712（香港中文大学（深圳）、深圳河套学院）
- 联系方式：
  - Menphina：`wangyuansen@outlook.com`
  - qq_40734045：`zhaoyumiao99@gmail.com`
  - dubai712：`1006529762@qq.com`

## 作品简介

本作品面向 HCCL ReduceScatter 集合通信算子，在 AICPU 上完成 Host 侧资源申请和 Device 侧通信编排。工程保留比赛提供的构建框架，并在允许修改的 `custom.h`、`reduce_scatter.cc` 和 `exec_op.cc` 中实现队伍方案。

## 技术方案

- Host 侧根据通信规模和数据量计算资源及切片信息；
- AICPU Kernel 负责 ReduceScatter 通信任务编排；
- 通过分片、流水化和拓扑感知降低通信等待与额外搬运；
- 代码结构、构建脚本和测试说明均随工程归档，便于复现。

## 环境与运行

- CANN Toolkit：9.1.0 配套赛事版本；
- 硬件：赛事指定昇腾环境；
- 构建：进入 `code/hccl_reducescatter_aicpu` 后执行 `bash build.sh`；
- Debug 构建：执行 `bash build.sh --debug`。

详细环境变量、构建步骤和测试方法见工程内 `README.md` 与 `README-test.md`。

## 性能与验证

测试方法、用例覆盖和原始结果说明集中记录在 `code/hccl_reducescatter_aicpu/README-test.md`。本归档 README 不重新填写未经核验的成绩数据。

## 归档内容

本目录归档初赛 ReduceScatter AICPU 算子工程：

```text
code/hccl_reducescatter_aicpu/
├── include/
├── op_host/
├── op_kernel_aicpu/
├── CMakeLists.txt
├── build.sh
├── README.md
└── README-test.md
```

代码来源：`https://gitcode.com/Menphina/ReduceScatter` 的 `Preliminary` 目录。

构建、环境配置和测试说明请阅读 `code/hccl_reducescatter_aicpu/README.md` 与 `README-test.md`。

## 提交记录

- Menphina：`adf3c933fd94d5f655b049e2754e81ef4de858a8`
- qq_40734045：`ac7c910dfefe9aab77cedc87968e2269e028634b`
- dubai712：
  - `981ec8b5a00c4a1ff6ffec157185d706f0444ec8`
  - `d4cfc3adfd9ec370f70d439e97424fa0c5b1ea40`
  - `c29e3627b5561460e526270ac79e4f3f9bc80d25`

本次归档不包含编译产物、运行日志或访问凭据。
