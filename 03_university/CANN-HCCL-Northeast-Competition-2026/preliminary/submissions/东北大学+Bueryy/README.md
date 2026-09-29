# 东北大学 Bueryy 队初赛代码归档

| 项目 | 内容 |
| --- | --- |
| 赛区 | 东北赛区 |
| 学校 | 东北大学 |
| 队伍 | Bueryy |
| 比赛阶段 | 初赛 |
| 归档版本（平台提交编号） | 263185 |
| 算子 | Scatter |
| 执行引擎 | AICPU |
| 团队成员账号 | SlienceTD（m0_611779081）、sssssssxz、Buery |

## 工程说明

本目录归档平台提交 263185 对应的 Scatter 源码。主机侧通过 `HcclScatter`
入口组织通信与执行资源，由 AICPU 侧代码完成相应的数据传输。
对外接口声明位于 `include/hccl.h`，实现入口位于 `op_host/scatter.cc`。

源码和构建文件直接位于本目录，不另设 `code/` 层级，也不包含本地交接包的外层文件夹。
官方归档路径为：

```text
03_university/CANN-HCCL-Northeast-Competition-2026/preliminary/submissions/东北大学+Bueryy/
```

## 文件结构

```text
东北大学+Bueryy/
├── README.md
├── LICENSE
├── build.sh
├── CMakeLists.txt
├── include/
├── op_host/
│   ├── CMakeLists.txt
│   ├── scatter.cc
│   ├── launch_aicpu_kernel.cc
│   └── launch_aicpu_kernel.h
└── op_kernel_aicpu/
    ├── CMakeLists.txt
    ├── aicpu_kernel.cc
    ├── exec_op.cc
    └── exec_op.h
```

- `include/`：算子接口、公共定义及本实现所需的头文件。
- `op_host/`：主机侧入口及执行组织代码。
- `op_kernel_aicpu/`：AICPU 侧执行代码。
- 根目录及子目录的 `CMakeLists.txt`：构建目标、源码列表、头文件路径及链接依赖。
- `build.sh`：配置、编译和安装到工程内部 `build/` 目录。

## 编译环境

- Linux，使用与赛事参赛环境兼容的 CANN Toolkit；CANN 的确切版本需以实际编译环境为准。
- CMake 3.16.0 或更高版本。
- 支持 C++17 的 GNU C/C++ 编译工具链及 Make。
- Bash、`nproc` 和支持长选项的 GNU `getopt`。
- 已通过 CANN 的 `set_env.sh` 初始化 `ASCEND_HOME_PATH`。
- CANN 主机侧头文件及 `${ASCEND_HOME_PATH}/lib64` 中的 `hcomm`、`acl_rt` 链接依赖。
- AICPU 交叉编译工具链：`${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/` 下的 `aarch64-target-linux-gnu-gcc`、`aarch64-target-linux-gnu-g++` 和 `aarch64-target-linux-gnu-ar`。
- AICPU 链接依赖：`${ASCEND_HOME_PATH}/devlib/device` 下的 `ccl_kernel` 库。

原始构建配置启用了 `-Wall`、`-Werror`，并为主机侧设置
`_GLIBCXX_USE_CXX11_ABI=0`；请保持编译器、头文件和链接库兼容。
macOS 可用于整理和上传源码，本构建流程应在上述 Linux/CANN 环境中执行。

## 编译步骤

先进入本 README 所在的工程根目录，再执行以下命令。
`/path/to/Ascend/cann/set_env.sh` 是占位路径，必须换成实际 CANN 安装中的脚本路径。

```bash
source /path/to/Ascend/cann/set_env.sh
printf '%s\n' "$ASCEND_HOME_PATH"
bash build.sh
```

第二条命令应输出有效的 CANN 安装路径；环境初始化失败或路径为空时，应先修正环境。
默认构建类型为 `Release`。脚本依次执行 CMake 配置、并行编译和安装；安装前缀为
本工程的 `build/`，不要求将产物安装到系统目录。

需要 Debug 构建时使用：

```bash
bash build.sh --debug
```

脚本也提供 `--format`，该选项会改写源码格式，不是编译选项；复现归档代码时使用上面的普通构建命令。

## 构建产物与使用

按当前 CMake 安装规则，成功构建后的主要产物为：

- `build/lib64/libhccl.so`：主机侧 Scatter 动态库。
- `build/lib64/libhccl_device.so`：AICPU 侧动态库。
- `build/include/hccl.h`：对外接口头文件。

该工程生成动态库，不包含独立的测试可执行程序。功能与性能验证应使用赛事配套的
测试程序及运行环境，并按其要求加载本工程产物。具体运行参数以该测试程序的说明为准。

## 本次归档核验

2026-09-23 已检查构建配置引用的本地源码路径，并完成 `bash -n build.sh` 语法检查。
本次整理没有执行 Linux/CANN 完整编译、HCCL-VM 数值验证或 NPU 真机性能测试；
平台提交编号仅用于版本对应，不作为本次重新验证通过的证明。

提交 PR 后还需按赛事要求完成 CLA 检查和 `/compile` 流水线，确认当前提交具备
`cann-cla/yes` 与 `ci-pipeline-passed` 标签；流水线通过范围以实际任务和日志为准。

## 归档范围与许可证

保留源码、头文件、构建脚本、CMake 配置、README 和 LICENSE。
不归档交接说明 `版本说明.md`、`.DS_Store`、本地 `.git/` 或 `build/` 构建产物。
许可证及版权信息以 `LICENSE` 和各源文件中的声明为准。
