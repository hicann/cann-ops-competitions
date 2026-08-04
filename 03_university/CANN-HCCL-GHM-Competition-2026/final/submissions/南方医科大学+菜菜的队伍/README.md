## 赛题说明

粤港澳赛区（ReduceScatter）- 决赛
一、赛题描述
实现集合通信算子ReduceScatter的操作接口，将通信域内所有rank的输入数据均分成rank size份，然后分别取每个rank的rank size之一份数据进行归约操作（如sum）。最后，将结果按照编号分散到各个rank的输出buffer。其计算原理如下图所示。

ops_pre.png

评测环境为4*8卡昇腾Ascend 950的仿真环境，拓扑如下所示，横向为Mesh互联，纵向为Clos组网。

拓扑由4个Server组成，每个Server内包含8个NPU。
Server内8个NPU之间为Full-Mesh互联，Server间通过Clos网络互通。
每个NPU连接Clos网络的带宽大致是Server内单条直连链路带宽的4倍。
基于完整32卡集群构建3种子拓扑：2 * 8、4 * 1、8 + 4。具体拓扑如下图所示。
拓扑类型：2*8fin_topo_1.png

拓扑类型：4*1fin_topo_2.png

拓扑类型：8+4fin_topo_3.png

二、核心定义与约束
2.1 函数原型与参数说明
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)

参数名	输入/输出	描述
sendBuf	输入	源数据buffer地址。
recvBuf	输出	目的数据buffer地址，集合通信结果输出至此buffer中。
recvCount	输入	参与ReduceScatter操作的recvBuf的数据size，sendBuf的数据size则等于recvCount * rank size。
dataType	输入	ReduceScatter操作的数据类型，HcclDataType类型。
op	输入	Reduce的操作类型，如sum、prod、max、min。
comm	输入	集合通信操作所在的通信域。
stream	输入	本rank所使用的stream。
接口成功返回HCCL_SUCCESS，其他失败。

2.2 算子约束
所有rank的recvCount、dataType、op均相同。
针对一个对端仅能申请1个channel进行通信。基于赛题提供的拓扑，每两个npu之间仅有一条物理链路，因此与同一个对端只使用一条channel完成通信的收/发、同步等操作即可，使用多个channel不会带来额外的性能收益。
2.3 规则要求
限定使用CCU通信引擎

算子实现需满足确定性要求，即在相同输入下（特别是浮点数输入），多次通信计算得到的输出结果相同

三、评分指标
赛题将从以下三个维度进行评分：

功能分：通过功能测试用例验证，通过得分，不通过得0分。
性能分：通过性能测试用例验证，不通过得0分，通过则按照带宽用量计分，带宽最高得满分，按照排名依次递减。
如不满足2.3中的规则要求，则不得分。

通信域覆盖3种不同拓扑类型，数据类型覆盖float32，（输入）数据size覆盖512KB、512MB、400MB+4B，Reduce类型为sum。

3.1 用例说明
功能用例：

共9个功能用例（数据量分别为512KB、512MB、400M+4B），覆盖三种拓扑类型。
代码提交后，等待实时判分。
性能用例：

共9个性能用例（数据量分别为512KB、512MB、400M+4B），覆盖三种拓扑类型。
每天统一出1次性能分，取第二天凌晨3点前最后1次提交的代码进行性能评分。
需保证功能用例全部通过才能参与性能评分。
四、参考资料
1. 📚 前置基础学习链接
资料名称	链接
全量学习资料(持续更新中)	HCCL 学习资料
HCCL—昇腾高性能集合通信库简介	昇腾高性能集合通信库简介
深度学习的分布式训练与集合通信（一）	深度学习的分布式训练与集合通信（一）
深度学习的分布式训练与集合通信（二）	深度学习的分布式训练与集合通信（二）
2. 📚 代码仓资料
资料名称	链接
HCCL代码仓库	CANN/hccl仓、CANN/hcomm仓
通信算子开发文档	HCCL 算子开发文档
3. 🚀 赋能视频链接
主题	直播回放	材料归档
HCCL软件架构介绍	HCCL软件架构赋能视频	HCCL软件架构赋能材料
HCCL算子开发介绍	HCCL算子开发赋能视频	HCCL算子开发赋能材料
HCCL北极星验证工具介绍	HCCL北极星验证工具赋能视频	HCCL北极星验证工具赋能材料
# ReduceScatter 集合通信算子

## 1. 项目介绍

```
├── CMakeLists.txt                  # 顶层 CMake 配置
├── build.sh                        # 构建脚本
├── .clang-format                   # 代码风格配置
├── include/                        # 头文件目录
│   ├── hccl.h                      # 集合通信算子头文件
│   ├── common.h                    # 通用数据结构定义
│   ├── custom.h                    # ★ 选手编写：自定义数据结构定义
│   ├── log.h                       # 日志宏定义
│   └── binary_stream.h             # 序列化类定义
├── op_host/                        # Host侧代码目录
│   ├── reduce_scatter.cc           # ★ 选手编写：Host侧资源申请逻辑
│   └── exec_op.cc                  # ★ 选手编写：通信算法编排逻辑
└── op_kernel_ccu/                  # CCU侧代码目录
    └── ccu_kernel.cc               # ★ 选手编写：通信算法编排逻辑
```

> [!NOTE] 注意：
> 算子工程中已提前预制好固有逻辑，选手仅允许修改 `custom.h`、`reduce_scatter.cc`、`exec_op.h`、`exec_op.cc`、`ccu_kernel.h`、`ccu_kernel.cc` 共 6 个文件内容。

## 2. 编译运行

### 2.1 安装 CANN-Toolkit 包

请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260701000328953/)，根据产品型号和环境架构下载对应软件包。安装命令如下，更多指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-${arch}.run
# 安装命令
./Ascend-cann-toolkit_9.1.0_linux-${arch}.run --full --install-path=${install_path}
```

### 2.2 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

### 2.3 编译算子工程

```bash
bash build.sh

# 编译 Debug 版本，便于断点调试
bash build.sh --debug
```

## 3. 代码格式

选手代码需符合 [.clang-format](.clang-format) 文件中的代码风格规范，可通过下列命令一键修改：

```bash
bash build.sh --format
```
