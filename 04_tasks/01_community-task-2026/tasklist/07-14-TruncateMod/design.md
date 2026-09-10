# TruncateMod 算子设计方案

# 1 需求背景（required）

## 1.1 需求来源

TruncateMod算子是PyTorch框架中的取余算子，返回self除以other的截断余数。PyTorch原生接口：

> **torch.fmod(self, other)**

（注：PyTorch中`torch.fmod`实现向零取整的余数计算，与C语言`fmod`语义一致。`torch.remainder`实现向负无穷取整，与本算子语义不同。）

计算公式（向零取整）：

> y = self - other * trunc(self / other)

其中trunc为向零取整（truncation toward zero），余数符号与被除数self相同。

支持浮点（fp16/fp32/bf16）与全套整型（i8/u8/i32/i64），广泛用于循环索引、数据分桶、量化网络余数映射等。

本次基于TBE版本全量重新开发为Ascend C算子，解决原TBE三大短板：未区分浮点/整型双路径、无整型除零检测、UB固定4B不兼容多整型位宽。

对齐PyTorch完整语义（取整方向、余数符号、广播、整型溢出Wrap Around、浮点除零IEEE 754+NaN传播、+0/-0规则），通过Aclnn适配PyTorch，支持ATC推理与Aclnn直调。依赖CANN 8.0.RC1及以上。

## 1.2 背景介绍

### 1.2.1 TruncateMod算子实现优化

基于TruncateMod算子历史TBE版本全量重新开发为Ascend C算子。参考实现：https://gitcode.com/cann/ops-math/tree/master/math/truncate_mod

### 1.2.2 TruncateMod算子现状分析

TBE版本支持：self/other支持DOUBLE/BF16/F16/F32/I32/I64/I8/U8；广播0-8维；向零取整。三大短板：未区分浮点/整型计算路径；无整型除零检测导致硬件异常风险；UB固定按4B计算不兼容i8(1B)/u8(1B)/i64(8B)。

---

# 2 需求分析

## 2.1 外部组件依赖

不涉及外部组件依赖。

## 2.2 内部适配模块

适配Aclnn接口和图模式调用，通过Aclnn桥接PyTorch框架。

## 2.3 需求模块设计

### 2.3.1 算子原型

| **名称** | **类别** | **dtype** | **format** | **shape** | **介绍** |
|---------|---------|-----------|-----------|-----------|---------|
| self | 输入 | fp16/fp32/bf16/i32/i64/i8/u8 | ND | 0-8维 | 被除数 |
| other | 输入 | fp16/fp32/bf16/i32/i64/i8/u8 | ND | 0-8维 | 除数 |
| out | 输出 | fp16/fp32/bf16/i32/i64/i8/u8 | ND | 0-8维 | 余数输出，dtype与计算类型一致 |

**完整混合dtype提升规则（优先级由低到高）：**

> u8 < i8 < i16 < i32 < i64 < fp16 = bf16 < fp32

(1) u8+i8：固定提升至i16（对齐PyTorch类型提升规则）。其余整型：i8+i32升i32；i32+i64升i64。向位宽更大整型提升。

(2) 浮点内部：fp16/bf16混合均直接升fp32。纯fp32保持。

(3) 浮点+整型混合：所有整型先Cast为fp32，整体走浮点计算路径。

(4) 高位整型混合回写：大位宽整型计算后，输出dtype严格等于提升后的计算类型。

(5) Cast舍入模式：RN（round-to-nearest-even），对齐PyTorch。

**整型溢出规则：**

有符号整型（i8/i32/i64）溢出采用循环截断（Wrap Around），对齐PyTorch。无符号u8按无符号循环取模。

**溢出检测模式：**
- DEBUG编译模式：Kernel侧逐元素判断溢出（结果超出类型表示范围则为溢出），统计溢出元素数量并输出DEBUG日志。
- Release编译模式：不检测溢出，依赖硬件补码运算自然Wrap Around，零额外开销。

u8专属规则：u8范围[0,255]。运算中间结果出现负数时，按u8无符号规则循环映射（即(result + 256) % 256，如-1→255）。DEBUG模式逐元素检测并计数。

格式约束：仅ND连续。非ND/E10001，非连续/E10002。维度0-8，满足广播。

**空张量规则：**
- numel==0（全空张量）：Host侧直接分配输出空内存并返回，不进Kernel。
- 部分空+非空按广播校验，不兼容返回E10007。
- 空张量输出分配合法空内存（非野指针）。
- 非空张量任意指针NULL返回E10005。

约束：Atlas A2/800I A2。CANN不小于8.0.RC1。浮点other含0按IEEE 754(+0/other=+Inf, -0/other=-Inf, 0/0=NaN, NaN传播)。整型other含0返回E10011。余数符号跟随self。

---

# 3 接口定义（API Specification）

## 3.1 Aclnn接口签名

```cpp
aclnnStatus aclnnTruncateMod(const aclTensor* self,
                              const aclTensor* other,
                              aclTensor* out,
                              uint64_t workspaceSize,
                              aclOpExecutor* executor);
```

## 3.2 参数说明

| **参数** | **类型** | **说明** |
|---------|---------|---------|
| self | const aclTensor* | 被除数输入张量，对应算子原型self |
| other | const aclTensor* | 除数输入张量，对应算子原型other |
| out | aclTensor* | 输出张量，对应算子原型out |
| workspaceSize | uint64_t | 工作空间大小，由aclnnTruncateModGetWorkspaceSize获取 |
| executor | aclOpExecutor* | 算子执行器，由aclnnTruncateModGetWorkspaceSize创建 |

## 3.3 返回值

| **返回值** | **说明** |
|-----------|---------|
| ACLNN_SUCCESS | 执行成功 |
| ACLNN_INVALID_PARAM | 参数错误（如NULL指针、不支持的dtype） |
| 其他错误码 | 映射到自定义错误码E10001~E10011 |

## 3.4 错误码映射

| **自定义错误码** | **Aclnn错误码** | **场景** |
|----------------|----------------|---------|
| E10001 | ACLNN_INVALID_PARAM | 非ND格式 |
| E10002 | ACLNN_INVALID_PARAM | 内存非连续 |
| E10003 | ACLNN_NOT_SUPPORT | CANN版本低 |
| E10004 | ACLNN_NOT_SUPPORT | 硬件不支持bf16 |
| E10005 | ACLNN_INVALID_PARAM | 非空张量指针NULL |
| E10006 | ACLNN_INVALID_PARAM | 维度超过8维 |
| E10007 | ACLNN_INVALID_PARAM | 广播不兼容 |
| E10009 | ACLNN_INTERNAL_ERROR | UB越界 |
| E10010 | ACLNN_INTERNAL_ERROR | 广播地址越界 |
| E10011 | ACLNN_INVALID_PARAM | 整型除数other=0（标量场景Host拦截） |
| 561002 | ACLNN_INTERNAL_ERROR | Tiling异常 |

---

# 4 算子注册信息

## 4.1 基本信息

| **属性** | **值** |
|---------|--------|
| op_type | TruncateMod |
| 输入数 | 2（self, other） |
| 输出数 | 1（out） |
| 支持format | ND |
| 支持维度 | 0-8维 |
| 支持广播 | 是 |

## 4.2 支持的数据类型组合

| **self dtype** | **other dtype** | **计算类型** | **out dtype** | **说明** |
|---------------|----------------|-------------|--------------|---------|
| fp16 | fp16 | fp32 | fp16 | 内部升fp32计算，输出Cast回fp16 |
| fp32 | fp32 | fp32 | fp32 | 纯fp32路径 |
| bf16 | bf16 | fp32 | bf16 | 内部升fp32计算，输出Cast回bf16 |
| fp16 | bf16 | fp32 | fp32 | 混合浮点升fp32 |
| i8 | i8 | i8 | i8 | 纯整型路径 |
| u8 | u8 | u8 | u8 | 纯整型路径 |
| i32 | i32 | i32 | i32 | 纯整型路径 |
| i64 | i64 | i64 | i64 | 纯整型路径 |
| i8 | i32 | i32 | i32 | 整型提升 |
| i32 | i64 | i64 | i64 | 整型提升 |
| u8 | i8 | i16 | i16 | u8+i8固定升i16 |
| fp32 | i32 | fp32 | fp32 | 浮点+整型混合，整型Cast为fp32 |
| bf16 | i64 | fp32 | fp32 | 浮点+整型混合，整型Cast为fp32 |

---

# 5 需求详细设计

## 5.1 使能方式

| **上层框架** | **勾选** | **备注** |
|-------------|---------|---------|
| PyTorch | v | Aclnn桥接，对应torch.fmod |
| ATC推理 | v | 全类型 |
| Aclnn直调 | v | 全类型 |

## 5.2 需求总体设计

### 5.2.1 Host侧设计

**Tiling策略：**

(1) 广播shape：维度不足者首部补1至最大维数，逐维取最大值。无广播校验shape一致。

(2) 数据类型识别：遍历输入dtype确定计算路径（纯整型/浮点），通过类型标记传Kernel。

**分核计算公式：**

> coreNum = min(ceil(total_length / MIN_BLOCK_ELEMS), AI_CORE_MAX)
>
> blocksPerCore = floor(total_length / coreNum)
>
> remainder = total_length - blocksPerCore * coreNum

AI_CORE_MAX=40，MIN_BLOCK_ELEMS=256。

余数分配：remainder个剩余有效元素由前remainder个AI Core各多处理1个元素，其余核处理基准块。不存在补零操作，所有数据均为有效元素。

极小数据兜底：total_length小于MIN_BLOCK_ELEMS时，单Core运行，支持标量/短向量指令路径。

**UB内存优化策略：**

A2单核UB 256KB。位宽按计算路径动态取值：浮点路径elemByte=4B；纯整型按dtype: i8/u8=1B, i16=2B, i32=4B, i64=8B。

UB容量校验公式（Host Tiling阶段预计算，防止Kernel越界）：

> 单缓冲：UB字节数 = 4 * elemByte * tile_elems（2路输入+1路临时+1路输出）
>
> UB越界判断：单缓冲字节数 > 256KB 则 Tile过大，返回E10009
>
> 双缓冲开启条件：单缓冲字节数 <= 128KB 且 total_length >= 512

i64(8B)动态限制：i64场景下主动缩小Tile元素数（上限=256KB/(4*8B)约8192），Tiling阶段预校验。

**边界短路优化（Host全局判断，仅标量场景）：**

(1) self为标量0且非空：直接输出0，不进Kernel。

(2) other为标量1：结果=self；other为标量-1：结果=-self。

**TilingKey位域定义（8位）：**

| **位域** | **宽度** | **说明** |
|---------|---------|---------|
| bit[1:0] | 2位 | 场景：00=同shape+同dtype, 01=有广播+同dtype, 10=同shape+混合dtype, 11=有广播+混合dtype |
| bit[2] | 1位 | 计算路径：0=浮点路径, 1=整型路径 |
| bit[3] | 1位 | 双缓冲开关：0=单缓冲, 1=双缓冲 |
| bit[7:4] | 4位 | 预留 |

**异常拦截（均在Host Tiling阶段执行）：**

(1) 整型除零：
- other为标量（0维或shape全1）：Host侧检查值是否为0，是则返回E10011，不进Kernel。
- other为张量：不在Host侧遍历，交由Kernel侧处理（整型除法指令遇0行为由硬件定义，需确保不崩溃）。

(2) UB越界：Tiling阶段预计算UB占用，超出返回E10009。

(3) 广播地址越界：Tiling阶段预校验地址映射范围，超出返回E10010。

(4) 指针判空：非空张量任意NULL返回E10005。

**错误码体系：**

| **错误码** | **场景** |
|-----------|---------|
| E10001 | 非ND格式 |
| E10002 | 内存非连续 |
| E10003 | CANN版本低 |
| E10004 | 硬件不支持bf16 |
| E10005 | 非空张量指针NULL |
| E10006 | 维度超过8维 |
| E10007 | 广播不兼容 |
| E10009 | UB越界 |
| E10010 | 广播地址越界 |
| E10011 | 整型除数other=0（标量场景Host拦截） |
| 561002 | Tiling异常 |

日志：SIMPLE_LOG_SWITCH开关，路径/var/log/npu/op/truncatemod.log。Host(INFO)：版本/TilingKey/coreNum/UB模式/短路/整型除零拦截。Kernel(DEBUG)：CopyInOut数据量/Compute耗时/整型溢出数/u8负数映射数/NaNInf计数。

### 5.2.2 Kernel侧设计

**标准Init+Process(CopyIn/Compute/CopyOut)三阶段。空张量在Kernel入口直接短路退出（兜底）。**

**CopyIn阶段--广播寻址策略：**

广播张量内存非连续，CopyIn阶段对广播数据做预重排（多维索引地址映射），配合双缓冲流水线优化访存。非广播场景直接连续搬入。

**路径1--浮点路径（纯浮点或浮点+整型混合）：**

统一升fp32计算。trunc向零取整使用Ascend C指令实现。

> trunc_div = trunc(self / other)  // vdiv + trunc指令
>
> mul_tmp = other * trunc_div  // vmul
>
> out = self - mul_tmp  // vsub

**trunc指令实现：**
- 首选：Ascend C `vtrunc` 指令（如果硬件支持，直接截断小数部分）
- 备选：fcvt.rz 指令（round toward zero，浮点转整型再转回浮点）
- 兜底：floor/ceil组合实现（`trunc(x) = x < 0 ? ceil(x) : floor(x)`）

浮点除零+0/-0规则（对齐PyTorch）：other=+0时self正/0/负对应+Inf/NaN/-Inf；other=-0符号取反。NaN参与运算结果NaN。

**路径2--纯整型路径（i8/u8/i32/i64）：**

昇腾AI Core整型除法指令默认向零取整，无须额外trunc。

> int_div = self / other  // 整型除法(硬件向零取整)
>
> mul_tmp = other * int_div  // 整型乘法
>
> out = self - mul_tmp  // 整型减法

整型溢出逐元素判断（DEBUG模式）：结果超出该整型表示范围标记溢出，Wrap Around处理。u8负数映射：中间结果小于0时按u8循环映射((result + 256) % 256)。Kernel DEBUG日志记录两类计数。

**整型除零处理（张量场景）：**
- Kernel侧不主动检测除零（避免分支开销）
- 依赖硬件整型除法指令行为（通常产生异常或不确定值）
- 确保：标量除零已在Host侧拦截，张量除零场景极少，且PyTorch行为亦未定义

bf16溢出阈值：+/-3.38953139e38（昇腾硬件bf16最大正规数）。正/负溢出截断，下溢截断0或最小正规数。硬件矢量指令保留NaN/Inf。

vdiv精度：对比PyTorch fp64基准10000组。fp16<1e-3/fp32<1e-6/bf16<1e-2。整型输出与PyTorch基准逐元素完全一致。

双缓冲Ping-Pong交替，硬件Barrier同步，读写互斥。

## 5.3 支持硬件

| **芯片** | **AI Core** | **UB** | **勾选** |
|---------|------------|--------|---------|
| Atlas800I/T A2 | 40 | 256KB | v |
| Atlas A2训练 | 40 | 256KB | v |

## 5.4 约束限制

0-8维仅ND连续。CANN不小于8.0.RC1。浮点除零IEEE 754含+0/-0规则+NaN传播。整型除零：标量场景Host拦截E10011，张量场景依赖硬件行为。余数符号跟随self。整型溢出Wrap Around。u8负数循环映射。空张量Host侧直接返回。非空NULL/E10005。

---

# 6 特性交叉分析

## 6.1 与其他算子的关系

TruncateMod算子可拆解为基础算子组合：Div → Trunc → Mul → Sub。但作为独立算子，优势在于：
- 单Kernel完成全部计算，减少GM-UB往返
- 整型路径利用硬件向零取整特性，无需显式trunc指令
- 统一的数据类型提升和广播处理

## 6.2 依赖关系

无外部算子依赖。内部依赖CANN基础运行时（AscendCL、Runtime）。

---

# 7 可维可测分析

## 7.1 精度/性能标准

验收：精度不低于TBE；性能Kernel耗时不超过TBE90%/加速比不小于核数x0.6（40核目标24倍）/带宽不小于理论峰值70%。同芯片+同CANN+同编译对比。

误差：相对误差=abs(out-base)/max(abs(base),1e-8)。基准PyTorch fp64。

| **dtype** | **阈值** | **说明** |
|----------|---------|---------|
| fp16 | < 1e-3 | 浮点 |
| fp32 | < 1e-6 | 浮点 |
| bf16 | < 1e-2 | 浮点 |
| i8/u8/i32/i64 | 完全一致 | 整型精确计算，输出与PyTorch基准逐元素相等 |
| 混合dtype（浮点+整型） | < 1e-6 | 整型Cast为fp32计算，按fp32标准 |

性能分三档(<1K/1K-1M/>1M)。小规模调度占比=调度耗时/总耗时放宽至50%。中大规模Compute占比不小于80%。预热5次采样50次。广播/无广播分别统计。

**性能指标说明：**
- "加速比不小于核数x0.6"：考虑Amdahl定律（串行部分约10%），40核理论最大加速比约10倍，实际目标24倍需确保并行度>95%
- 小规模数据(<1K)因调度开销，加速比要求放宽至"不小于5倍"

## 7.2 兼容性

**CANN版本兼容**：向前兼容CANN 8.0.RC1及以上版本。低版本返回E10003。

**硬件兼容**：仅Atlas A2/800I A2系列。其他硬件返回E10004（bf16不支持时）。

**算子版本兼容**：v1.0→v1.3接口保持兼容，仅内部实现优化。

---

# 8 测试方案

## 8.1 框架

昇腾算子测试框架：功能+精度+性能。

## 8.2 用例覆盖

| **维度** | **覆盖** |
|---------|---------|
| dtype | fp16/fp32/bf16/i32/i64/i8/u8全量 |
| 混合dtype | 浮点内部/整型内部/浮点+整型；u8+i8/i8+i64/u8+i64组合 |
| 规模 | 空/极小(<256)/小(256-1K)/中(1K-1M)/大(>1M) |
| 广播 | 无广播/标量/多维/部分空 |
| 0维标量 | self/other组合标量 |
| 边界 | 单元素/全1/全0/负值/other=1/other=-1/self=0(短路)/极值(最大/最小) |
| 整型专项 | 溢出(Wrap Around)/u8负数映射/整数除零(E10011) |
| 浮点专项 | NaN传播/Inf传播/+0/-0除零规则验证 |
| 内存对齐 | shape=[1,31,32,33,63,64,65]等32B对齐边界 |
| 双缓冲边界 | total_length=[511,512,513]验证开关 |
| UB越界边界 | Tile大小=[255KB,256KB,257KB]验证E10009 |
| 多核不均衡 | total_length=[1,255,256,257,1023,1024,1025] |
| 异常 | 非ND/非连续/CANN版本/UB越界(E10009) |
| 长稳 | 1000次循环验证内存/UB/时序 |
| 并发 | 多线程线程安全 |

## 8.3 调用方式

| **方式** | **说明** |
|---------|---------|
| geir | 基础 |
| Aclnn | 图模式 |
| ATC | 推理 |
| PyTorch | 全链路，对应torch.fmod |

---

# 9 编译构建说明

## 9.1 依赖环境

| **依赖项** | **版本要求** | **说明** |
|-----------|-------------|---------|
| CANN | >= 8.0.RC1 | 昇腾AI软件栈 |
| CMake | >= 3.16 | 构建工具 |
| 编译器 | g++ / aarch64-linux-gnu-gcc | 支持C++11及以上 |
| Ascend C工具链 | 随CANN安装 | 算子编译工具 |

## 9.2 编译命令

```bash
# 1. 配置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置
cmake .. -DCMAKE_BUILD_TYPE=Release          -DCMAKE_INSTALL_PREFIX=/usr/local/Ascend/opp/

# 4. 编译
make -j$(nproc)

# 5. 安装
make install
```

## 9.3 验证

```bash
# 运行功能测试
cd tests && python test_truncate_mod_st.py

# 运行精度测试
python test_truncate_mod_precision.py

# 运行性能测试
python test_truncate_mod_performance.py
```

---

# 10 提交目录结构

```
YourTeamName/
├── docs/
│   ├── aclnnTruncateMod.md      # API接口文档（必选）
│   ├── design.md                # 设计文档（必选）← 本文档
│   └── op_info.md              # 算子信息说明（建议新增）
│
├── examples/
│   ├── aclnn_example.cpp         # Aclnn直调示例（必选）
│   ├── atc_example.sh           # ATC推理示例（必选）
│   └── README.md               # 示例说明（建议新增）
│
├── op_host/                    # Host端实现（必选）
│   ├── truncate_mod_tiling.h    # TilingData结构体定义
│   ├── truncate_mod_tiling.cc   # Tiling策略实现（分核/UB计算/类型识别）
│   ├── truncate_mod_op_info.h   # 算子注册信息（op_type/输入输出/dtype组合）
│   ├── truncate_mod_op_info.cc  # 算子注册实现
│   └── truncate_mod.cpp        # Host侧主入口（参数校验/Tiling调用/空张量处理）
│
├── op_kernel/                  # Kernel端实现（必选）
│   ├── truncate_mod_kernel.h    # Kernel侧头文件（宏定义/结构体/函数声明）
│   ├── truncate_mod_kernel.cc   # Kernel实现（Init/Process/双路径/溢出检测）
│   └── truncate_mod_common.h    # 公共定义（错误码/常量/位宽宏）
│
├── tests/                      # 测试用例（可选但建议保留）
│   ├── st/                     # 系统测试
│   │   └── test_truncate_mod_st.py
│   └── ut/                     # 单元测试
│       └── test_truncate_mod_ut.cpp
│
├── CMakeLists.txt              # 构建配置（必选）
│   # 关键配置：
│   # - 编译器选项：-O2 -fPIC
│   # - 依赖库：ascendcl
│   # - 目标架构：aarch64
│   # - 算子注册：opp包结构
│
└── README.md                    # 算子说明（必选）
    # 内容：功能简介/支持类型/编译步骤/使用示例/性能数据/已知限制
```

---

# 11 风险分析

## 11.1 技术风险

| **风险项** | **影响** | **概率** | **缓解措施** |
|-----------|---------|---------|-------------|
| trunc指令不存在 | 浮点路径无法实现向零取整 | 中 | 已准备3套备选方案（vtrunc/fcvt.rz/floor+ceil组合） |
| i64位宽导致Tile过小 | i64场景性能下降50%+ | 高 | Tile上限8192，确保UB利用率；必要时单缓冲兜底 |
| 混合dtype提升规则与PyTorch不一致 | 精度验收失败 | 低 | 与PyTorch基准全量对比测试，覆盖所有dtype组合 |
| 整型除零张量场景未定义 | 硬件异常或不确定值 | 低 | 标量场景Host拦截；张量场景文档明确行为 |
| 双缓冲开启条件过严 | 性能未达预期 | 中 | 动态调整阈值，单缓冲兜底保证正确性 |

## 11.2 缓解措施优先级

1. **高优先级**：验证trunc指令可用性（开发第一周完成）
2. **高优先级**：i64场景性能基准测试（开发第二周完成）
3. **中优先级**：混合dtype全量对比测试（开发第三周完成）
4. **低优先级**：整型除零张量场景行为确认（开发第四周完成）

---

# 附录A：关键宏定义

```c
#define AI_CORE_MAX 40
#define UB_CAPACITY_KB 256
#define MIN_BLOCK_ELEMS 256
#define SMALL_SCALE 1024
#define LARGE_SCALE 1048576
#define DOUBLE_BUF_MIN_ELEMS 512

// Byte-width macros
#define I8_BYTES 1
#define U8_BYTES 1
#define I16_BYTES 2
#define I32_BYTES 4
#define I64_BYTES 8
#define FP32_BYTES 4

// UB derived macros
#define UB_TOTAL_BYTES (UB_CAPACITY_KB * 1024)
#define UB_HALF_BYTES (UB_TOTAL_BYTES / 2)

// TilingKey bit fields
#define TILING_KEY_SCENE_MASK 0x03      // bit[1:0]
#define TILING_KEY_PATH_MASK 0x04       // bit[2]
#define TILING_KEY_DOUBLE_BUF 0x08    // bit[3]

// Scene values
#define SCENE_SAME_SHAPE_SAME_DTYPE 0x00
#define SCENE_BROADCAST_SAME_DTYPE 0x01
#define SCENE_SAME_SHAPE_MIX_DTYPE 0x02
#define SCENE_BROADCAST_MIX_DTYPE 0x03

// Path values
#define PATH_FLOAT 0x00
#define PATH_INTEGER 0x04

// Error codes
#define E10001_FORMAT_NOT_ND 10001
#define E10002_NOT_CONTIGUOUS 10002
#define E10003_CANN_VERSION_LOW 10003
#define E10004_BF16_NOT_SUPPORTED 10004
#define E10005_NULL_POINTER 10005
#define E10006_DIM_EXCEED 10006
#define E10007_BROADCAST_INCOMPAT 10007
#define E10009_UB_OVERFLOW 10009
#define E10010_ADDR_OVERFLOW 10010
#define E10011_INT_DIV_ZERO 10011
#define E561002_TILING_FAILED 561002
```

---

# 附录B：参考资料

1. PyTorch文档 - torch.fmod: https://pytorch.org/docs/stable/generated/torch.fmod.html
2. PyTorch文档 - torch.remainder: https://pytorch.org/docs/stable/generated/torch.remainder.html
3. CANN Ascend C编程指南: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/
4. IEEE 754浮点运算标准: IEEE Std 754-2019
5. 参考实现 - ops-math/truncate_mod: https://gitcode.com/cann/ops-math/tree/master/math/truncate_mod
6. 昇腾算子开发规范: CANN算子开发指南
