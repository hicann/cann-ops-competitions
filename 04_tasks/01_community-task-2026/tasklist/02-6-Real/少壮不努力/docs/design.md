# Real算子AscendC实现设计文档

## 一、 需求背景

### 1.1 需求来源

通过社区任务完成开源仓算子贡献的需求

### 1.2 背景介绍

#### 1.2.1 Real算子实现优化

基于Real算子历史TBE版本使用AscendC 编程语言进行优化
Real算子（TBE）实现路径和相关API路径

- Real算子实现路径为: /usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/real.py
- Real算子实现中的API路径: /usr/local/Ascend/cann-9.0.0/python/site-packages/tbe/dsl/compute/math.py

#### 1.2.2 Real算子现状分析

通过对Real算子TBE版本的功能分析，当前支持的能力如下：

- Input是float16/float32数据类型的时候，执行VS_add(x, 0)操作，直接返回原数据
- Input是complex32/complex64 数据类型的时候，创建TVM placeholder 调用 tbe.real(input) api生成 elsewise_single_real IR指令
- 自动调优，调用tbe.auto_schedule(res)进行自动调优

Real算子TBE版本的整体流程图如下图所示:

![math_real_docs_tbe.png](https://raw.gitcode.com/user-images/assets/7649531/7cd7cd29-5667-4972-8738-24e43e928da1/math_real_docs_tbe.png 'math_real_docs_tbe.png')

## 二、 需求分析

### 2.1 外部组件依赖
不涉及外部组件适配

### 2.2 内部适配模块

内部aclnn接口已有，其他tiling，kernel，graph图需要适配

### 2.3 需求模块设计

#### 2.3.1 AscendC算子原型

| 名称 | 类别 | dtype | format | shape | 介绍
|--|--|--|--|--|--|
| input | 输入 | fp16/fp32/complex32/complex64 | ND | all | 输入tensor
| Tout | 输入 | Int | ND | (1,) | 属性输入
| output | 输出 | fp16/fp32 | ND | 同输入 | 输出tensor

- 相关约束:
    Atlas A2训练系列产品/Atlas 800I A2推理产品float16、float32、complex32、complex64

## 三、 需求详细设计

### 3.1 使能方式
| 上层框架 | 涉及的框架 |
|---------|---------|
| **TF训练/推理** | 不涉及 |
| **PyTorch训练/推理** | 涉及 |
| **ATC推理** | 涉及 |
| **Aclnn直调** | 涉及 |
| **OPAT调优** | 不涉及 |
| **SGAT子图切分** | 不涉及 |

### 3.2 需求总体设计

#### 3.2.1 host侧设计

##### 3.2.1.1 分核策略

参考库上类似算子Range，LinSpace，128个数据对齐，少于128个数据的单核，大于128数据的按128对齐后均分核

```
totalLength: 总元素个数
totalCoreNum: AI Core总数
usedCoreNum: 实际使用的核数

分核逻辑：
if (totalLength <= 128) {
    usedCoreNum = 1;  // 小数据量单核处理
} else {
    // 128个数据对齐切分
    numOfPerCore = Align128Ceil(totalLength / totalCoreNum);
    usedCoreNum = min(totalLength / numOfPerCore, totalCoreNum);
}
```
##### 3.2.1.2 数据分块和内存优化策略

UB总空间减去预留空间，再除以输出数据类型的字节数，再除以计算中间ub占用块（需要考虑开启Double Buffer）的情况，
对于非complex输入，数据直接拷贝到输出GM，需要UB空间为2（Double Buffer）
对于comoplex输入，数据需要拷贝到ub，提取实数，再拷贝输出，需要UB空间为 3(输入+输出，其中输入是输出的2倍) * 2（Double Buffer）= 6


| 输入类型 | inQueue | outQueue | 总倍数 | 说明 |
|---------|---------|----------|:------:|------|
| complex | 2x | 1x | 6 | 输入需存储完整复数（2x），双缓冲 |
| real | 共享 | 共享 | 2 | TQueBind复用buffer |

具体实现公式参考如下代码块
```cpp
// 根据输入类型选择不同的UB倍数
bool isComplexInput = (inputDtype == DT_COMPLEX32 || inputDtype == DT_COMPLEX64);
int64_t ubMultiplier = isComplexInput ? 6 : 2;  // complex需要额外空间存放复数数据

// 计算单次循环可处理的元素数
int64_t ubAvailable = totalUbSize - RESERVED_UB_SIZE;  // 预留256字节
int64_t ubOneLoopNum = Align128FloorSize(ubAvailable / (outputSize * ubMultiplier));
```
##### 3.2.1.3 tilingKey规划策略

根据输入类型确定tilingKey
| 输入类型 | tilingKey |
|---------|---------|
| complex32 | 1 |
| complex64 | 2 |
| complex128 | -预留 |
| float16 | 4 |
| float32 | 5 |

```cpp
enum class RealTilingKey : int64_t {
    TILINGKEY_COMPLEX32 = 1,   // complex32 -> float16
    TILINGKEY_COMPLEX64 = 2,   // complex64 -> float
    TILINGKEY_FLOAT16 = 4,     // float16 -> float16 (identity)
    TILINGKEY_FLOAT = 5        // float -> float (identity)
};
```

#### 3.2.2 kernel侧设计

##### 3.2.2.1 kernel侧实现描述

Real算子针对不同输入类型采用不同的处理策略：

1. **复数输入（complex32/complex64）**：
   - 使用 `GatherMask` API提取复数实部
   - 复数在内存中以交织方式存储：`[real0, imag0, real1, imag1, ...]`
   - 通过mask模式提取偶数索引位置的元素（实部）

2. **实数输入（float/float16）**：
   - 使用 `TQueBind` 实现零拷贝identity操作
   - 输入输出共享同一块UB buffer，减少内存拷贝开销

##### 3.2.2.2 AscendC实现流程图
![math_real_docs_ascendc.png](https://raw.gitcode.com/user-images/assets/7649531/633dba3e-ad2c-4881-8ace-329f2e028864/math_real_docs_ascendc.png 'math_real_docs_ascendc.png')

##### 3.2.2.3 TBE与AscendC实现流程图存在的差异点和原因

**整体架构对比：**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          实现架构对比                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TBE 架构:                    AscendC 架构:                                 │
│  ┌──────────┐               ┌──────────┐                                   │
│  │ 用户代码  │               │ 用户代码  │                                   │
│  │(Python)  │               │  (C++)   │                                   │
│  └────┬─────┘               └────┬─────┘                                   │
│       │                          │                                          │
│       ▼                          ▼                                          │
│  ┌──────────┐               ┌──────────┐                                   │
│  │TBE DSL   │               │AscendC   │                                   │
│  │real.py   │               │API调用   │                                   │
│  └────┬─────┘               └────┬─────┘                                   │
│       │                          │                                          │
│       ▼                          ▼                                          │
│  ┌──────────┐               ┌──────────┐                                   │
│  │  TVM     │               │ 编译器   │                                   │
│  │IR生成    │               │直接编译  │                                   │
│  └────┬─────┘               └────┬─────┘                                   │
│       │                          │                                          │
│       ▼                          ▼                                          │
│  ┌──────────┐               ┌──────────┐                                   │
│  │自动调度  │               │手动调度  │                                   │
│  │优化      │               │优化      │                                   │
│  └────┬─────┘               └────┬─────┘                                   │
│       │                          │                                          │
│       └──────────┬───────────────┘                                          │
│                  ▼                                                          │
│          ┌──────────────┐                                                     │
│          │ MOVEMASK     │                                                     │
│          │ 硬件指令     │                                                     │
│          └──────────────┘                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**详细对比表：**

| 对比维度 | TBE实现 | AscendC实现 | 说明 |
|---------|---------|------------|------|
| **编程语言** | Python DSL | C++ Template | TBE更简洁，AscendC更底层 |
| **调用入口** | `real.py:real()` | `real.cpp:real()` | 算子注册入口 |
| **计算定义** | `real_compute()` | `RealKernel::Process()` | 计算逻辑定义 |
| **API调用** | `tbe.real(input)` | `GatherMask(dst, src, mode=1)` | 高层API对比 |
| **中间表示** | TVM IR (`tir.real`) | 无中间层 | TBE通过TVM IR |
| **类型分发** | 运行时 `if/else` | 编译期 `constexpr if` | AscendC编译期优化 |
| **形状分类** | `classify()` 自动 | 无需分类 | TBE支持多shape自动处理 |
| **多核切分** | `auto_schedule()` 自动 | 手动计算 | TBE自动，AscendC手动 |
| **UB切分** | `auto_schedule()` 自动 | 手动计算 `ubMultiplier` | TBE自动，AscendC手动 |
| **双缓冲** | 框架自动管理 | 手动管理 `TQue` | TBE自动，AscendC手动 |
| **流水线** | 框架自动优化 | 手动优化 | TBE自动优化 |
| **指令映射** | `"elewise_single_real"` → `"vector_real"` | `GatherMask(mode=1)` → `MOVEMASK` | 最终相同 |
| **实数处理** | `elewise_single_VS_add(x, 0)` | `TQueBind` Identity | 实数类型优化方式不同 |
| **编译产物** | `.o` / `.json` / `.bin` | `.o` / `.bin` | 构建产物略有差异 |

**数据流对比：**

```
TBE数据流:                    AscendC数据流:
────────────                 ────────────
GM → UB → Compute → UB → GM   GM → UB → Compute → UB → GM
  ↑      ↑        ↑      ↑      ↑      ↑        ↑      ↑
  │      │        │      │      │      │        │      │
auto   auto    auto   auto   手动   手动    手动   手动
```

**关键差异原因：**

1. **设计理念不同**：
   - TBE：声明式编程，描述"做什么"，框架自动优化"怎么做"
   - AscendC：命令式编程，明确指定"怎么做"

2. **自动化程度不同**：
   - TBE：高度自动化（分核、UB切分、双缓冲、流水线）
   - AscendC：手动控制所有细节

3. **灵活性不同**：
   - TBE：快速开发，但优化空间受限
   - AscendC：开发复杂，但优化空间更大

4. **底层实现相同**：
   - 两者最终都调用 **MOVEMASK** CCE指令
   - 复数内存布局相同：`[r0,i0, r1,i1, ...]`
   - 提取原理相同：提取偶数索引元素

差异的原因，tbe实现分核逻辑是框架自动调优的，AscendC实现参考的库上Range算子，和Linspace算子。

### 3.3 支持硬件
| 支持的芯片版本 | 是否支持 |
|---------|---------|
| 香橙派OrangePi AIpro | 不支持 |
| Atlas 200I/500 A2推理产品 | 不支持 |
| Atlas 800I/T A2 | 支持 |


### 3.4 算子约束限制

- complex128 由AICPU原生支持，aclnn层面路由
- 不支持广播

## 四、 特性交叉分析

## 五、 可维可测分析

### 5.1 精度标准/性能标准

精度和TBE保持完全一致。

性能参考Real算子任务书，有如下要求：
- 算子整体性能需与原TBE实现算子持平
- 暂仅要求所有核参与计算场景下，性能不低于原TBE算子的 95%。
- 如小shape无法达标(10us以下场景相差3us),提供性能仿真图和分析结论证明AscendC实现和TBE完全一致或优于TBE实现

### 5.2 兼容性分析
- proto算子原型，legacy仓有一份旧的，要完全兼容旧的