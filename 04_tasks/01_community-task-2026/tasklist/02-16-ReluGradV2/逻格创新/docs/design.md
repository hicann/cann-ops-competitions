# ReluGradV2 算子设计文档

# 1需求背景（required）
## 1.1需求来源
通过社区任务完成开源仓Relu_Grad_V2算子贡献的需求。
## 1.2背景介绍
### 1.2.1 Relu_Grad_V2算子实现优化
基于Relu_Grad_V2算子历史TBE版本使用Ascend C编程语言进行优化。

Relu_Grad_V2算子（TBE）实现路径和相关API路径  
kernel实现：
/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/
算子原型：
/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/
算子信息库：
/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b

### 1.2.2 Relu_Grad_V2算子现状分析
基于 ReluGradV2 算子 TBE 版本的功能分析，当前支持的能力如下：  输入 gradients 支持 float16、float32、int32、int8、uint8、bfloat16 六种数据格式，输入 mask 支持 uint1、uint8 两种数据格式。  
内部计算精度根据平台指令支持能力和实现模式自适应选择 float16 或 float32。   
在数值稳定处理方面，因核心为条件选择 / 逐元素乘操作，无指数 / 对数类数值溢出风险；bfloat16 计算精度下通过 round 操作保证精度，int8/uint8 类型下先转换为 float16 完成计算后转回原类型，避免精度损失。  
计算表达式实现的核心数学表达式为 y = gradients * mask（逐元素，mask=1 保留 gradients，mask=0 置 0），TBE 实现通过组合基本算子完成：首先根据数据类型 / 平台能力对 gradients 做类型转换，然后通过 Select (mask, gradients, 0) 完成条件选择（或生成 mask 张量后通过 Mul 与 gradients 相乘），最后将计算结果转换为 gradients 原始类型输出。    
平台自适应特性方面，自动检测硬件对 tik.vcopy、Select 指令的支持能力（区分 Milan/Tuscany 芯片），优先使用 HIGH_PERFORMANCE 模式（非 Milan 芯片 float32 转 float16 计算提升性能），降级到 HIGH_PRECISION 模式保证计算精度，全芯片兼容 Select/Mul 基础指令。  
形状处理能力支持动态形状输入，输入 gradients 和 mask 形状需保持一致，无 keep_dims 参数，输出形状与输入 gradients 完全一致，自动适配 NC1HWC0/ND 等格式的维度处理。    
类型安全保证通过类型转换机制确保数据类型一致性：输入 gradients 转换为适配指令的计算精度类型，mask 固定为 uint1/uint8 无需转换，中间结果进行类型一致性检查，计算结果转换为 gradients 原始类型输出，同时通过参数校验保证输入类型在支持列表内。

ReluGradV2 算子TBE版本的整体流程图如下图所示：
![flow1.png](https://raw.gitcode.com/user-images/assets/9516645/a960bcb5-b8f2-428c-903c-c96bc933651e/flow1.png 'flow1.png')

# 2需求分析
## 2.1 外部组件依赖
不涉及外部组件依赖。
## 2.2 内部适配模块
适配Aclnn接口。
### 2.3.1 算子原型
#### 1) 原型设计
| 名称       | 类别       | dtype                  | format |        shape                     | 介绍                                                                 |
|:---------:|:---------:|:----------------------:|:------:|:------------------------:|:--------------------------------------------------------------------:|
|gradients | 必选输入 | float16/float32/int32/int8/uint8/bfloat16 | NC1HWC0 | 任意形状（支持动态形状）|输入张量，为 ReLU 反向传播的上游梯度，是反向梯度计算的原始数据源   |
| mask   | 必选输入   | uint1/uint8  | ND     | 与 gradients 形状一致（支持动态形状）| ReLU 正向计算生成的掩码张量，标记输入元素正向激活时保留（1）/ 置 0（0）的位置 |
|backprops | 必选输出   | float16/float32/int32/int8/uint8/bfloat16 | NC1HWC0 | 与 gradients 形状一致  | ReLU 反向传播的最终梯度结果，由 gradients 和 mask 逐元素条件选择计算得到 |

#### 2)相关约束
无。

# 3 需求详细设计
## 3.1 使能方式
Aclnn直调。
## 3.2 需求总体设计
### 3.2.1  host侧设计：
#### tiling策略：   
ReluGradV2 为逐元素（Elemwise）算子，无归约操作，核心是基于 mask 对 gradients 做逐元素条件选择计算。在 host 侧获取输入 gradients 的元素数量、vector核数量和UB大小，根据需要使用的张量数量，计算出切片大小。
#### 1)分核策略  
优先使用满核的原则，结合 32B 内存对齐规则（BLOCK_SIZE=32），按高维度优先级拆分数据：
维度拆分优先级：优先拆分 N 维度，N 维度无法均分则拆分 H 维度，其次为 W、C1 维度；C0 维度（昇腾硬件最小计算粒度）不拆分，保留在单个核心内处理。  

#### 2)数据分块和内存优化策略
充分使用 UB 空间的原则，兼顾 Double Buffer 机制（BUFFER_NUM=2）、不同精度适配及昇腾 C0 维度对齐规则：需考虑不同硬件的 UB 大小差异、Double Buffer 机制、多数据类型内存占用，综合优化单核内切分大小。  

#### 3)tilingkey规划策略
由于 ReluGradV2 算子为标准 Elemwise 算子，计算模式固定，结合 AscendC 标准 TilingKey 生成规范，采用极简的标准化策略：tilingKey = GET_TPL_TILING_KEY (0) // 基于 Elemwise 模板的标准化 TilingKey，替代自定义拼接逻辑同时需将 BlockDim 设置为实际使用的核心数（coreNum），保证核数分配及分块规则正确。  

数据检测：    
gradients 仅支持 float16/float32/int32/int8/uint8/bfloat16 类型，mask 仅支持 uint1/uint8 类型；  
若 gradients/mask 为其他类型（如 int64/double/uint16 等），或 gradients 与 mask 形状不一致，直接返回不支持的错误日志并终止 Tiling 流程。  

### 3.2.2 kernel侧设计：
#### 3.2.2.1 Kernel侧实现描述
Kernel 侧核心分为Init初始化阶段和Process计算阶段，其中Process阶段严格拆分为数据搬入（CopyIn）、核心计算（Compute）、数据搬出（CopyOut） 三个子阶段，整体逻辑与 Elemwise 类标准算子（如 Mul/Select）对齐，且完全对齐 TBE 版数据类型约束（gradients 支持 float16/float32/int32/int8/uint8/bfloat16，mask 支持 uint1/uint8），具体实现如下：    

##### 1. Init初始化阶段
- 核心目标：完成输入输出地址绑定、TilingData 参数读取、参数合法性校验；  
- 关键操作：
  1. 校验输入 gradients、mask，输出 backprops，TilingData 指针非空，若为空则返回 GRAPH_FAILED 并输出错误日志；  
  2. 校验数据类型合法性（对齐 TBE 版约束）：  
     - gradients 仅支持 float16/float32/int32/int8/uint8/bfloat16，拒绝 int64/double 等类型；  
     - mask 仅支持 uint1/uint8，拒绝其他整型 / 浮点型；  
     - 校验 gradients 与 mask 的 shape 维度、元素数完全一致，不一致则返回错误；  
  3. 从 TilingData 中读取核心参数（与 Host 侧 Tiling 结果完全对齐）：  
     - 基础参数：inputNum（gradients 总元素数）、tileDataNum（单 tile 处理元素数）、dataType（gradients 数据类型）、maskType（mask 数据类型）；  
     - 适配参数：implMode（实现模式：HIGH_PERFORMANCE/HIGH_PRECISION）、coreType（芯片类型：Milan/Tuscany，通过 tik.vcopy 指令支持性判定）、c0Size（对应数据类型的 C0 对齐粒度）；
  4. 绑定 gradients、mask 的 Global Memory（GM）地址，绑定 backprops 的 GM 写入地址，为后续数据搬运做准备；    
  5. 初始化 UB 缓冲区（按 Double Buffer 机制分配）：为 gradients、mask、backprops 分别分配 2 组 UB 缓冲区（BUFFER_NUM=2），保证数据搬运与计算流水线并行。  

##### 2. Process计算阶段
- 核心目标：按 tileDataNum 分块循环处理所有元素，每个 tile 处理 tileDataNum 个 gradients 和 mask 元素（逐元素条件选择计算），整体循环逻辑为： 
  `for (int64_t tileIdx = 0; tileIdx < totalTileNum_; tileIdx++) { CopyIn → Compute → CopyOut }`  （注：totalTileNum_ = inputNum_ /tileDataNum_，若无法整除则最后一个 tile 处理剩余元素）  
- 分阶段详细实现：  

###### （1）CopyIn数据搬入阶段
- 核心目标：从 GM 读取指定 tile 范围的 gradients 和 mask 数据到 Unified Buffer（UB），按 C0 对齐粒度搬运，兼顾 Double Buffer 流水线；  
- 关键操作：
  1. 计算当前 tile 的 GM 起始偏移：`tileStartIdx = tileIdx * tileDataNum_`，结束偏移`tileEndIdx = min((tileIdx + 1) * tileDataNum_, inputNum_)`；
  2. 基于 Double Buffer 机制选择当前可用 UB 缓冲区（BufferA/BufferB），避免数据搬运与计算冲突；  
  3. 从 GM 读取数据到 UB：   
     读取`tileDataNum_个 gradients `元素到 UB 的 `gradients `缓冲区，数据类型匹配`dataType`；  
     读取`tileDataNum_`个 `mask `元素到 UB 的 mask 缓冲区，数据类型匹配`maskType`；  

###### （2）Compute核心计算阶段
- 核心目标：基于 TBE 版分支逻辑完成逐元素条件选择计算`（backprops = mask?gradients:0）`，适配不同芯片 / 数据类型，保证性能与精度；  
- 关键操作（分分支适配，对齐 TBE 版计算逻辑）：
  1. 分支 1：float32 + HIGH_PERFORMANCE 模式 + Tuscany 芯片（非 Milan，无 tik.vcopy 支持）：  
     - 步骤 1：将 UB 中 float32 的 gradients 转换为 float16（AscendC::Cast<float32, float16>）；  
	 - 步骤 2：生成 float16 的全 1 / 全 0 张量（AscendC::Broadcast），通过AscendC::Select基于 mask 选择 1/0，生成 mask 张量；  
	 - 步骤 3：将 mask 张量转换回 float32（AscendC::Cast<float16, float32>）；  
	 - 步骤 4：逐元素相乘（AscendC::Mul）：backprops_ub = gradients_ub * mask_ub；  
  2. 分支 2：int32 + Milan 芯片（有 tik.vcopy 支持）：
	 - 步骤 1：将 UB 中 int32 的 gradients 转换为 float16（AscendC::Cast<int32, float16>）；  
	 - 步骤 2：生成 float16 的全 1 / 全 0 张量，通过AscendC::Select基于 mask 选择 1/0，生成 mask 张量；  
	 - 步骤 3：将 mask 张量转换回 int32（AscendC::Cast<float16, int32>）；  
	 - 步骤 4：逐元素相乘（AscendC::Mul）：backprops_ub = gradients_ub * mask_ub；  
   3. 分支 3：bfloat16 类型：  
	  - 步骤 1：将 UB 中 bfloat16 的 gradients 转换为 float32（AscendC::Cast<bfloat16, float32>）；  
	  - 步骤 2：通过AscendC::Select基于 mask 选择 gradients_ub（float32）或 0，生成 backprops_ub；  
	  - 步骤 3：将结果四舍五入后转换回 bfloat16（AscendC::Round + AscendC::Cast<float32, bfloat16>）；  
    4. 分支 4：int8/uint8 类型 或 Select 指令不支持原生类型：  
	   - 步骤 1：将 UB 中 int8/uint8 的 gradients 转换为 float16（AscendC::Cast<int8/uint8, float16>）；  
	   - 步骤 2：通过AscendC::Select基于 mask 选择 gradients_ub（float16）或 0，生成 backprops_ub；  
	   - 步骤 3：将结果转换回原类型（AscendC::Cast<float16, int8/uint8>）；  
    5. 分支 5：通用分支（float16 / 其他兼容场景）：  
	   - 步骤 1：Milan 芯片：生成与 gradients 同 shape 的 float16 全 0 张量，通过AscendC::Select(mask, gradients_ub, zeros_ub)计算；  
	   - 步骤 2：Tuscany 芯片：直接通过AscendC::Select(mask, gradients_ub, 0)计算；  
 

###### （3）CopyOut数据搬出阶段
- 核心目标：将 UB 中计算完成的 tileDataNum 个 backprops 元素写入 GM 的输出地址；  
- 关键操作：
  1. 调用DataCopy把backprops写回GM；


##### 3. 核函数调度与模板化适配
- 基于schMode（调度模式）实现编译期分支选择，与 Host 侧 TilingKey、芯片类型联动：  
  - schMode=0：适配 float16/float32 类型 + Milan 芯片，实例化ReluGradV2<Float16/Float32, Milan>类；
  - schMode=1：适配 float16/float32 类型 + Tuscany 芯片，实例化ReluGradV2<Float16/Float32, Tuscany>类；
  - schMode=2：适配 int32/int8/uint8 类型，实例化ReluGradV2<Int32/Int8/Uint8, CoreType>类；
  - schMode=3：适配 bfloat16 类型，实例化ReluGradV2<BFloat16, CoreType>类；
- 核函数启动时，直接复用 Host 侧 Tiling 设置的BlockDim（核心数），保证核间数据分配与 Host 侧分核策略一致；  
- 针对不同芯片的指令支持性，编译期选择指令实现：Milan 芯片使用tik.vcopy相关指令优化数据搬运，Tuscany 芯片使用基础指令兜底。  

#### 3.2.2.2 AscendC实现流程图  
![flow2.png](https://raw.gitcode.com/user-images/assets/9516645/48341337-4649-4cdb-a481-a8d282677daa/flow2.png 'flow2.png')
#### 3.2.2.3 AscendC实现流程图与TBE流程图存在差异点和原因

因为 TBE 侧 ReluGradV2 的核数分配由框架隐式处理，无显式的 32B 内存对齐分核、大核 / 小核数据分配逻辑；因此，在 AscendC 的 Host 侧 Tiling 流程中，新增了基于 32B 内存对齐的满核分核策略（计算 totalInputBlockNum、bigCoreDataNum/smallCoreDataNum），并补充核数边界约束（单块数据≥输入总量则核数 = 1），从而手动实现核间数据均分与余数处理，保证硬件满核利用。

因为 TBE 侧 ReluGradV2 无编译期分支选择逻辑，仅通过运行时判断 impl_mode / 芯片类型执行不同计算路径；因此，在 AscendC 的 Kernel 侧流程图中，新增了基于 schMode 的编译期分支选择（实例化不同模板类适配数据类型 / 芯片），从而提前剔除冗余分支，提升算子运行时性能，这是 TBE 解释型运行时判断无法实现的编译期优化。

原TBE ReluGradV2的mask输入为uint1类型，在AscendC中作为uint8类型来处理，可能需要算子调用时把mask打包为uint8类型。在AscendC的Select是从每个uint8数值的低位选取元素，低位bit0对应第一个元素，高位bit7对应最后一个元素，可能导致位序不一致的问题。可能需要增加一个reverse_bit函数做mask的位序反转，从而增加额外的计算。

## 3.3支持硬件
| 支持的芯片版本 | 涉及勾选 |
|:----:|:----:|
香橙派OrangePi AIpro   
Atlas 200I/500 A2推理产品   |
Atlas 800I/T A2 |√
## 3.4算子约束限制
不支持广播。  
# 4特性交叉分析
无
# 5可维可测分析
## 5.1精度标准/性能标准
| 验收标准 | 描述(不涉及说明原因) | 标准来源 |  
|:----:|:----:|:----:|
精度标准|不低于TBE版本95%|算子任务书   
性能标准|整体不低于TBE版本|算子任务书   
## 5.2兼容性分析
新算子，不涉及兼容性分析