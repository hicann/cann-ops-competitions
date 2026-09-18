# MaxPool2dWithMask 算子设计文档

贡献者：Huilan123；任务：09-46-MaxPool2dWithMask；日期：2026-09-17。

状态：设计评审稿。A2 功能原型已验证，310P 兼容口径待确认，性能尚未达标。

任务讨论：https://gitcode.com/cann/ops-nn/discussions/8

## 1. 需求背景

按任务书使用 Ascend C 实现二维最大池化与最大值位置输出，目标平台为 Atlas 800T A2、Atlas 300V Pro，CANN 9.0.0/9.1.0。最终贡献至 ops-nn experimental/pooling，包含 op_host、op_kernel、接口适配、测试及 README。当前独立 kernel launch 工程仅为算法验证原型。

已阅读 ops-nn 9.0.0 分支 pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool2d_with_indices.cpp：A2 非 1x1 窗口走 MaxPool3DWithArgmaxV2Ncdhw，其他分支涉及 MaxPoolWithArgmaxV1 和格式转换，不能直接推断所有平台的 indices 内存语义一致。

CANN 9.1.0、Ascend910B3 实测：官方接口及自研原型均通过配套 153 条 golden 对比。Ascend310P3 同版本最小例 F001_s01_k3s2p1_fp32：NCHW 调用成功，out 一致，但 indices 前 16 字节按 int32 解释为 [0,65024,-8388609,-8388609]，golden 为 [0,7,13,13]。float16 F002 返回 161002，提示仅支持 DT_FLOAT；ND float32 返回 561103，提示不支持 ND -> NC1HWC0，改为 NCHW 后可运行。当前设计先遵循任务 golden，310P 验收口径请维护者确认。

## 2. 需求分析

### 2.1 接口和约束

- 输入 NCHW/ND，float32/float16，A2 另支持 bfloat16；配套 3D 输入按 N=1 归一化。
- kernelSize 长度 1/2、正数；stride 为空默认 kernelSize，否则长度 1/2、正数；padding 长度 1/2、0<=p<=floor(k/2)；dilation 仅为 1。
- out 与输入 dtype 相同；非连续输出通过临时连续 tensor + ViewCopy 回写。
- indices 是 INT8 容器，shape=[N,C,kH×kW,(ceil(OH×OW/16)+1)×32]；有效内容为整个容器起始位置连续排列的 int32 argmax，剩余字节清零。不是每个 NC 子容器分别填充。
- argmax 只包含通道内 H×W 偏移；行优先遍历，严格大于才更新，重复最大值取第一个有效输入位置。
- NaN/-Inf 输入不在任务范围，+Inf 正常比较。

### 2.2 待确认契约

1. 310P 是否统一使用 int32 argmax 并要求 float16？旧官方接口与任务约定存在上述差异。
2. 1x1 容量冲突：[1,1,8,8]、kernel=1、stride=1、padding=0，公式仅分配 160 字节，64 个 int32 需要 256 字节。请确认修订 shape、限制范围或其他约定。确认前 Host 应拒绝容量不足，绝不越界写入。
3. FLOAT 转 FLOAT16 的描述是否仅对应旧产品？A2 给定用例符合 float32 golden，当前保持 float32 比较，并补充近似相等值测试。
4. 原型将逻辑输入 view 转为连续，不等于已经验证正式非连续输出接口。

## 3. 详细设计

### 3.1 公式

窗口起点 ih=oh×sH-pH、iw=ow×sW-pW，仅遍历满足 0<=ih+kh<H、0<=iw+kw<W 的点；选中点索引为 (ih+kh)×W+iw+kw，不含 NC 偏移。kh/kw 递增扫描，严格大于时同步更新 value/index。

单维 L：floor 输出为 floor((L+2p-k)/s)+1，ceil 输出为 ceil((L+2p-k)/s)+1；ceil 时若 (out-1)×s>=L+p，则减 1。乘积和地址使用 int64，检查溢出及 int32 索引上界。

### 3.2 Host、分核与 Tiling

检查维度、dtype、属性、输出 shape、mask 容量及平台能力，归一化 3D/attrs。非连续输入走 Contiguous，非连续 out 通过 ViewCopy 回写，indices 不接受非连续；非法参数在 launch 前返回错误。

读取真实 AIV 数量、UB 容量及搬运能力，正式实现不硬编码 40 核（40 核仅用于当前 910B3 实验）。核数取可用核数与非空 tile 数的较小值。TilingData 包含 N/C/H/W/OH/OW、k/s/p、有效输出数、每核 tile 范围、input span、tile 长度、mask 字节数和 dtype；TilingKey 区分展平向量、逐行向量、通用回退及平台。

沿 N×C×OH×OW 展平输出切分，tile 长度为 16 的倍数，保证 fp16/bf16、fp32、int32 按 32 字节边界分核；仅末核处理全局尾块。每个输出只有一个写入者，无原子操作。

A2 当前原型 tile=256、最大输入 span=16384 元素，UB 工作缓冲约 span×4+12×tile×4+128 字节，另有输入 span×sizeof(T)、输出 tile×sizeof(T)、索引 tile×4，float32 合计约 145536 字节。正式 Host 严格核对所有缓冲及保留空间，不满足时缩小 tile 或回退，暂不开双缓冲。

小 H/W、大 NC 跨通道合并输出，摊薄队列与地址生成开销。下一步复用偏移表、分离无 padding 区域，减少标量 SetValue 和重复比较；大 span 保留逐行向量路径。任何优化后重跑完整 golden。

### 3.3 Kernel 流程

1. CopyIn：计算 tile 覆盖的合法连续输入区间并搬入 UB；半精度/bfloat16 转 float32 比较。有效掩码避免越界 Gather。
2. Compute：best=-Inf、argmax=0；每个 kh/kw 生成地址及有效掩码，Gather 后以严格大于且有效的条件 Select 更新 value/index。索引浮点快路径仅允许 H×W<2^24，其他范围采用整数索引回退；最终 int32 必须可表达。
3. CopyOut：value 转回原 dtype，索引按 int32 连续写入 INT8 容器。全局尾块使用平台支持的安全尾搬运；310P 不直接复用 A2 DataCopyPad，单独适配搬运及必要的尾部暂存。
4. mask 尾零：正式 kernel 按不重叠区间写零，或采用明确初始化步骤并计入端到端开销。原型目前由桥接层 aclrtMemset 完成，尚不是 kernel 自身功能。

TPipe 管理队列生命周期；标量偏移生成后 S_V 同步；向量依赖插入必要屏障，profiling 确认后精简。通用回退用于功能覆盖，不能作为性能达标依据。

### 3.4 工程集成

计划在 experimental/pooling/max_pool2d_with_mask 组织 op_host、op_kernel、tests、README，参考目标分支已有实验算子接入构建/注册。保留 GetWorkspaceSize/执行两阶段接口，遵循 opbase 的 workspace、执行器、stream 生命周期。旧平台格式/dtype 转换不得隐式改变 argmax 语义，待评审确认适配边界。

## 4. 可维可测分析

### 4.1 当前状态

2026-09-17、HiDevLab、CANN 9.1.0、Ascend910B3：

- 自研展平原型通过任务 153 条（float32 49、float16 54、bfloat16 50），out 和完整 mask 精确一致；官方接口同样通过。
- 额外 36 组边界通过：ceil 修正、空 stride、矩形窗口、重复最大值、负值、单元素、3D、+Inf、窄宽度、逻辑输入 view、近似相等值、跨 tile 尾部。
- view 在 harness 中转连续，正式非连续输出、ops-nn 集成和 310P 自研 kernel 尚未完成验证。

### 4.2 精度与泛化计划

逐元素比较 out、全部有效 int32 argmax 和尾部零字节；使用独立 shape 公式核对 golden。覆盖属性长度、空 stride、矩形、大 padding、ceil、重复/+Inf、近似相等、3D、尾对齐、非连续输入输出、超 UB、非法 dtype/shape/容量、整数溢出。1x1 契约未明确前记录为阻塞项，不冒险运行。索引错误不能用浮点容差掩盖；最终同时满足 opbase 生态算子精度标准。

### 4.3 性能与复现

A2 目标不低于原 TBE 的 95%；小 shape 按任务书 100us/30% 条件补充仿真。310P 按 A2(910B3)基准 5 倍及超出 30% 条件评估。

当前数据是主机 launch+stream synchronize 时延，未包含完整 GetWorkspaceSize，不能替代设备 profiling/TBE 验收。原型 F163 约 446us、F218 约 619us，仍需优化，未宣称达标。正式报告固定环境/输入、预热后重复采样，保留设备 profiler 原始数据、TBE 基线、统计方法和必要仿真图。

交付固定随机种子、构建/运行脚本、环境版本、逐例 JSON、最小失败输入、README；精度和性能分开报告并记录源码提交号。补齐 ops-nn UT/ST、泛化及平台编译验证后申请验收。

## 5. 兼容性与风险

保持既有接口约定，不修改其他算子。A2/310P 分别编译验证，310P 不实例化 bfloat16 或 A2 专有搬运路径。优先确认第 2.2 节契约冲突，再固定最终接口和基线。本次仅申请设计评审，不申请验收。
