# TurboQuantAttentionScore 算子设计方案

- 任务：9月社区任务-turbo_quant_attention_score算子开发
- 提交人：gai0718
- 邮箱：xinyisu0607@gmail.com
- 版本：2026-09-23，设计评审稿
- 实现分支：[feat/turbo-quant-attention-score](https://gitcode.com/gai0718/ops-transformer/tree/feat/turbo-quant-attention-score)
- 本文对应实现提交：`98ddd87f358881ffe1f927655595217abc390471`

本次提交申请设计评审。已有独立算子实现和阶段自测，但附件专项精度比较规则、KL/PPL 联测条件仍待确认，不代表已完成全部验收。

# 需求背景（required）

## 需求来源

9 月社区任务：TurboQuantAttentionScore 算子开发。目标仓库 `cann/ops-transformer`，
贡献目录 `experimental/attention/turbo_quant_attention_score`。
本文按组织方的[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织。

## 背景介绍

MHA/GQA 的长上下文 Key cache 占用大量显存。TurboQuant 将旋转空间中的主分量编码为少量 bit，
并用 QJL 符号向量修正残差。本算子直接从压缩 cache 生成分数，不在 GM 中完整展开 Key。
与仓内 TurboQuant 稀疏 attention 算子不同，本接口输出稠密 attention score，且不包含 V 聚合。

# 需求分析（required）

## 需求描述

实现 A2 / CANN 9.1.0 的 Ascend C 算子和 aclnn 两阶段接口。
输入输出、dtype 和布局在下文完整列出。以任务附件 `golden.py` 为编码语义基准。

## 需求拆解

1. 正确解析 2/3/4 bit 主编码以及 1 bit QJL 符号。
2. 使用 Paged KV cache，并支持 GQA、单序列 prefill 和批量 decode。
3. 通过 FP32 计算保留 golden 常量精度，不以 BF16 centroid 额外引入误差。
4. 覆盖五组指定 shape 与边界场景，记录真实 aclnn 精度、耗时和内存数据。

# 详细设计（required）

## 算子分析

### 数学公式

记 \(c_j\) 为查表得到的主分量，\(s_j\in\{-1,+1\}\) 为 QJL 符号：

$$
\mathrm{score}=\mathrm{norm}\sum_{j=0}^{127}q_jc_j+
\frac{\sqrt{\pi/2}}{128}\,\mathrm{gamma}\sum_{j=0}^{127}q^{\mathrm{QJL}}_js_j.
$$

query 的旋转及投影由上游完成。该输出不额外乘以 \(1/\sqrt{128}\)，不包括 causal mask、softmax 和 V 聚合。
主编码按低 bit 优先解包，3 bit 索引允许跨字节；QJL bit 0 对应 -1，bit 1 对应 +1。
centroid 使用附件 golden 给出的 FP32 常量；任务文字中的 FP16/BF16 解码描述与此存在精度选择差异，提请设计评审确认。
QJL 分支是实数 query 与符号向量的点积，不能直接替换为 popcount。

### 支持数据类型与形状

全部 Tensor 使用 ND 布局。记 T 为 query token 数，Hq/Hk 为 query/KV head 数，N 为物理块数，B 为请求数，P 为每个请求页表容量，L 为 `max(seq_lens)`。

| 参数 | 形状 | 数据类型 | 含义 |
| --- | --- | --- | --- |
| query_rotated | `[T,Hq,128]` | BF16 | 上游完成旋转后的 query |
| query_qjl | `[T,Hq,128]` | BF16 | 上游完成 QJL 投影后的 query |
| key_cache_idx | `[N,128,Hk,16*mse_bits]` | UINT8 | 主量化编码 |
| key_cache_qjl | `[N,128,Hk,16]` | UINT8 | QJL 符号编码 |
| key_cache_norm | `[N,128,Hk]` | BF16 / FP16 | 主分量范数 |
| key_cache_gamma | `[N,128,Hk]` | 与 norm 相同 | 残差尺度 |
| block_table | `[B,P]` | INT32 | 逻辑页到物理块的映射 |
| seq_lens | `[B]` | INT32 / INT64 | 每个请求的有效 KV 长度 |
| attn_scores（输出） | `[T,Hq,L]` | FP32 | 稠密 attention score |
| mse_bits（属性） | 标量 | INT | 2、3 或 4，默认 3 |

head_dim、qjl_dim、block_size 固定为 128；当前 centroid 和 golden 仅覆盖该维度。
Hq 必须为 Hk 的正整数倍。B=1 时全部 T 个 query 使用同一请求的 cache；B>1 时要求 T=B，每个请求一个 query。
有效长度范围为 `[0,128*P]`，有效物理块索引范围为 `[0,N)`。
短请求在输出长度 L 内的尾部填负无穷；全部长度为零时输出最后一维为零。
不按页表容量扩大输出，不支持一般广播。

## 算子实现

### host 侧设计

OpDef 注册 BF16/FP16 norm/gamma 与 INT32/INT64 seq_lens 的四种组合，自动生成 aclnn API。
形状推导声明 `seq_lens` 的 OPTIONAL 输入值依赖，数据在 host 时输出最后一维为真实最大长度；值未知时保留未知维度。
aclnn Tensor 路径的 optional 值依赖可能保留设备数据；该场景使用调用方预分配输出的最后一维，
不在 host 解引用设备指针。CPU 整数数组路径验证真实最大长度与输出形状一致。
tiling 检查 rank、形状关系、GQA 整除、batch/token 映射、属性和 dtype，再保存八个 UINT32 参数：tokens、qHeads、kvHeads、batch、blocks、tableWidth、maxLen、bits。
按设备 AIV 核数分派任务。tiling key 0/1 分别为 BF16/FP16 norm/gamma。
内核不使用 GM workspace；框架可能因连续化等需要返回额外 workspace，调用方必须使用实际返回值。

### aclnn 接口与生命周期

安装算子包后包含 `aclnn_turbo_quant_attention_score.h`，链接 `libcust_opapi.so`。

```cpp
aclnnStatus aclnnTurboQuantAttentionScoreTensorGetWorkspaceSize(
    const aclTensor *queryRotated, const aclTensor *queryQjl,
    const aclTensor *keyCacheIdx, const aclTensor *keyCacheQjl,
    const aclTensor *keyCacheNorm, const aclTensor *keyCacheGamma,
    const aclTensor *blockTable, const aclTensor *seqLens,
    int64_t mseBits, const aclTensor *attnScores,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    const aclTensor *queryRotated, const aclTensor *queryQjl,
    const aclTensor *keyCacheIdx, const aclTensor *keyCacheQjl,
    const aclTensor *keyCacheNorm, const aclTensor *keyCacheGamma,
    const aclTensor *blockTable, const aclIntArray *seqLens,
    int64_t mseBits, const aclTensor *attnScores,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnTurboQuantAttentionScore(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

Tensor 路径要求调用方预分配 FP32 输出，最后一维必须等于真实最大长度。设备上的 seq_lens 不得在 host 直接解引用。
CPU 整数数组路径检查输出最后一维与实际最大长度一致。两条路径共享执行接口。
第一阶段获得 workspace 大小和 executor，第二阶段在指定 stream 异步执行；每次调用重新获取 executor。
输入、输出和 workspace 在 stream 完成前必须保持有效。
无效形状、dtype 和属性由 host 拒绝；设备长度的合法性是调用方契约，非法长度及非法物理块由内核防护为负无穷，防护不代表支持非法输入。

### kernel 侧设计

每个任务处理某一 KV head 的 16 个连续 key 位置。单页为 128 个 token，因此 tile 不跨页。
prefill 每组最多八个 query，共享 tile 解码结果；同组所有 GQA head 同样复用解码结果。
核间按 task index 跨步分配，每块输出仅有一个写入者。

1. 按页表映射物理块，以带 stride 的 DataCopyPad 搬入编码、norm 和 gamma。
2. 在 UB 中将 byte 经 FP16 转成 FP32，再 Gather 两个相邻 byte。组合值最大为 65535，FP32 可精确表示。
3. 乘以精确的二进制倒数、向下取整和取模，得到精确量化索引；再 Gather centroid。
4. QJL 分支按同样方法解出符号。实数 query 参与点积，不使用错误的 popcount 等价替换。
5. query 转 FP32 后进行向量乘法和树形归约，分别乘 norm/gamma 并叠加修正项。
6. 写出 16 个或尾部不足 16 个分数。越过当前请求长度的元素写负无穷。

每核固定分配 160 KiB UB，包含索引、二进制缩放向量、解码块、query 缓存和归约临时空间。
GQA 比例不超过 4 时一次搬入最多八个 query token 的全部同组 head，向量化归约合并及范数缩放；
更大 GQA 比例走逐 head 搬运路径，避免越过 query 缓存容量。
所有向量缓冲区 32 字节对齐。跨流水使用显式 barrier；尾部搬运只访问有效行。
当前选择 Vector Core 保持 FP32 centroid 和计算精度；后续若引入 Cube 或 BF16 centroid 路径，需单独验证精度与性能，不能直接替换。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B） | √ |

开发目标为 CANN 9.1.0；本次阶段自测设备为 Ascend 910B3。

## 算子约束限制

不支持 MLA、任意 head_dim/qjl_dim、多序列 prefill、Value 解码、softmax 或 V 聚合。
batch 大于 1 时每个请求只能有一个 query token；缺少 query 边界的接口不能推导更多映射。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 验证方式 |
| --- | --- |
| 压缩域计算正确性 | 附件 golden 与独立整数解码参考、全输出比较 |
| Decode 延迟 | aclnn 实际执行，与任务书给定三组 baseline 对照；另列主机调用耗时 |
| Key cache 压缩 | 默认 3 bit 主编码 + 1 bit 符号 + 两个 16 bit 范数，共 68 B/key，原 BF16 Key 为 256 B，压缩比约 3.765 |
| 总显存 | 单独列出输出大小、cache 和 workspace，不能将 Key 压缩比宣称为进程整体压缩比 |
| KL / PPL | 需发布者提供编码器、模型/数据集及算子2联测条件，随机编码附件不能完成该验收 |

## 阶段自测与性能记录

测试日期：2026-09-23。环境：Ascend 910B3、CANN 9.1.0、torch 2.9.0、torch_npu 2.9.0.post6。
已执行附件五组场景、39 组边界场景和 11 组非法输入检查，并检查 CPU 长度数组、INT64 Tensor 接口结果一致。
有效输出与 CPU 参考、附件 golden 按仓库 FP32 混合误差标准比较通过；相对附件 golden 的最大绝对误差不超过 `2.384185791015625e-6`。

采用的阶段标准来自 opbase 提交 `0a83b915a6f6be49a03b907713d6f000b3dea1ea` 的 `docs/zh/ops_precision_standard/experimental_standard.md`：
FP32 atol 为 \(2^{-16}\)，rtol 为 \(2^{-10}\)，匹配比例至少 0.99，最大绝对误差不超过 0.01。
任务附件 `err_threshold=[0.001,0]` 的具体比较器语义仍待确认；若解释为所有元素均须满足无绝对容差的纯相对误差，部分接近零的输出不能通过，不能据阶段标准宣称专项验收已通过。

以下场景均为 B=1、Hq=32、Hk=8、D=128、mse_bits=3。

| 场景 | T | KV 长度 | NPU event 中位耗时（μs） | 任务给定 baseline（μs） |
| --- | --- | --- | --- | --- |
| gqa_decode_kv4k | 1 | 4096 | 407.860 | 1501.634 |
| gqa_decode_kv32k | 1 | 32768 | 2503.250 | 9278.507 |
| gqa_decode_kv128k | 1 | 131072 | 9655.190 | 35681.690 |
| gqa_prefill_kv4k | 512 | 4096 | 86947.800 | 75984.253 |
| gqa_prefill_kv32k | 512 | 32768 | 696017.792 | 未提供 |

耗时为预热后 10 次采样的中位数，包括设备端解码、点积和结果写出；主机规划、提交和同步耗时另行记录。
baseline 是任务附件的 Paged Full-QK 数据，并非同机重新测得的 FlashAttention 数据。
三组 decode 约为所给 baseline 的 0.27 倍；4k prefill 约为 1.144 倍，仍慢于所给 baseline；32k prefill 缺少对应 baseline，暂不作性能达标结论。

默认编码每个 Key 为 48 B 主编码、16 B QJL 和 4 B norm/gamma，共 68 B；原 BF16 Key 为 256 B，约压缩 3.765 倍。
该比例仅针对 Key cache。32k prefill 的 FP32 分数输出本身为 2 GiB，不能宣称整个进程显存也降低 3.765 倍。
当前自测 workspace 返回值为零；内存计数记录不等价于 sanitizer 的越界检查。

## 待评审确认与后续验证

1. 确认以附件 FP32 centroid 为计算基准是否符合任务中的解码 dtype 要求。
2. 确认 `err_threshold=[0.001,0]` 的比较器、近零值处理与验收容差。
3. 确认性能 baseline 的环境和比较口径；继续优化 prefill 的数据复用及输出开销。
4. KL < 0.01 与 PPL 相对变化 < 1% 尚未验证，需要编码器、量化前完整 K、算子2及模型/数据集的联测条件。随机编码 cache 的单算子测试无法替代该验证。
5. 若后续要求多序列 prefill、其他 head_dim 或 MLA，需要先补充接口与编码约定，再扩展实现和测试。

## 兼容性分析

新增独立算子，不修改其他算子行为。使用仓库默认 experimental 自动发现和 aclnn 生成机制。


## 实现与自测材料索引

以下为实现仓库中的材料，本次设计 PR 仅提交本文，不包含源码、测试日志或安装包。

- [算子说明及参数约束](https://gitcode.com/gai0718/ops-transformer/blob/98ddd87f358881ffe1f927655595217abc390471/experimental/attention/turbo_quant_attention_score/README.md)
- [aclnn 接口说明](https://gitcode.com/gai0718/ops-transformer/blob/98ddd87f358881ffe1f927655595217abc390471/experimental/attention/turbo_quant_attention_score/docs/aclnnTurboQuantAttentionScore.md)
- [实现与测试目录](https://gitcode.com/gai0718/ops-transformer/tree/feat/turbo-quant-attention-score/experimental/attention/turbo_quant_attention_score)
- [阶段交付材料](https://gitcode.com/gai0718/ops-transformer/tree/feat/turbo-quant-attention-score/task_submission)
