# MaxPool2dWithMask 算子设计文档

> 团队：jh051126 ｜ 更新日期：2026-09-23 ｜ 对应代码：`jh051126/ops-nn:maxpool2d-opt-a2`，提交 `2486a78b3`
>
> 当前版本已在 Atlas 800T A2 通过精度回归，尚未达到性能要求，也未完成 Atlas 300V Pro 适配。本文将已实现内容与后续方案分开说明。

# 1. 需求背景（required）

## 1.1 需求来源

依据社区任务《MaxPool2dWithMask 算子开发任务书》和 [aclnnMaxPool2dWithMask 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMaxPool2dWithMask.md)，在 `ops-nn/experimental/pooling` 中以 Ascend C 实现二维最大池化及索引输出。任务交付包括设计文档、可复现测试、精度与性能报告，以及验收后的代码 PR。

## 1.2 功能与现网基线

输入为 `[N,C,H,W]`，或按 `N=1` 处理的 `[C,H,W]`。窗口在 H、W 上滑动，输出最大值及其在输入通道平面内的展平索引。填充区域不参与最大值选择；相同最大值保留按 `kh` 再 `kw` 扫描时首次出现的位置。

现网 A2 非 1×1 路径复用三维池化，INT8 `indices` 容器头部实际承载 int32 argmax 字节流。专用二维内核减少维度适配，但当前大形状实测仍明显慢于现网，详见 §4。

# 2. 需求分析（required）

## 2.1 接口与范围

| 参数 | 类型与形状 | 约束及处理 |
| --- | --- | --- |
| `x` | FP16、FP32、BF16；3D/4D，ND | BF16 仅 A2；IR 以 `AutoContiguous` 处理非连续输入 |
| `kernelSize` | int64 列表 | 长度 1 或 2，值大于 0；单值同时用于 H/W |
| `stride` | int64 列表 | 长度 0、1 或 2；空列表默认等于 `kernelSize`，非空值大于 0 |
| `padding` | int64 列表 | 长度 1 或 2；每维满足 `0 ≤ p ≤ k/2` |
| `dilation` | int64 列表 | 长度 1 或 2；目前仅支持各维为 1 |
| `ceilMode` | bool | 默认 false；true 时修正末端完全落在 padding 的窗口 |
| `out` | 与 `x` 同类型、同维数 | 空间维度为 `Ho,Wo` |
| `indices` | INT8，与 `x` 同维数 | 最后两维为 `kH*kW, 32*(ceil(Ho*Wo/16)+1)`；不支持非连续输出 |

任务书要求 Atlas 800T A2 相对原 TBE 实现不低于 95% 的性能，并要求 Atlas 300V Pro 功能与精度。100 µs 以下的小形状若慢于 TBE 超过 30%，还需性能仿真图及分析；300V Pro 超过其基准 30% 的形状同样需要仿真图。输入 NaN、负无穷不在任务支持范围内。

## 2.2 关键验收点

1. 输出形状、ceilMode 修正、窗口边界及首次最大值索引正确。
2. `out` 数值与 `indices` 整段逐字节正确，包括未承载索引流的尾部清零。
3. 空 stride、单元素参数、3D、非连续输入及非法参数有明确处理。
4. A2 与 300V Pro 分别完成设备验证；大、小形状符合相应性能门槛。

# 3. 详细设计（required）

## 3.1 数学与存储语义

令 `(h0,w0)=(h*sH-pH,w*sW-pW)`，则 `out[...,h,w]` 是所有有效 `x[...,h0+kh,w0+kw]` 的最大值。按行优先扫描，`argmax=ih*W+iw`；只在严格大于当前值时更新，因而并列值保留首次位置。FP16/BF16 有限值无损升至 FP32；通用路径以 FP32 比较。正负零比较相等，保留首次值及索引。

单个空间维度的输出尺寸：

```text
q = floor((in+2*p-d*(k-1)-1)/s)+1    (ceilMode=false)
q = ceil ((in+2*p-d*(k-1)-1)/s)+1    (ceilMode=true)
```

`ceilMode=true` 且 `(q-1)*s >= in+p` 时，`q -= 1`。目前仅允许 `d=1`；非正输出尺寸报错。

设 `S=N*C`（3D 时 `N=1`）、`P=Ho*Wo`、`M=32*(ceil(P/16)+1)`。`indices` 外形为 `[N,C,kH*kW,M]` 或 `[C,kH*kW,M]`。已验证的 A2 路径按以下字节布局：

```text
indices[0 : 4*S*P]          = S*P 个小端 int32 argmax，按 N,C,Ho,Wo 平铺
indices[4*S*P : S*kH*kW*M] = 0
```

字节流可以跨过单个 `(n,c)` 容器分块，不能按每片单独清零。必须满足 `kH*kW*M >= 4*P`。1×1 且 `P` 较大时该条件不成立；现有精度集未覆盖该组合，现网 1×1 还可能采用另一种 mask 语义，因此不宣称其完整契约已实现。

## 3.2 Host 侧实现

IR 名称为 `MaxPool2dWithMask`，一个输入 `x`、两个输出 `out/indices`，属性依次为 `kernelSize/stride/padding/dilation/ceilMode`。当前仅添加 `ascend910b` AICore 配置。InferShape 校验 3D/4D、`C/H/W>0`、属性长度及数值，推导两个输出形状。Tiling 使用平台 AIV 核数和 UB 大小，不申请额外 workspace。

### 3.2.1 多核与 UB 切分

一个 `(n,c)` 通道平面是一片。设 `S=N*C`、AIV 核数为 `K`，`slicesPerCore=ceil(S/K)`，`usedCoreNum=ceil(S/slicesPerCore)`。每核处理连续的片，片内按输出行分 tile；片间无依赖。`S=0` 时设单核安全 tiling，kernel 直接返回。

输入、输出、索引行分别按 32 B 对齐。设对齐后的每行字节数为 `Bi,Bo,Bidx`，其中 `Bidx=align32(4*Wo)`。双缓冲队列下：

```text
first = (dH*(kH-1)+1)*Bi + Bo + Bidx
more  = sH*Bi + Bo + Bidx
R     = min(Ho, 1+floor((UB-40960-2*first)/(2*more)))
```

预算不足时代码以 `R=1` 兜底。实际分配为 `inputQueue`、`yOutputQueue`、`indicesQueue` 各 2 份，另有 `zeroQueue` 的 `2*16 KiB`；40960 B 预留包括后者及余量。极大输入宽度下，`R=1` 仍需检查 UB 能否容纳，现有测试不构成通用证明。TilingKey 按 FP16、FP32、BF16 实例化内核。

## 3.3 Kernel 侧实现

1. `CopyInRows`：搬入本组输出行覆盖的输入行；越界行填充类型最小有限值，计算时仍检查边界，填充值不参与比较。
2. `ComputeTile`：逐输出位置扫描有效窗口。通用路径预计算有效 `kh/kw` 边界与行基址。完整内部 2×2、3×3、7×7 方形窗口编译期展开；FP16/BF16 内部路径直接比较 16 位有序键，保持有限值顺序，并把 `+0/-0` 归为同一键。FP32 和边界窗口按 FP32 比较。结果写入 UB 中的值与 int32 索引。
3. `CopyOut`：值按输出元素偏移写回；索引按 `4*(sliceIdx*P+tileOffset)` 字节偏移写入容器头部。
4. `ZeroIndicesTail`：各核分摊 `[4*S*P,S*kH*kW*M)`，经 int16 视图与专用队列清零；与头部写入区间不重叠。

`TPipe` 在内核入口创建一次。当前最大值计算仍是逐位置标量扫描；运行在 AIV 核上不等于已完成向量化。

## 3.4 后续性能设计

第 2 轮在大形状的 `msprof` 数据中，AIV scalar 占 `27103.496/27154.580 µs`（约 99.9%）。继续局部减少标量指令难以达到 95% 性能门槛。下一版拟先验证适合向量计算的窗口布局：参考现网 argmax 路径，将 NCHW 分块转换至带 C0 的布局，或将窗口展开为连续数据。必须先核实数据搬运与重排 API、UB 容量及索引回写开销，再选实现，不把该方案列作已完成能力。

验证顺序：单片单 dtype 原型与 golden 逐字节对拍；扩展边界、ceilMode、三种 dtype 与全部 168 条 A2 用例；最后重测六个固定性能 case。300V Pro 需要独立注册、编译与设备验证，不能由 A2 结果推断。

## 3.5 现有限制

| 硬件 | 当前状态 |
| --- | --- |
| Atlas 800T A2（910B3） | 已编译并完成现有 A2 精度、性能用例；性能未达标 |
| Atlas 300V Pro | 未注册、未编译、未进行设备验证 |

- 自动生成的 aclnn 适配层对空 `stride` 列表的接受情况与 IR 默认规则不一致；InferShape/Tiling 已实现默认值，但 aclnn 端到端接口仍需适配验证。
- 1×1 窗口的 `indices` 容量及现网分支语义尚未闭环。
- 仅在 Atlas 800T A2（910B3、CANN 9.1.0）编译和运行，未验证 Atlas 300V Pro。
- 暂不支持 NaN、负无穷；不接受 NHWC，`indices` 不支持非连续输出。

# 4. 可维可测分析

## 4.1 精度验证

当前提交已在 A2 编译安装：官方 153 条自测用例与自补 15 条边界用例共 **168/168 通过**。比对包括 `out` 逐位、`indices` 整段逐字节及尾部全零。另做 FP16/BF16 正负零并列值设备用例，核对首次索引和输出位型；16 位有序键在主机侧穷举了全部有限 FP16/BF16 位型的排序关系。上述结果不替代 1×1、极端宽度及 300V Pro 的泛化验证。

## 4.2 性能验证与结论

Atlas 800T A2 / 910B3、CANN 9.1.0；六个固定 FP16 case，ACL event 各 20 次均值。表中“现网”为同机 `aclnnMaxPool2dWithMask` 实测参照；任务书规定的原 TBE 验收口径仍须单独确认。比值根据表中四舍五入后的毫秒数计算，均为约值。

| Case | V1 (ms) | 当前版 (ms) | 现网 (ms) | 当前版/现网 |
| --- | ---: | ---: | ---: | ---: |
| `(2,64,224,224) k7s2p3` | 94.967 | 22.168 | 0.394 | 56.22 |
| `(1,64,112,112) k3s2p1` | 2.525 | 0.768 | 0.044 | 17.45 |
| `(1,32,28,28) k3s1p1` | 0.324 | 0.113 | 0.033 | 3.42 |
| `(1,64,56,56) k2s2p0` | 0.320 | 0.142 | 0.032 | 4.44 |
| `(1,1,4,4) k3s2p1` | 0.017 | 0.016 | 0.029 | 0.55 |
| `(8,512,7,7) k3s1p1` | 1.909 | 0.728 | 0.057 | 12.77 |

当前版是三轮局部优化后的正确性基线。大形状相对 V1 快约 4.28 倍，但与现网仍有明显差距，尚未满足任务性能要求。小形状 1–2 µs 差异接近计时波动，不作为稳定收益结论。

## 4.3 兼容性与交付状态

代码位于 `ops-nn/experimental/pooling/max_pool2d_with_mask`，新增目录，不替换现网实现。A2 精度已有设备证据；性能、空 stride 的 aclnn 端到端契约、1×1 容器以及 300V Pro 适配仍是验收前待闭环项。设计文档 PR 可供方案评审，不代表算子代码已达到最终验收条件。
