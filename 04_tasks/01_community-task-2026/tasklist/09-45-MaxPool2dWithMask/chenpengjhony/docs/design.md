# MaxPool2dWithMask 算子详细设计文档

## 1. 文档说明

| 项目 | 内容 |
|------|------|
| 算子名称 | MaxPool2dWithMask |
| 开发语言 | Ascend C |
| 适配硬件 | Atlas 800T A2（910B3，性能验收基准）、Atlas 300V Pro（310P3/dav-m200）；本自测环境芯片为 910B4-1（dav-c220，频率略低于 910B3，自测性能数据偏保守，报告需注明） |
| CANN 版本 | CANN 9.0.0 / 9.1.0 |
| 目标仓库 | ops-nn `experimental/pooling/max_pool2d_with_mask` |

## 2. 需求分析

### 2.1 功能描述

对 4D（NCHW）/3D（CHW，按 \(N=1\) 处理）输入做 2D 最大池化，输出池化结果 `out` 与每个输出点对应的最大值在**输入平面内的展平索引**（argmax，`ih*W+iw`，每通道独立）：

$$out(n, c, h, w) = \max_{\substack{0 \le m < k_H \\ 0 \le n' < k_W}} x\big(n,\ c,\ s_H \cdot h + m - p_H,\ s_W \cdot w + n' - p_W\big)$$

- padding 位置语义为"负无穷"；dilation 仅支持 1。
- **tie 语义**：行优先（kh 外循环、kw 内循环）扫描，严格大于才更新，即相等时保留先行、先列的索引（与 golden 函数 `vfull > cur` 严格大于一致）。
- 若某输出窗口内全部为 pad（无有效输入），则 `out = -inf`、`argmax = 0`。

### 2.2 输出 shape 推导

\(d=1\) 时（floor 模式）：

$$H_{out} = \left\lfloor \frac{H + 2p_H - k_H}{s_H} \right\rfloor + 1$$

ceil 模式将 floor 换为 ceil，并做末尾窗口修正（golden 中 `_pool_out_dim`）：

```text
out = ceil((in + 2p - k) / s) + 1
if (out - 1) * s >= in + p:  out -= 1     # 最后一个窗口起点不得越过 in+p
```

`indices` shape（与输入值无关，仅依赖输出尺寸）：

$$[N,\ C,\ k_H \cdot k_W,\ maskW], \qquad maskW = \left(\left\lceil \frac{H_{out} \cdot W_{out}}{16} \right\rceil + 1\right) \times 32$$

### 2.3 关键语义分析：indices 的物理布局（本算子最大设计风险点）

`indices` 逻辑 shape 为 `[N,C,kH*kW,maskW]`（int8），但 golden 函数的实际写入方式为：

```python
container = np.zeros((N, C, kH*kW, maskW), np.int8)
argmax_bytes = idxstar.reshape(NC, hw).astype('<i4').view(np.int8).reshape(-1)
container.reshape(-1)[:argmax_bytes.size] = argmax_bytes   # 平铺在容器线性内存头部
```

即 **GM 上的物理布局与逻辑视图无关**：

- 容器前 \(N \cdot C \cdot H_o \cdot W_o \cdot 4\) 字节 = 按 \((n c,\ h,\ w)\) 顺序紧密排列的 **int32 小端** argmax 字节流（字节偏移 \(= (n c \cdot H_o W_o + h \cdot W_o + w) \times 4\)）；
- 其后所有字节（\(N C k_H k_W \cdot maskW - N C H_o W_o \cdot 4\) 字节）= 0；
- argmax 写入与"每通道 `[kH*kW, maskW]` 平面边界"**无关**，是纯粹的线性字节流前缀。

> 因此 kernel 对 indices 的操作只有两类：① 把 argmax int32 连续写到 `mask_gm + nc*Ho*Wo*4`；② 把 `[validBytes, totalBytes)` 区间清零。两者区间互补、无交叠。

### 2.4 精度设计（0 误差路径）

- **计算 dtype 分派**（CANN 9.1.0 c220/m200 头文件实测：`Max/Compare/Select` 仅支持 half/float）：
  - fp16 / fp32：**原 dtype 直通**做 Max/Compare/Select；
  - bf16：`Max/Compare/Select` 不支持 bf16，加载后经 `Cast(bf16→fp32, CAST_NONE)` 入行缓存，**以 fp32 为计算 dtype**，out 写回前 Cast 回 bf16（`CAST_RINT`，值为输入精确复制，无舍入）。
- fp16/bf16 → fp32 是**精确且保序**的转换，因此（直通或经 fp32 中转的）比较结果与 golden 的 fp32 视图**完全一致**，tie 行为一致，输出为输入值的精确复制，**无任何舍入**。
- argmax 索引通道在向量侧以 **float 承载**（`Select` 不支持 int32，见 §5.5），写回前 `Cast(fp32→int32)`；float 精确整数上限为 \(2^{24}\)，故 host 校验 \(H\cdot W\le2^{24}\)（argmax 最大值 \(HW-1\) 可精确表示），值域内转换无任何舍入。
- `out` 相对 golden 误差恒为 0，`indices` 完全一致（int8 阈值 0）。精度标准（0.001 阈值）大幅裕量。
- 任务书约束"FLOAT 在 Atlas 训练系列转 FP16 计算有精度损失"是原 TBE 的实现行为；本设计在 910B3 上对 fp32 直接以 fp32 计算，**精度优于原实现**，不违反验收标准（误差不劣化）。

### 2.5 泛化范围（依据 153 条用例 + 参数说明）

| 维度 | 范围 |
|------|------|
| dtype | float16 / float32 / bfloat16（bf16 仅 800T A2） |
| 维度 | 4D NCHW、3D CHW（用例含 `[C,H,W]`，按 N=1 归一） |
| shape | `[1,1,4,4]` ~ `[32,512,7,7]`、`[2,64,224,224]` 等，N/C/H/W 任意泛化 |
| kernel | 2/3/7（用例），约束为任意 >0，kH≠kW 合法 |
| stride | 0（缺省=k）、1、2（用例），任意 >0 泛化，sH≠sW 合法 |
| padding | 0~3（用例），约束 0 ≤ p ≤ k/2 |
| ceilMode | 用例全为 False，泛化必须支持 True |
| 非连续 | out 支持非连续（host 侧框架转连续后入 kernel） |
| 平面规模 | \(H\cdot W\le2^{24}\)（argmax float 承载的精度约束，host 校验，见 §2.4/§5.5） |
| 推理系列 ceilMode | 任务书约束 4（300V Pro 上 ceilMode=True 的部分 stride 场景原算子不支持）：与原算子行为对齐，不纳入泛化承诺范围 |

## 3. 总体方案

```
GM:x ──DataCopyPad + 槽内 -inf 复位──► UB 整行缓存(kH 行 ring buffer, [0,padLeft)∪右 pad 恒 -inf)
                                     │  逐输出行 oh × 输出列块(curTW ≤ 64)
                                     ▼
              kH×kW 窗口内：Gather + Compare(GT) + Select ──► curMax[Tcomp], curIdxF[float→Cast int32]
                                     │
              out(UB→GM, DataCopyPad)  │  argmax(逐 tile 直写, DataCopyPad→GM 线性前缀, 含 ow0 偏移)
              mask 尾区: Duplicate(0)+DataCopyPad 多核分摊清零(A/B 区, 无 SyncAll)
```

**核心设计决策**：

1. **单 kernel 泛化**：一套通用 kernel 支持全部合法参数，`SetTilingKey(0)`；`(k,s)` 高频组合模板特化未实施（性能优化后置项，见 §7.1 #9）。
2. **按 NC 维分核**，核内逐平面计算；行缓存保证**每个输入元素只搬一次**。
3. **Wo 维向量化 + 全向量指令 64 元素分块**：dav-c220 实测 count>64 的整 tensor 向量调用（counter mask / repeat 尾块）不可靠（见 §9），所有向量 API（Compare/Select/Adds/Muls/Duplicate/Cast/CreateVecIndex/Gather）统一按 64 元素分块调用；窗口候选统一 `Gather` 按槽内字节偏移取值（任意 sW，规避非对齐向量读）。
4. **mask 清零多核分摊**，与各核计算自然并行（区间互不交叠）。
5. **无需 workspace**：所有中间量驻留 UB，argmax/out 直写 GM。
6. **SoC 支持范围**：v1 仅交付 c220 路径（910B3/910B4/910_93，`DataCopyPad` 非对齐搬运 + 槽内 -inf 复位）；m200（300V Pro / 310P3）**无 DataCopyPad**（CANN 9.1.0 头文件实测 `ASCENDC_REPORT_NOT_SUPPORT`），其“对齐 DataCopy + Duplicate 填 pad + 读改写”替代路径本期**未实现**（tiling `isM200` 恒 0，OpDef 仅注册 ascend910b/ascend910_93）。
7. **计算 dtype 分派**：fp16/fp32 原 dtype 直通；bf16 加载后 Cast 为 fp32 计算；argmax 索引通道 float 承载、写回前 Cast 至 int32，host 校验 \(H\cdot W\le2^{24}\)（§2.4/§5.5）。

## 4. Host 侧设计（Tiling 重点）

### 4.1 工程结构

```text
max_pool2d_with_mask/                       # 对应 ops-nn experimental/pooling/max_pool2d_with_mask
├── op_host/
│   ├── max_pool2d_with_mask.cpp            # 原型注册 + InferShape + TilingFunc（单文件）
│   └── max_pool2d_with_mask_tiling.h       # TilingData 结构定义
├── op_kernel/
│   └── max_pool2d_with_mask.cpp            # kernel 入口 + KernelMaxPool2dWithMask<T,Tcomp>（单文件）
├── build.sh / deploy.sh                    # msopgen 编译；产物同步 opp/vendors/custom_nn（本机自测）
├── tests/                                  # 用例编排 run_cases.py + analyze_case.py + testdata（交付件 2）
├── unittest/                               # aclnn 两段式 runner（自测驱动，读 meta.ini+x.bin）
└── README.md                               # 算子 readme（ops-nn 仓库规范，交付件 4）
```

### 4.2 InferShape / 参数归一与校验

1. **维度归一**：3D `[C,H,W]` → `N=1, C=shape[0]`；4D 直取。3D/4D 差异仅影响 host 输出 shape 写回（InferShape 直接处理），tiling 不下发 is3D 标志、kernel 无差别。
2. **属性归一**：
   - `kernelSize/padding` 长度 1 → 复制为 2；长度 2 直取；
   - `stride` 长度 0 → 取 kernelSize；长度 1 → 复制为 2；
   - `dilation` 必须为 `{1,1}`，否则 host 报错（`OP_LOGE` + 返回失败）；
   - `ceilMode` 为 optional 属性，缺省按 false 处理。
3. **合法性校验**：`k>0`、`s>0`、`0 ≤ p ≤ k/2`；`H+2pH ≥ kH`、`W+2pW ≥ kW`（保证 Ho,Wo ≥ 1）；`H·W ≤ 2^24`（argmax float 承载精度约束，见 §5.5，超出时 OP_LOGE 拦截并与原 TBE 支持范围对齐说明）。
4. **输出 shape**：按 §2.2 公式（含 ceilMode 修正），`out=[N,C,Ho,Wo]`、`indices=[N,C,kH*kW,maskW]`。
5. **dtype**：`x` 与 `out` 一致，`indices` 固定 int8；bf16 仅 910B3 放行（按 `platformInfo` SoC 版本过滤）。

### 4.3 TilingData 结构

```cpp
struct TilingData {                // op_host/max_pool2d_with_mask_tiling.h, 与实现一致
    // ---- 基础形状/属性（3D 已归一 N=1, 属性已归一二元组）----
    int64_t n, c, h, w;            // 输入
    int64_t ho, wo, hw;            // 输出, hw = ho*wo
    int64_t nc;                    // n*c
    int64_t kH, kW, sH, sW, pH, pW;
    int64_t maskW;                 // indices 最后一维 = ((hw+15)/16+1)*32

    // ---- NC 分核: big core 在前且每核多 1 平面 ----
    int64_t coreNum;               // = min(aivNum, nc)
    int64_t planesPerCore;         // 小核平面数 = nc/coreNum
    int64_t bigCoreNum;            // 多 1 平面的大核数 = nc%coreNum
    int64_t zeroPlanesPerCore;     // [预留] 清零分摊(实现改为 kernel 按 coreNum 现场均摊, 字段未用)
    int64_t zeroBigCoreNum;        // [预留] 同上

    // ---- 行缓存/分块 ----
    int64_t rowBufElems;           // 整行槽元素数 = alignUp(padLeft+(wo-1)*sW+kW, 32B)
    int64_t tileWo;                // 单 tile 输出宽度(≤64, 见 §4.5)
    int64_t tileWoNum;             // ceil(wo / tileWo)

    // ---- 路径选择 ----
    int64_t isM200;                // 恒 0(m200 路径未实现)
    int64_t strideMode;            // 0: sW=1; 2: sW>=2(占位, kernel 统一 Gather 不分派)
    int64_t esize;                 // 输入 dtype 字节数
};
```

字段说明：is3D 由 host InferShape 直接处理不下发；argmax 线性前缀/总字节由 hw/maskW 现算；argmax 不攒块故无 argmaxStoreRows；尾块 curTW=wo-ow0 现算故无 tileWoTail。

### 4.4 分核策略

- **主切分维度：NC**（每平面计算量 \(H_o W_o k_H k_W\) 完全均等 → 天然负载均衡），**big core 在前**：

```text
coreNum       = min(aivNum, N*C)                   // aivNum 取 PlatformAscendC.GetCoreNumAiv
planesPerCore = N*C / coreNum                      // 小核平面数
bigCoreNum    = N*C % coreNum                      // 前 bigCoreNum 个核每核多 1 平面
```

  核内平面区间由 kernel 按 `GetBlockIdx` 现算（big core 段连续在前）；NC=1 的极小 shape 自动退化为单核（该场景启动开销占主导，详见 §8）。
- **二级切分（未实施）**：910B3/910B4 的 AIV 为 **40 核**（`Ascend910B3.ini`/`Ascend910B4-1.ini`：ai_core_cnt=20、vector_core_cnt=40），用例中 `N*C < 40` 的 shape（m01 NC=16、m03 NC=3 等）欠载面大。W_out 维二级切分（每核处理平面内列区间）仍列为性能调优首选，v1 未合入。
- **mask 清零分摊（A/B 区，无跨核同步）**：A 区 = argmax 线性前缀 `[0, NC*Ho*Wo*4)`，被各核逐平面精确覆写，**无需清零**；B 区 = 尾部 `[NC*Ho*Wo*4, NC*kH*kW*maskW)`，kernel 按 `coreNum` 现场均摊（每核段 32B 对齐取整，UB→GM `DataCopyPad` 字节精确写）。A/B 互补无交叠 ⇒ **无需 SyncAll**（v1 初版曾用 SyncAll 等待清零完成，重构后移除，153/153 回归通过）。

### 4.5 UB 预算与 W 维 tile 计算（泛化安全的核心）

单核 UB（910B3/910B4 为 192KB、310P3 为 256KB，以 `PlatformAscendC` 运行时查询为准）分配。**esizeComp 取计算 dtype 字节数**（fp16=2B；fp32/bf16=4B，bf16 行缓存为 Cast 后的 fp32，见 §2.4）：

| 缓冲 | 大小 | 说明 |
|------|------|------|
| rowBuf | `kH × rowBufElems × esizeComp` | 行缓存 ring buffer（整行，固定项先行扣除） |
| stageRow | `rowBufElems × esize` | bf16 搬运暂存行（仅 bf16 分配） |
| cand / curMax | `tileWo × esizeComp × 2` | 窗口候选 / 当前最大值 |
| curIdxF / idxCandF / idxStepF | `tileWo × 4B × 3` | argmax 三通道（float 承载） |
| gatherIdx | `tileWo × 4B` | Gather 槽内字节偏移 |
| cmpMask | `tileWo / 8 B`（32B 对齐） | Compare 输出的 CMPMASK |
| outStage / idxStage | `tileWo × (esize + 4)` | 写回暂存（bf16 out 需 Cast） |
| zeroBuf | 8KB | mask 清零用 |

**行槽坐标系（v3 改：整行缓存 + 显式 padLeft）**：槽内 `[0,padLeft)` 恒 -inf 左 pad，`[padLeft, padLeft+W)` 为有效数据，其余恒 -inf 右 pad；`padLeft = alignUp(pW, 32B/esizeComp)` 保证数据区 32B 对齐且左 pad 覆盖 pW。Gather 槽下标最大 `padLeft+(wo-1)*sW-pW+kW-1`，故：

$$\text{rowBufElems} = \mathrm{alignUp}\big(\text{padLeft} + (W_o-1)\cdot s_W + k_W,\ 32B/\text{esizeComp}\big)$$

整行缓存使 Wo 方向多 tile 时行只载一次、tile 共享窗口行（§5.3）；rowBuf 作为固定项先行扣除，剩余 UB 反推 tileWo。

**tileWo 推导**（host 计算）：

```text
fixedBytes = kH * rowBufElems * esizeComp   (+ bf16: rowBufElems * esize 暂存行)
avail      = ubSize - fixedBytes - 8KB(栈/标量系统预留)
tileWoMax  = avail / perElemBytes           // perElem ≈ 2*esizeComp + 5*4B + esize + 1B
tileWoMax  = min(tileWoMax, 64)             // 硬上限: curTW>64 实测整核静默不写(§9)
tileWo     = min(alignUp(Wo, alignElem), tileWoMax 向下取 alignElem 整倍), 兜底 alignElem
tileWoNum  = ceil(Wo / tileWo)
```

用例量级示例：

| 用例 | W | kH | rowBuf(整行) | tileWo | UB 合计 |
|------|---|----|--------------|--------|---------|
| m02 56×56 k3s2 fp32 | 56 | 3 | ~0.9KB | 32（整 Wo 对齐） | < 16KB |
| l04 224×224 k3s2 fp16 | 224 | 3 | ~1.6KB | 64（达上限）×2 tile | < 24KB |
| 极限 4096 宽 k7s2 fp32 | 4096 | 7 | ~115KB | 64（达上限）×32 tile | ~120KB |

> 设计保证：192KB UB 下常规 shape tileWo 恒 ≥ alignElem（兜底分支不可达）；极端大 W（数万级）使 `kH*rowBufElems` 固定项逼近/超出 UB 为已知限制（H·W≤2^24 内 W 理论可达 2^24，用例最大 W=224 远未触及，host 未显式拦截，见 §9）。

### 4.6 Tiling 注册

- msopgen 工程标准 `RegisterTilingFunc`；`SetTilingKey(0)` 单一通用 kernel，dtype 实例化由框架按 OpDef 注册的 dtype 列表生成（kernel 内 `DTYPE_X` 宏 ∈ {half, float, bfloat16_t}）；`strideMode` 经 TilingData 下发（占位字段，kernel 统一 Gather 不分派）。
- 模板特化（性能后置项，未实施）：tilingKey 追加 `(kH,kW,sH,sW)` 命中内置组合的 bit，kernel 侧 `switch` 分发到 constexpr 模板实例。

## 5. Kernel 侧设计（实现重点）

### 5.1 类骨架与主流程

```cpp
template <typename T, typename Tcomp>      // Tcomp = T; bf16 时 Tcomp = float(§2.4 中转)
class KernelMaxPool2dWithMask {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, GM_ADDR tiling);
    __aicore__ inline void Process();        // ZeroIndices + 逐平面 ProcessPlane

private:
    __aicore__ inline void ZeroIndices();                        // ① B 区尾部分摊清零(无 SyncAll)
    __aicore__ inline void ProcessPlane(int64_t plane);          // ② 单平面全流程
    __aicore__ inline void LoadRow(int64_t xPlaneOff, int64_t r, // ③ 载入整行到 ring 槽
                                   LocalTensor<Tcomp> rowBuf);   //   (+尾部 -inf 重置, §5.4)
    // ④ 窗口计算/写回内联于 ProcessPlane: 统一 Gather + Compare/Select + 逐 tile 直写
    // 全部向量 API 经 64 元素分块 wrapper(类外模板函数): CompareGtExact/GatherExact/
    // SelectExact/AddsExact/MulsExact/DuplicateExact/CastExact/CreateVecIndexExact(§5.5)

    GlobalTensor<T> xGm_, outGm_;   GlobalTensor<uint8_t> idxGm_;
    TBuf<VECCALC> rowBuf_, stageRowBuf_, candBuf_, curMaxBuf_, curIdxFBuf_,
                  idxCandFBuf_, idxStepFBuf_, gatherIdxBuf_, cmpMaskBuf_,
                  outStageBuf_, idxStageBuf_, zeroBuf_;
    int64_t planeStart_, planeEnd_;            // 本核平面区间(big core 在前)
    // ... 形状/属性/tiling 成员(n_..nc_, kH_..pW_, maskW_, rowBufElems_, tileWo_, padLeft_ 等)
};

extern "C" __global__ __aicore__ void max_pool2d_with_mask(
    GM_ADDR x, GM_ADDR out, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    // dtype 实例化由框架按 OpDef 注册 dtype 生成(DTYPE_X ∈ {half, float, bfloat16_t})
    KernelMaxPool2dWithMask<DTYPE_X, CompType<DTYPE_X>::type> op;
    op.Init(x, out, indices, tiling);        // GET_TILING_DATA 在 Init 内完成
    op.Process();
}
```

`Process()` 主循环：

```cpp
__aicore__ inline void Process()
{
    ZeroIndices();                         // B 区清零, 与其他核的计算/清零天然并行(无 SyncAll)
    for (int64_t plane = planeStart_; plane < planeEnd_; ++plane) {
        ProcessPlane(plane);
    }
}
```

### 5.2 mask 尾区清零（ZeroIndices，A/B 区无同步）

- **A 区** `[0, NC*Ho*Wo*4)`：argmax 线性前缀，各核对本核平面逐 tile 精确覆写（int32 连续 4B 粒度），无需清零。
- **B 区** `[NC*Ho*Wo*4, NC*kH*kW*maskW)`：尾部清零。`Duplicate(zeroBuf, 0)` 一次；按 `coreNum` 现场均摊（`per = alignUp(bTotal/coreNum, 32B)`，本核段 `[bBegin+blkIdx*per, min(+per, bEnd))`），循环 `DataCopyPad(idxGm + zb0, zeroBuf, len)`，`len ≤ 8KB`；MTE3 顺序写，接近满带宽。
- **UB→GM DataCopyPad 字节精确**：无 §5.4 所述 GM→UB 方向的尾块污染，B 区起点（4B 对齐）写不越界进入 A 区。
- **无 SyncAll**：A/B 区间互补无交叠，各核清零段互不交叠；v1 初版曾用 `SyncAll` 等待全核清零完成，重构后移除，153/153 回归通过。
- m200 路径未实现（`isM200` 恒 0）。

### 5.3 单平面处理：行缓存 ring buffer（消除重复搬运的关键）

逻辑行号 \(r \in [-p_H,\ H+p_H)\)，输出第 oh 行需要输入行 \(r \in [oh\cdot s_H - p_H,\ oh\cdot s_H - p_H + k_H)\)。ring 槽号 = `r % kH`（仅载 r∈[0,H) 的有效行，取模无负数）。

```cpp
__aicore__ inline void ProcessPlane(int64_t plane)
{
    // 平面开始: kH 个槽整槽 DuplicateExact(-inf) 复位一次
    //   → 左 pad [0,padLeft) 与右 pad 区常驻 -inf; 非法行槽自然为 -inf, 无需 negInfRow
    int64_t nextRow = 0;
    for (int64_t oh = 0; oh < ho_; ++oh) {
        // 单调载入: 仅载新进入窗口的行 [max(nextRow,0), min(oh*sH-pH+kH-1, H-1)]
        //   sH>kH 时窗口外整行自然跳过; 每行全程仅载 1 次
        for (int64_t t = 0; t < tileWoNum_; ++t) {   // oh 外 tile 内(恒次序)
            // tile 共享行缓存, 索引向量按 ow0=t*tileWo 在 tile 内重算(§5.5)
            // 写回地址含 tile 偏移: out[oh*Wo+ow0], argmax 字节 (oh*Wo+ow0)*4
        }
    }
}
```

- **搬运量**：每平面被引用输入元素恰好加载 1 次（相对朴素逐窗口读取的 \(k_H/s_H \times k_W/s_W\) 倍放大，k3s2 从 2.25× → 1×，k7s2 从 12.25× → 1×）。
- **sH > kH 的跳跃窗口**：窗口外整行直接跳过不加载（`nextRow` 前跳），ring buffer 语义不变，无多余搬运。
- **多 tile 循环次序（v3 改，与 v2 相反）**：v2 设计“rowBuf 仅缓存 tile 列宽、多 tile 时 t 外 oh 内独立滑窗”；实现为**整行缓存 + oh 外 t 内恒次序**——rowBuf 缓存整行（§4.5），tile 间共享窗口行零重复载入；v1 曾按 v2 次序实现，实测每 tile 重复载行（tileWoNum 倍搬运），交换后回归 153/153 确认。
- **非法行**：kh 循环内 `ih<0 || ih>=H` 直接 `continue`（槽复位语义下等价全 pad 行，cand=-inf 恒不更新）。
- **-inf 常量**：`DuplicateExact(槽, (Tcomp)(-__builtin_inff()))`（device 无 `INFINITY` 宏）；三种 dtype -inf 位模式 fp16 `0xFC00` / bf16 `0xFF80` / fp32 `0xFF800000`，由 `Duplicate` 直接生成。

### 5.4 载入一行（LoadRow）：DataCopyPad 直搬 + 尾部 -inf 重置

槽坐标系见 §4.5（`[0,padLeft)` 左 pad 恒 -inf，数据写 `slot[padLeft]`）：

```cpp
DataCopyPad(slot[padLeft], xGm_[xPlaneOff + r*W], {1, W*esize, 0, 0, 0},
            {/*isPad=*/false, 0, 0, 0});       // 不带 pad 参数
// GM→UB DataCopyPad 按 32B 块搬运: blockLen 非 32B 整倍时, 尾部块会把 GM 后续数据
// (下一行行首)带入槽内右 pad 区 → 每次载入后重置 [padLeft+W, alignUp) 为 -inf
for (i in [padLeft+W, alignUp(padLeft+W, 32B/esizeComp)):
    slot.SetValue(i, -inf);
```

- GM 首地址 `xGm_[...]` 非 32B 对齐由 DataCopyPad 硬件非对齐能力支持。
- 左 pad 区 `[0,padLeft)` 不被搬运触及，由平面开始时的整槽 -inf 复位常驻（§5.3）；alignUp 之后的右 pad 区同理由复位常驻（污染不超出一个尾部 32B 块）。
- **实测硬件行为（v2 未预见）**：GM→UB 方向尾块污染 UB 右 pad 区（用例 F185 右 pad 列 argmax 曾指向下一行行首）；UB→GM 方向字节精确无此问题。bf16 路径先搬暂存行再 `Cast(CAST_NONE)` 入槽，Cast 按 count 精确写，天然免疫。
- v2 的“DataCopyPad leftPadding/rightPadding 一步填 -inf”方案未采用（pad 参数无法覆盖尾部污染，且左 pad 已由复位兜底）。
- 非法行（r<0 / r≥H）不进 LoadRow（kh 循环 skip，§5.3）；m200 路径未实现。

### 5.5 窗口内向量化 Max/Argmax（ComputeRowBlock，核心计算）

对一个列块（`curTW ≤ 64` 个输出列，尾块 `curTW = wo - ow0`；bf16 入行缓存时已 Cast 为 fp32，按 fp32 路径计算）。**索引通道 float 承载**（`Select` 仅支持 half/float，CANN 9.1.0 c220/m200 头文件实测）：

```cpp
// tile 级索引向量(向量生成, 避免标量写 UB 后向量读的可见性问题):
//   gather 槽内字节下标 gatherIdx[j] = (padLeft + (ow0+j)*sW - pW) * esizeComp
//   argmax 步进        idxStepF[j]   = (ow0+j)*sW            (float)
CreateVecIndexExact(gi, 0, curTW);
MulsExact(gi, gi, sW*esizeComp, curTW);
AddsExact(gi, gi, (padLeft + ow0*sW - pW)*esizeComp, curTW);
CreateVecIndexExact(idxStepF, 0.0f, curTW);
MulsExact(idxStepF, idxStepF, (float)sW, curTW);   // sW=1 时跳过
AddsExact(idxStepF, idxStepF, (float)(ow0*sW), curTW);

DuplicateExact(curMax, -inf, curTW);               // 全 pad 窗口 → out=-inf
DuplicateExact(curIdxF, 0.0f, curTW);              //           → argmax=0
for (kh = 0; kh < kH; ++kh) {
    ih = oh*sH - pH + kh;
    if (ih < 0 || ih >= H) continue;               // 非法行: 槽复位语义下等价全 pad
    slot = rowBuf[(ih % kH) * rowBufElems];
    for (kw = 0; kw < kW; ++kw) {
        GatherExact(cand, slot, gatherIdx, kw*esizeComp, curTW);  // 任意 sW 统一 Gather
        AddsExact(idxCandF, idxStepF, (float)(ih*W + kw - pW), curTW);
        CompareGtExact(cmpMask, cand, curMax, curTW);   // 严格大于 → tie 保留先行先列
        SelectExact(curMax,  cmpMask, cand,    curMax,  curTW);
        SelectExact(curIdxF, cmpMask, idxCandF, curIdxF, curTW);
    }
}
CastExact(idxStage, curIdxF, CAST_RINT, curTW);         // fp32→int32, 整数值无舍入
// bf16: CastExact(outStage, curMax, CAST_RINT, curTW)  // fp32→bf16, 值为输入精确复制
```

**与 v2 的三处实现差异**：

1. **统一 Gather（strideMode 不分派）**：v2 的 sW=1 直接偏移 / sW=2 DeInterleave 预拆 / sW≥3 Gather 三分派未实施——直接偏移要求向量地址按 dtype 对齐，而槽下标 `padLeft+(ow0+j)*sW-pW+kw` 任意；Gather 以槽内字节偏移取值天然规避（c220 支持 half/float），strideMode 字段保留占位。
2. **全向量 API 64 元素分块**：dav-c220 实测 count>64 的整 tensor 向量调用（counter mask / repeat 尾块）不可靠（fp16 count=112 单 repeat 亦出错）；`Compare/Select/Adds/Muls/Duplicate/Cast/CreateVecIndex/Gather` 统一包 `*Exact` wrapper 按 64 分块，mask 类按位偏移切片（`mask[done/8]`），Compare 用 `SetMaskCount + SetVectorMask<C,COUNTER>` 模式。
3. **基本 API 生成索引**：v2 的 `ArithProgression`（高阶 API）未采用，用 `CreateVecIndex + Muls + Adds`（纯 basic）；`CreateVecIndex` 步长恒 1，仿射变换由 Muls/Adds 完成。

**正确性论证**：

1. **tie 语义**：Compare 用 `GT`（非 `GE`），相等不更新；扫描顺序 kh 外、kw 内 = 线性索引 `ih*W+iw` 升序 ⇒ 首个最大值被保留，与 golden `vfull > cur` 逐元素一致。
2. **pad 语义**：pad 位 cand=-inf，`-inf > cur` 恒 false（cur 初值 -inf）⇒ 永不更新，等价于 golden 的 `valid` 掩码；全 pad 窗口保持 `(-inf, 0)`，一致。
3. **ceilMode 右越界**：行缓存右 guard 区填充 -inf（§4.5 的 rowBufElems 覆盖 \((W_o-1)s_W + k_W\) 上界），越界列等价 pad，一致。
4. **argmax 值域与 float 承载**：`ih*W+iw ≤ H*W-1 ≤ 2^24-1`（host 校验 \(H\cdot W\le2^{24}\) 保证），float 精确表示整数无舍入，`Cast(fp32→int32)` 精确；pad 位 idxCand 可为负值（如 iw<0），因 cand=-inf 永不被选中，不影响结果；原 int32 方案的 \(2^{31}\) 上界收紧为 \(2^{24}\) 是本设计相对任务书泛化范围的主动约束（单平面 1677 万元素，覆盖全部现实场景）。

### 5.6 输出写回（逐 tile 直写，含 ow0 偏移）

- **out**：每 (oh, tile) 一次 `DataCopyPad(outGm_[plane*hw + oh*Wo + ow0], outRow, curTW*esize)`，非对齐由 DataCopyPad 处理；fp16/fp32 时 `outRow = curMax`（输入精确值），bf16 时先 Cast 回 bf16（值不变），无精度损失。
- **argmax**：`Cast(fp32→int32)` 后的 `idxStage` 同节奏直写线性前缀：`DataCopyPad(idxGm_ + plane*hw*4 + (oh*Wo+ow0)*4, idxStage, curTW*4)`（字节偏移，int32 语义，布局见 §2.3）。
- **ow0 偏移（实测修复）**：多 tile 时写回地址必须含 `ow0 = t*tileWo`，否则 t>0 覆写 t=0 区域（v1 调试实测 bug，用例 050 暴露）。
- v2 的“argmax 攒块 argmaxStoreRows 行写回”未实施：直写粒度 curTW*4 ≤ 256B 偏小，但 MTE3 碎片在 153 用例规模下不构成瓶颈；大 shape 性能调优时可再评估攒块。
- m200 对齐 + 读改写路径未实现。

### 5.7 同步与流水

- v1 采用 `PipeBarrier<PIPE_ALL>` 保守同步：槽复位(V) → 行载入(MTE2) → 窗口计算(V) → 写回(MTE3) → 下一 tile 复写缓冲，每阶段一道屏障，正确性优先。
- v2 设计的 `SetFlag/WaitFlag` 显式 event 三段软件流水（载入 ∥ 计算 ∥ 写回）未实施，列为性能优化后置项（§7.1 #4）。

### 5.8 边界与异常场景覆盖

| 场景 | 处理 |
|------|------|
| 3D 输入 | host 归一 N=1；kernel 无感 |
| stride=0 缺省 | host 归一 s=k |
| kH≠kW / sH≠sW / pH≠pW | 全部独立参数化，无对称假设 |
| s > k（跳跃窗） | ring buffer 取模覆盖，无分支 |
| 全 pad 输出点 | 初值 (-inf, 0)，与 golden 一致 |
| Ho*Wo 非 16 对齐 | maskW 公式含 +1 余量，写回用 DataCopyPad 精确长度 |
| 极小 shape（NC=1） | 单核串行，清零/计算合并，减少同步开销 |
| Wo 尾块 | 尾块 curTW=wo-ow0，各 *Exact wrapper 按实际 count 分块截断，无越界读写 |
| 多 tile（Wo>64） | oh 外 tile 内共享整行缓存，写回地址含 ow0 偏移（§5.3/§5.6） |

## 6. Ascend C 基础 API 清单

### 6.1 Kernel 侧（AscendC 命名空间）

| 分类 | API | 用途 |
|------|-----|------|
| 资源管理 | `TPipe`、`TBuf`、`InitBuffer` | UB 缓冲分配（TBuf 直管，无队列） |
| 数据搬运 | **`DataCopyPad`**（`DataCopyExtParams` / `DataCopyPadExtParams`，仅 c220） | GM→UB 非对齐行加载（不带 pad 参数）；UB→GM 字节精确写（out/argmax/清零）；GM→UB 尾块污染见 §5.4 |
| 数据重排 | **`Gather`**（half/float，uint32 槽内字节偏移） | 窗口候选统一取值（任意 sW，规避非对齐向量读） |
| 向量计算 | **`Compare`**（`CMPMODE::GT`，half/float，`SetMaskCount`+COUNTER 模式） | argmax 更新条件（严格大于，保证 tie 语义） |
| 向量计算 | **`Select`**（`SELMODE::VSEL_CMPMASK_SPR`，仅 half/float） | 按 CMPMASK 同步更新 curMax 与 float 承载的 curIdxF |
| 向量计算 | **`Cast`**（bf16→fp32 CAST_NONE、fp32→bf16/fp32→int32 CAST_RINT） | bf16 中转；argmax 写回前 float→int32 |
| 向量计算 | `Adds` / `Muls`（int32/float） | 索引向量仿射变换（gather 偏移、argmax 步进/候选） |
| 向量计算 | `Duplicate`（half/float/int32） | 槽 -inf 复位、curMax/curIdxF 初始化、清零缓冲 |
| 索引生成 | `CreateVecIndex`（int32/float，步长恒 1） | 等差序列基底（basic API，替代 v2 的 ArithProgression） |
| 标量写 | `LocalTensor::SetValue` | LoadRow 尾部 -inf 重置（§5.4） |
| 同步 | `PipeBarrier<PIPE_ALL>` | 阶段间保守同步（V↔MTE2/MTE3） |
| mask 控制 | `SetMaskCount` / `SetVectorMask<C,MaskMode::COUNTER>` / `ResetMask` / `SetMaskNorm` | Compare counter mask 计数 |
| 系统 | `GetBlockIdx` | 核内平面区间与清零段定位（核数由 TilingData.coreNum 下发） |
| 类型 | `GlobalTensor<T>`、`LocalTensor<T>`、`half`、`bfloat16_t` | 基本数据抽象 |
| 自研封装 | `CompareGtExact` / `GatherExact` / `SelectExact` / `AddsExact` / `MulsExact` / `DuplicateExact` / `CastExact` / `CreateVecIndexExact` | 全部向量 API 的 64 元素分块 wrapper（§5.5，dav-c220 实测约束） |

### 6.2 Host 侧

| 分类 | API | 用途 |
|------|-----|------|
| 框架注册 | `IMPL_OP` / `RegisterTilingFunc` / `OP_ADD` 系列宏 | 算子原型与 tiling 注册 |
| Tiling 上下文 | `TilingContext`、`GetInputShape`、`GetAttrs`、`SetTilingKey`、`SetBlockDim`、`SetWorkspaces` | tiling 数据落盘与运行参数 |
| 平台信息 | `PlatformAscendC::GetCoreNumAiv`、SoC 版本查询 | 分核数与 bf16 支持判定 |
| 数据落盘 | `SaveTilingData` / `GET_TILING_DATA`（kernel 侧配对） | TilingData 传递 |
| 校验 | `OP_LOGE` / `OP_CHECK` 系列 | dilation≠1 等非法参数拦截 |

## 7. 性能设计与优化（对应验收：≥ TBE 95%）

### 7.1 优化点清单（按收益排序）

| # | 优化 | 收益机制 | 状态（v3） |
|---|------|----------|------------|
| 1 | **行缓存 ring buffer（整行）** | 每输入元素 GM→UB 仅 1 次，消除 \(k_H k_W/(s_H s_W)\) 倍重复搬运（k3s2:2.25×、k3s1:9×、k7s2:12.25× → 1×） | 已实施 |
| 2 | **Wo 维全向量化（64 元素分块）** | 每 ≤64 lane 一批，kH·kW 次 Gather+Cmp+Select 完成整块输出；无标量循环 | 已实施（受 §9 硬件约束，上限 64） |
| 3 | **mask 清零多核分摊 + 大粒度顺序写** | 清零量 = mask 总字节 - 有效字节（约占 mask 流量 60~78%，见 §7.2），按核均分、8KB/次 DataCopyPad 打满 MTE3 | 已实施（A/B 区无同步） |
| 4 | 三段软件流水 | 行加载(MTE2) ∥ 向量计算(V) ∥ 写回(MTE3)，隐藏搬运 | **未实施**（v1 PipeBarrier 保守同步，性能后置） |
| 5 | argmax 攒块写回 | 每平面 GM 写次数从 Ho 次降为 Ho/storeRows 次 | **未实施**（逐 tile 直写 ≤256B；当前规模非瓶颈） |
| 6 | DataCopyPad 非对齐搬运 + 槽内 -inf 复位 | 任意 GM 偏移单指令搬运；pad 区由平面级整槽复位 + LoadRow 尾部重置维护 | 已实施（v2 “pad 参数填 -inf”方案未采用，见 §5.4） |
| 7 | **fp16/fp32 dtype 直通比较（无 Cast）** | fp16/fp32 不升 fp32，UB 占用省、无 Cast 指令（bf16 受硬件限制必须 Cast 中转，见 §2.4） | 已实施 |
| 8 | **统一 Gather 窗口取值** | 槽内字节偏移取值，任意 sW 无需分派，规避非对齐向量读 | 已实施（v2 的 sW 三分派/DeInterleave 未采用） |
| 9 | 常用 (k,s) 模板特化 | k2s2/k3s2/k3s1/k7s2 编译期常量，kH·kW 循环全展开、地址计算常量折叠 | **未实施**（性能后置） |
| 10 | tileWo 32B 对齐 | 列块宽度按 alignElem 取整，写回粒度对齐 | 已实施 |

### 7.2 典型用例性能估算（910B3：HBM ≈ 1.6TB/s，AIV 40 核 = vector_core_cnt 40，以 PlatformAscendC 实测为准）

| 用例 | 输入 | 输出 out | mask(总/有效) | 总流量 | 搬运下界@60%BW | 计算下界 | 判定 |
|------|------|----------|---------------|--------|----------------|----------|------|
| F046 m01 `[2,8,32,32]` f32 k3s2 | 128KB | 32KB | 147KB/32KB | ~0.3MB | ~0.6 µs | ~0.1 µs | 启动开销主导 |
| F061 m02 `[4,16,56,56]` f32 k3s2 | 0.8MB | 0.2MB | 0.9MB/0.2MB | ~1.9MB | ~2 µs | ~0.4 µs | 搬运 bound |
| F076 m03 `[1,3,224,224]` f16 k3s2 | 0.3MB | 75KB | 0.4MB/0.15MB | ~0.9MB | ~1 µs | ~0.1 µs | NC=3 欠载，二级切分备选 |
| F092 m04 `[8,64,56,56]` f16 k3s2 | 3.2MB | 0.8MB | 7.4MB/1.6MB | ~11MB | ~12 µs | ~2.5 µs | 搬运 bound |
| F151 l04 `[2,64,224,224]` f16 k3s2 | 12.8MB | 3.2MB | 28.9MB/6.4MB | ~45MB | ~47 µs | ~8 µs | mask 清零占半 |
| F163 l05 `[32,512,7,7]` f16 k3s2 | 0.5MB | 0.13MB | 0.9MB/0.26MB | ~1.5MB | ~1.6 µs | ~0.2 µs | NC 大，满载 |

> mask 流量是 TBE 同样必须承担的语义开销（容器布局由 golden 固定），双方基准一致；本方案通过"清零分摊 + 顺序大粒度写"保证该段贴近带宽极限。
> AIV 按 40 核计后，NC<40 的用例（m01 NC=16、m03 NC=3）欠载更明显，二级切分（§4.4）为主要性能补强点；表中"计算下界"列亦应按 40 核重估（不改变"搬运 bound"的定性结论）。

### 7.3 与 TBE 对齐的策略

- TBE 原实现同为搬运 bound，核心变量是**重复搬运系数**与**清零实现效率**；本设计行缓存将搬运系数压到 1（理论下界），清零分摊到全核，达成 ≥95% 的置信度高。
- 小 shape（< 100 µs 场景）：耗时由 kernel 启动/调度固定开销主导，Ascend C 与 TBE 开销结构相同；若个别用例超 30%，按任务书第 2 条提交 msprof/性能仿真对比图与分析结论。
- 验收前用 `msprof` 采集 153 用例双向耗时，输出对比表（交付件 3）。本自测环境芯片为 910B4-1（频率略低于 910B3），自测数据偏保守，报告中注明硬件型号与基准换算。

## 8. 精度设计

| 输出 | 机制 | 预期误差 |
|------|------|----------|
| out | 原 dtype（bf16 经 fp32 中转）精确比较与复制，无算术运算 ⇒ 无舍入 | 0（阈值 0.001） |
| indices | float 承载的展平索引（H·W ≤ 2^24 精确）确定性生成、Cast 回 int32，tie 语义与 golden 逐元素一致 | 完全一致（阈值 0） |

确定性保障：单输出点仅由单线程（向量 lane）顺序更新，无跨核归约、无原子操作，天然确定性（满足任务书约束 1）。输入不支持 NaN/-Inf（约束 2）：NaN 比较恒 false 会导致 argmax=0，与 golden 不一致，host 文档中声明该限制即可。

## 9. 风险与备选方案

| 风险 | 等级 | 应对 |
|------|------|------|
| **GM→UB DataCopyPad 尾部 32B 块污染**（blockLen 非 32B 整倍时，尾部块把 GM 后续数据——下一行行首——带入 UB pad 区；用例 F185 实测右 pad 列 argmax 指向越界行） | 已修复（v2 未预见） | LoadRow 后 `SetValue` -inf 重置 `[padLeft+W, alignUp)`（§5.4）；UB→GM 方向字节精确无此问题；bf16 Cast 入槽天然免疫 |
| **dav-c220 向量 tile>64 整核静默不写输出**（tileWo>64 时所有核不写任何输出、host 无报错、GM 保持初始值；手工分块全部向量 API 无效，根因未定位） | 已规避（v2 未预见） | host `tileWo` 硬上限 64（§4.5）+ 全向量 API 64 元素分块（§5.5）；根因定位后可放宽上限 |
| UB 内任意元素偏移向量访问 | 已规避 | 统一 `Gather` 按槽内字节偏移取值（任意 sW），规避非对齐向量读约束；`DeInterleave` 特化未采用 |
| DataCopyPad 非对齐写带宽略低于对齐写 | 低 | out/argmax 逐 tile 直写（≤256B 粒度，153 用例规模下非瓶颈）；清零段 32B 对齐取整 + 8KB 大粒度顺序写 |
| NC < 核数的小中 shape 欠载（AIV=40，如 m01 NC=16、m03 NC=3） | 中 | W_out 维二级切分（§4.4，未实施，性能调优首选） |
| 极端大 W 整行缓存固定项超 UB（H·W≤2^24 内 W 理论可达 2^24；kH·rowBufElems 超预算时 host 未显式拦截） | 低 | 已知限制；用例最大 W=224 远未触及，需要时加 host 校验拦截 |
| 超大 kH（如 >31）行缓存超预算 | 低 | host tiling 按 UB 预算收缩 tileWo；kH 使固定项超预算时同“极端大 W”限制 |
| 300V Pro（dav-m200）无 DataCopyPad、无 bf16 | 高 | **m200 路径本期未实现**（`isM200` 恒 0，OpDef 仅注册 ascend910b/ascend910_93）；后续按 v2 方案补对齐+读改写路径 |

## 10. 测试计划（对应交付件 2/3）

> **当前状态（v3 回写，2026-09）**：① 功能/精度已完成——自研 Python 编排（`tests/run_cases.py`，语义同 AscendOpTest：生成数据 → aclnn 两段式执行 → golden 比对）跑通随任务 153 用例，**153/153 通过**（fp32×49 / fp16×54 / bf16×50；out 误差恒 0、indices 完全一致，优于 0.001/0 阈值）；② 泛化自补用例、性能 msprof 对比 TBE（目标 ≥95%）未做；③ m200（300V Pro）路径未实现，300V Pro 复核不适用。

1. **功能/精度**：AscendOpTest 框架跑通随任务 153 用例（dtype × shape × (k,s,p) 全覆盖，含 3D 输入），比对 out（0.001）与 indices（0）。
2. **泛化自补**：kH≠kW、sH≠sW、pH≠pW、ceilMode=True、stride 缺省（s=k，含 sW≥3 触发 Gather 路径）、s>k（验证跳行逻辑）、大 W 触发 tile 切分、H·W 接近 2^24 边界等用例。
3. **性能**：800T A2（本机 910B4-1）上 msprof 采集全部用例耗时，与 TBE 基线对比（目标 ≥95%）；300V Pro 复核功能与精度，重点回归 m200 搬运替代路径（对齐读、Duplicate 填 pad、首尾读改写、清零首块合并写）。
4. **交付**：测试 readme（复现步骤）、自测报告（参数/精度截图/性能数据）。
