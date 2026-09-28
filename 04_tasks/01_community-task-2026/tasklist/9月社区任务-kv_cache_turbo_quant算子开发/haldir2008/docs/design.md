# KvCacheTurboQuant（Encode）算子设计文档

> 任务：[kv_cache_turbo_quant](https://www.hiascend.com/activities/task-center/details/6d3df013343845608677c85f3c58d4d6)  
> 目标仓：`ops-transformer/experimental/attention` · 登记芯片：**Atlas 800T A2（ascend910b）**；本机自证：**A3（ascend910_93）+ CANN 9.0**  
> 状态：**设计评审稿**（与本机 NPU 实现路径对齐）。本机 A3 上精度门 + 五档性能 ≥10× 已自证；**本 PR/本文不等于官方最终验收通过**。

---

## 1. 需求背景

对标准 MHA/GQA 的 K/V 向量做在线 TurboQuant 编码：正交旋转 + MSE 主量化 + QJL 残差 1-bit 符号。首版不含 MLA、混合 3.5-bit、独立 decompress、Decode Attention。

`head_dim=128`、`mse_bits=3` 时物理约 **4.25 bit/channel**，相对 BF16 约 **3.76×** 编码压缩（任务书 §2 存储表口径；不以广场「3.5 bit / 4×」宣传文案覆盖）。

## 2. 接口与约束

| 参数 | 方向 | dtype | shape |
|------|------|-------|-------|
| kv_vectors | 入 | BF16 | [T, H, 128] |
| rotation_matrix | 入 | FP32 | [128, 128] |
| qjl_matrix | 入 | FP32 | [128, 128] |
| quant_idx | 出 | UINT8 | [T, H, ceil(128·mse_bits/8)] |
| quant_qjl | 出 | UINT8 | [T, H, 16] |
| quant_norm / quant_gamma | 出 | BF16 | [T, H] |

- `mse_bits ∈ {2,3,4}`（默认 3）+ 1-bit QJL；aclnn 两段式；不 Host 解引用 Device。  
- 适用范围：标准 MHA/GQA；**不支持 MLA**。

## 3. 数学语义（与 golden / encode_one_core 一致）

```text
n = ||x||_2
u = x / max(n, 1e-30)          # n=0 → 0
y = u @ H.T                    # 正交时等价 Hx（Cube SetTensorB(trans=true)）
idx = Lloyd-Max 中点阈值桶（二分选桶 ≡ 线性阈值链）
ŷ = centroids[idx]
r = y - ŷ;  γ = n · ||r||_2
ru = r / max(||r||_2, 1e-30)
qjl_bit = (ru @ S.T >= 0)      # 1-bit pack，LSB-first
```

质心表、PackBits 位序与任务书 `golden.py` / 仓内 `encode_one_core.h` 共用。

## 4. Host / Tiling

- `SetBlockDim(aicNum)`；逻辑 AIV = `aicNum×2`；`coreRows = ceil(N / vecCoreNum)`，`N=T·H`。  
- Cube tiling：`GemmTile=128` 行切片；Matmul 系统 workspace 预留 32MB + `2·N·128·sizeof(float)`（units / matOut）。  
- 属性校验：`mse_bits`、shape、dtype；非法返回参数错误。

## 5. Kernel 数据流（MIX AIC:AIV=1:2）

分阶段 + Prefetch 后 `SyncAll`：

1. **Prefetch**：BF16 KV → FP32 → `unitsGm`（tile=`kPrefetchTile=64`）  
2. **Rotate**：Cube `units @ H.T` → `matOutGm`  
3. **QuantizeAndResUnit**（热路径，`kQuantBatch=32` 多行一批）：  
   - 批量加载旋转结果；一次 Mul 全 batch 平方 + 按行 ReduceSum/Sqrt 得 norm，单位化  
   - **中点二分 Compare/Select**（nCen=8/4 展开；nCen=16 少轮 Gather 二分）  
   - `Cast→int32` → PackBits（bit2/3/4 特化）合并 DMA 写 `quant_idx`  
   - **Gather** 查质心得 primary；残差一次 Mul 平方后按行范数；单位化合并写回 `unitsGm`  
   - tile=64 批量 Cast 写 BF16 norm/gamma  
4. **Project**：Cube `units @ S.T`  
5. **PackQjl**：批量 `Compare(GE,0)`；1-bit mask 与 PackBits 同构，直写 GM  

已知债（设计层标明，禁止误抄）：WholeReduceSum 批量 norm、Gather×8 Pack、HW int4 Pack、TQue+Local DataCopy 预取 Quant——均已试并回退。

## 6. UB / Workspace 预算（量级）

- UB：batch×128 行缓冲（rotated/thr/idxf/tmp/primary 等）+ mask×2 + Pack 输出 + Gather 偏移 + tile64 scal + 绿基线占位缓冲。  
- Workspace：Matmul 系统区 + `2·N·D·4` 字节用户区。

## 7. 精度 / 性能标准（任务书不放宽）

| 项 | 标准 |
|----|------|
| 相对 MSE | `< 0.05`（低 bit 可 WARN，以 IP95 硬门槛为准） |
| IP 相对误差 95th | `< 0.1` |
| 性能 | 相对任务书 PyTorch eager baseline **≥10×**（warmup + SynchronizeStream） |

### 本机 A3 自证摘要（2026-09-24）

| case | relMSE | IP95 | ker_x（vs 任务书 baseline） |
|------|--------|------|------------------------------|
| gqa_decode_b1 | 0.0360 PASS | 0.0385 PASS | ~31× |
| gqa_decode_b64 | 0.0342 PASS | 0.0421 PASS | ~26× |
| gqa_prefill_t2048 | 0.0340 PASS | 0.0577 PASS | ~11.2× |
| gqa_mse2_t2048 | 0.1161 WARN | 0.0851 PASS | ~10.8× |
| gqa_mse4_t2048 | 0.0093 PASS | 0.0227 PASS | ~14.5× |

证据日志（可复制）：`docs/overnight_runs/auto_logs/gate_20260924_094122.log`、`docs/overnight_runs/perf_all10x_lock.log`。

## 8. 验证计划

1. CPU 孪生 `encode_one_core` ↔ golden 字节对拍  
2. NPU：五档 case decode exact + §3.2 矩阵 + `npu_perf_bench`  
3. poison→调用→sync→读回；多核分片  
4. 交付：`task_submission` 四件套 + 个人仓邀 `Ascend-CANN`；验收登记芯片 A2 若要求则补复测  

## 9. 支持硬件

| 芯片 | 状态 |
|------|------|
| Ascend 910_93（A3） | 本机功能+精度+性能自证（CANN 9.0） |
| Atlas 800T A2（ascend910b） | 任务书验收登记；待有机复测 |

## 10. 待确认（不自行放宽门槛）

1. 验收是否强制 A2+A3 双平台证据  
2. mse2 `relMSE≈0.116` 与 `<0.05` 字面冲突时，是否以 IP95 为主（与 twin 一致；本机已按此过门）  
3. 官方 rotation/qjl 矩阵是否提供固定种子文件  
4. 最终目录名与 aclnn 符号是否需与讨论区冻结名完全一致  
