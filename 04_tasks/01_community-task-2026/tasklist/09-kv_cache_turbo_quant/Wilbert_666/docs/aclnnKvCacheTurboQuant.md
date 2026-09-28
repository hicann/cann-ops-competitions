# aclnnKvCacheTurboQuant

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：不支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：√
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：√
- <term>Atlas 200I/500 A2 推理产品</term>：×
- <term>Atlas 推理系列产品</term>：×
- <term>Atlas 训练系列产品</term>：×

## 功能说明

- 接口功能：在线编码一条标准 MHA/GQA KV cache。输出主量化码字、QJL 残差符号、输入 L2 范数和原尺度残差范数。

- 计算逻辑：

  ```text
  x_norm = ||x||_2
  u      = x / x_norm
  y      = H @ u
  idx    = LloydMax(y, mse_bits)
  r      = y - dequant(idx)
  gamma  = x_norm * ||r||_2
  qjl    = sign(S @ r / ||r||_2)
  ```

  `H @ u` 与 `S @ r_unit` 都按右乘转置实现，即 `unit @ H.T`、`residual_unit @ S.T`。码字比较使用严格大于中点边界。打包时每 8 个通道组成一个小端字节组，通道 0 位于最低位。

- 存储：`mse_bits = 3` 时每 head 为 48 + 16 + 2 + 2 = 68 字节。

## 函数原型

可执行实现位于 `examples/test_aclnn_kv_cache_turbo_quant.cpp`。它在调用方 stream 上按顺序启动：

```cpp
kvtq_cast_fp32(vecCores, nullptr, stream, kv, fpKv, rows, vecCores);
aclnnLinalgVectorNorm(...);  // x_norm
aclnnDiv(...);                // u = x / x_norm
mm_fp32_transb(cubeCores, nullptr, stream, unit, rotation, rotated, workspace, tiling);
kvtq_quantize(vecCores, nullptr, stream, rotated, quantIdx, residual, rows, mseBits, vecCores);
aclnnLinalgVectorNorm(...);  // ||r||
aclnnDiv(...);                // r / ||r||
aclnnMul(...);                // gamma = x_norm * ||r||
kvtq_store_bf16(vecCores, nullptr, stream, xNorm, gammaFp, quantNorm, quantGamma, rows, vecCores);
mm_fp32_transb(cubeCores, nullptr, stream, residualUnit, qjlMatrix, projected, workspace, tiling);
kvtq_sign(vecCores, nullptr, stream, projected, quantQjl, rows, vecCores);
```

行范数和除法走 `aclnnLinalgVectorNorm` / `aclnnDiv`，与 `torch.linalg.vector_norm` 同一实现，保证和 golden 逐位一致。`mm_fp32_transb` 是 AIC-only FP32 GEMM，`TransB = true`，每个 Cube 核只计算自己的 M 行。`kvtq_cast_fp32`、`kvtq_quantize`、`kvtq_store_bf16`、`kvtq_sign` 是 AIV kernel。Vector 核数保证每核至少 8 行；不足 8 行时退化为单核。BF16 写回使用标量 ties-to-even，与 numpy bfloat16 一致。

Host 原型与算子定义一致：

| 参数 | 方向 | 类型 | shape |
| --- | --- | --- | --- |
| kv_vectors | 输入 | BF16 | `[num_tokens, num_kv_heads, 128]` |
| rotation_matrix | 输入 | FLOAT32 | `[128, 128]` |
| qjl_matrix | 输入 | FLOAT32 | `[128, 128]` |
| mse_bits | 属性 | INT | 2、3 或 4，默认 3 |
| quant_idx | 输出 | UINT8 | `[num_tokens, num_kv_heads, 16 * mse_bits]` |
| quant_qjl | 输出 | UINT8 | `[num_tokens, num_kv_heads, 16]` |
| quant_norm | 输出 | BF16 | `[num_tokens, num_kv_heads]` |
| quant_gamma | 输出 | BF16 | `[num_tokens, num_kv_heads]` |

workspace 至少包含 4 份 `rows * 128` 的 FP32 中间张量（单位向量、旋转结果、残差、投影）、三份 FP32 行向量（`x_norm`、`||r||`、`gamma`），以及 32MB Cube workspace。单位向量缓冲在第二段 GEMM 前被残差单位向量覆盖。`op_host/kv_cache_turbo_quant_tiling.cpp` 按这个公式回填 workspace。

## 返回值

直调样例在精度或性能不满足任务书时返回非 0。入参不满足 `head_dim = 128`、`num_kv_heads ∈ [4, 32]`、`mse_bits ∈ {2, 3, 4}` 时，tiling 返回 `GRAPH_FAILED`。

## 约束

- 不支持 MLA。
- 不支持空 tensor。
- 旋转矩阵与投影矩阵在推理期间固定，本接口不负责生成它们。
- 重构不在编码接口内。精度报告里的反量化使用旋转域主码字，加上按 `||S^T signs||` 归一化的 QJL 残差方向，再左乘 `x_norm * H^T`。

## 调用示例

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash examples/build.sh
```
