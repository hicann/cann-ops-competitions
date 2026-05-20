# CANN ops-math 总决赛算子测试报告

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 团队目录 | `AHNU_doushidui` |
| 团队名称 | `doushidui` |
| 所属单位 | 安徽师范大学 |
| 提交阶段 | 总决赛 |
| 测试对象 | Add、Cumsum |
| 运行环境 | 远程 Ascend 910_93 NPU 真机环境 |
| 报告类型 | 总决赛测试总报告 |

> 说明：本报告根据 Add 与 Cumsum 两份测试源码整理，用于说明测试设计思路、覆盖率统计方式、精度分析与问题发现情况。若后续在远程服务器实际运行得到完整 `gcov -b` 输出，可将“实际覆盖率统计”表中的待填写项替换为真实数值。

---

## 2. 赛题理解

总决赛要求参赛者为 CANN `ops-math` 仓库中的算子编写端到端测试用例，在真实 Ascend 910_93 NPU 环境下尽可能覆盖算子的执行路径，并对算子精度特性进行分析。

本次提交针对以下两个算子：

| 题目 | 算子 | 数学语义 | 测试重点 |
|---|---|---|---|
| 题目 1 | Add | `y = x1 + alpha * x2` | alpha 参数、V3 API、inplace、broadcast、dtype、bool guard、异常参数 |
| 题目 2 | Cumsum | `y[i] = sum(x[j])` | 标准 API、V2 API、`exclusive`、`reverse`、负维度、误差累积、非法参数 |

---

## 3. 测试文件说明

建议提交结构如下：

```text
AHNU_doushidui/
├── README.md
├── code/
│   ├── Add/
│   │   └── test_aclnn_add.cpp
│   └── Cumsum/
│       └── test_aclnn_cumsum.cpp
└── report/
    └── report.md
```

| 算子 | 测试文件 | 主要内容 |
|---|---|---|
| Add | `test_aclnn_add.cpp` | 覆盖 `aclnnAdd`、`aclnnAdds`、`aclnnInplaceAdd`、`aclnnInplaceAdds`、`aclnnAddV3`、`aclnnInplaceAddV3` 六类 API 变体 |
| Cumsum | `test_aclnn_cumsum.cpp` | 覆盖 `aclnnCumsum` 与 `aclnnCumsumV2`，包含 `exclusive` / `reverse` 参数组合、接口校验和精度测试 |

---

## 4. 实验环境

| 项目 | 内容 |
|---|---|
| 运行平台 | 远程 Ascend 910_93 NPU 服务器 |
| 工作目录 | `/root/ops-math` |
| SOC 参数 | `ascend910_93` |
| vendor name | `custom` |
| 覆盖率参数 | `--cov` |
| 覆盖率工具 | `gcov -b` |
| 是否使用模拟器 | 否 |
| 是否使用 Docker | 否 |

---

## 5. 总体测试设计思路

### 5.1 端到端执行链路覆盖

每个测试用例均按照 ACLNN 端到端调用流程组织：

1. 初始化 ACL 环境；
2. 构造 host 输入数据；
3. 分配 device 内存；
4. 创建 `aclTensor` 或 `aclScalar`；
5. 调用 `GetWorkspaceSize` 获取 workspace 与 executor；
6. 分配 workspace；
7. 调用对应 ACLNN 执行接口；
8. 同步 stream；
9. 将 device 输出拷贝回 host；
10. 在 host 侧计算期望结果；
11. 逐元素比较实际输出与期望输出；
12. 打印 PASS / FAIL，并统计测试结果。

### 5.2 覆盖率导向设计

测试不只验证常规功能，还主动覆盖以下路径：

- 不同 dtype 的类型检查与 cast 分支；
- Tensor-Tensor、Tensor-Scalar、Scalar-Tensor 分支；
- inplace 与非 inplace 分支；
- broadcast shape 推导分支；
- 空 tensor 与异常参数分支；
- Cumsum 的 `exclusive` 与 `reverse` 组合分支；
- Cumsum 不同 shape 对 tiling 策略的影响；
- 浮点、半精度、BF16 的数值容差与误差累积路径。

---

## 6. Add 算子测试设计

### 6.1 测试目标

Add 算子的数学语义为：

```text
out = self + alpha * other
```

测试目标包括：

1. 覆盖 6 类 API 变体；
2. 验证 alpha 参数为 1、0、负数、小数时的结果；
3. 验证 FP32、FP16、BF16、INT、BOOL、DOUBLE、COMPLEX 等不同 dtype 组合；
4. 验证 broadcast、非连续 tensor、空 tensor 等 shape 路径；
5. 验证 bool 特殊 guard、mixed dtype 和输出 dtype cast 行为；
6. 验证 nullptr、shape mismatch、rank 超限、不支持类型转换等异常路径。

### 6.2 API 覆盖情况

| API 类型 | 接口 | 覆盖方式 |
|---|---|---|
| Tensor + Tensor | `aclnnAdd` | 普通相同 shape、broadcast、mixed dtype、complex workspace-only |
| Tensor + Scalar | `aclnnAdds` | scalar 输入、bool guard、FP16/BF16 promote |
| Inplace Tensor + Tensor | `aclnnInplaceAdd` | self 原地更新、nullptr 参数校验 |
| Inplace Tensor + Scalar | `aclnnInplaceAdds` | scalar 原地更新、异常路径 |
| Scalar + Tensor V3 | `aclnnAddV3` | scalar self + tensor other、rank 边界、bool 不支持路径 |
| Inplace V3 | `aclnnInplaceAddV3` | tensor 原地更新、alpha/cast 分支 |

### 6.3 主要测试用例

| 类别 | 用例示例 | 设计目的 |
|---|---|---|
| 基础功能 | `Add_FP32`、`Add_INT32` | 验证基础逐元素加法 |
| alpha 参数 | `alpha = 1.0`、`alpha = 0`、`alpha = -1`、`alpha = 1.5` | 覆盖 alpha 特殊值和小数分支 |
| 半精度 | FP16、BF16 | 验证半精度舍入误差和容差设置 |
| 混合精度 | BF16 promote 到 FP32、FP16 + FP32 | 覆盖类型提升与输出 dtype |
| bool 特殊路径 | BOOL tensor + BOOL scalar + BOOL alpha | 覆盖 bool guard，避免将逻辑加法误判为普通数值加法 |
| broadcast | `[2, 3] + [1]`、`[2, 1] + [1, 3]` | 验证广播索引映射 |
| 非连续布局 | 自定义 stride 与 storage shape | 验证逻辑索引与物理存储映射 |
| 空 tensor | shape 含 0 | 验证空输入路径 |
| V3 API | `AddV3`、`InplaceAddV3` | 覆盖 scalar-tensor 和 inplace V3 |
| 异常路径 | nullptr、shape mismatch、rank > 8、非法 cast | 验证接口校验分支和错误返回 |

### 6.4 Add 期望值计算与结果验证

Add 测试在 host 侧构造期望值：

```text
expected = lhs + alpha * rhs
```

对于不同 API：

- `aclnnAdd` / `aclnnInplaceAdd`：`lhs` 与 `rhs` 均来自 tensor；
- `aclnnAdds` / `aclnnInplaceAdds`：`rhs` 来自 scalar；
- `aclnnAddV3` / `aclnnInplaceAddV3`：`lhs` 来自 scalar，`rhs` 来自 tensor。

对于 broadcast 场景，测试根据输出坐标反推输入逻辑索引，避免简单按线性下标比较导致误判。对于 FP16 / BF16 / FP32 设置不同容差；对于整数与 bool 类型采用语义化比较。

### 6.5 Add 精度分析

Add 的主要精度风险来自以下方面：

1. **alpha 放大误差**  
   当 `alpha` 为小数或较大数时，`alpha * other` 会先产生乘法舍入误差，再与 `self` 相加。FP16 / BF16 场景下更明显。

2. **半精度舍入误差**  
   FP16 有效尾数较短，BF16 尾数更短但指数范围较大。测试中应对 FP16 / BF16 使用相对误差和绝对误差联合判断。

3. **混合精度类型提升**  
   当输入 dtype 与输出 dtype 不一致时，实际计算路径可能涉及 cast 或 promote。测试应验证输出 dtype 下的最终数值，而不是仅按输入 dtype 判断。

4. **bool 特殊语义**  
   BOOL 类型不应简单套用浮点加法容差，测试中应按逻辑值语义比较。

5. **inplace 更新风险**  
   inplace 接口直接覆盖输入 tensor，若底层实现存在读写顺序问题，可能造成后续元素计算受已写结果影响。测试通过完整输出比对检查该风险。

---

## 7. Cumsum 算子测试设计

### 7.1 测试目标

Cumsum 的数学语义为沿指定维度进行累积求和：

```text
out[i] = sum(input[0], input[1], ..., input[i])
```

当启用 V2 参数时：

- `exclusive = true`：当前位置不包含自身；
- `reverse = true`：沿指定维度反向累积。

测试目标包括：

1. 覆盖标准 `aclnnCumsum` API；
2. 覆盖 `aclnnCumsumV2` API；
3. 覆盖 `exclusive` / `reverse` 四种组合；
4. 覆盖正维度和负维度；
5. 覆盖空 tensor；
6. 覆盖多种 shape，尽量触发不同 tiling 路径；
7. 覆盖整型、浮点、半精度和 BF16 的结果校验；
8. 分析长序列累加带来的误差累积。

### 7.2 API 覆盖情况

| API | 参数 | 覆盖重点 |
|---|---|---|
| `aclnnCumsum` | `self`、`dim`、`dtype`、`out` | 标准累积求和、dtype 参数、shape 与 out 校验 |
| `aclnnCumsumV2` | `self`、`dim`、`exclusive`、`reverse`、`out` | V2 参数组合、同 dtype 要求、反向与排除自身逻辑 |

### 7.3 主要测试用例

| 类别 | 用例示例 | 设计目的 |
|---|---|---|
| 基础功能 | `float32_basic_dim0` | 验证 dim=0 的标准累加 |
| 空 tensor | `float32_empty_tensor` | 验证 shape 含 0 时不崩溃 |
| 负维度 | `float32_negative_dim_mixed_sign` | 验证负 dim 归一化 |
| 小 shape | `float32_mrnlessercl_case` | 覆盖小规模 tiling 路径 |
| 大 shape | `float32_mrngreatercl_case`、`float32_rngreatercl_*` | 覆盖大规模 tiling 路径 |
| V2 exclusive | `exclusive = true` | 验证当前位置不包含自身 |
| V2 reverse | `reverse = true` | 验证反向累加 |
| V2 组合 | `exclusive = true, reverse = true` | 验证反向排除自身 |
| 整型精确比较 | INT32 / INT64 | 验证整数累加精确结果 |
| 浮点容差比较 | FP32 / FP16 / BF16 | 验证浮点误差容忍 |
| 接口异常 | nullptr、dim 越界、out shape mismatch、dtype mismatch | 覆盖参数校验分支 |

### 7.4 Cumsum 期望值计算与结果验证

Cumsum 测试使用 host 侧 CPU 参考实现计算期望值。参考实现将 `dim` 归一化后，按 `outer / axis / inner` 三层逻辑遍历：

```text
for outer:
  for inner:
    sum = 0
    for axis:
      if exclusive:
        out = sum
        sum += input
      else:
        sum += input
        out = sum
```

当 `reverse = true` 时，axis 维度从后向前遍历。期望值使用 `long double` 作为中间累加类型，以降低 CPU 参考结果自身的舍入误差。比较时根据 dtype 设置 exact 或 tolerance 判断。

### 7.5 Cumsum 精度分析

Cumsum 是总决赛中精度分析的重点，因为累积求和会随着 axis 长度增加不断放大舍入误差。

#### 7.5.1 误差来源

1. **顺序累加误差**  
   浮点加法不满足严格结合律，长序列累加时误差会逐步累积。

2. **正负抵消误差**  
   当输入中存在正负交替数据时，累加结果可能出现消去现象，导致有效数字损失。

3. **半精度累加误差**  
   FP16 尾数较短，连续累加小数时更容易丢失低位信息。BF16 指数范围大但尾数短，适合大范围数值但细粒度小数累加误差更明显。

4. **reverse 改变累加顺序**  
   `reverse = true` 会改变累加方向。由于浮点加法对顺序敏感，正向与反向的误差分布可能不同。

5. **exclusive 改变输出时刻**  
   `exclusive = true` 并不改变累计过程本身，但每个位置输出的是加入当前元素之前的状态，因此可暴露边界位置和初始值处理错误。

#### 7.5.2 测试策略

- 对整数类型使用精确比较；
- 对 FP32 使用较严格的 `1e-5` 级别容差；
- 对 FP16 / BF16 使用更宽容差；
- 对长 axis 输入记录最大误差与最大误差所在位置；
- 使用正负混合、小数步进、大 shape 等数据模式观察误差累积。

---

## 8. 覆盖率统计

### 8.1 覆盖率统计方法

编译时开启覆盖率插桩：

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=<op> --vendor_name=custom --cov
```

运行测试：

```bash
bash build.sh --run_example <op> eager cust --vendor_name=custom --soc=ascend910_93 --cov
```

查看覆盖率：

```bash
find build -name "*.gcda" | grep <op>
gcov -b <gcda文件路径>
```

重点记录：

```text
Lines executed: XX.XX% of YY
Branches executed: XX.XX% of YY
```

### 8.2 实际覆盖率统计表

| 算子 | 文件/模块 | 行覆盖率 | 分支覆盖率 | 备注 |
|---|---|---:|---:|---|
| Add | op_api 层 | 待运行后填写 | 待运行后填写 | 重点关注 6 类 API 变体 |
| Add | op_host / tiling 层 | 待运行后填写 | 待运行后填写 | 编译后需确认 `add_tiling*.gcno` 存在 |
| Cumsum | op_api 层 | 待运行后填写 | 待运行后填写 | 重点关注 Cumsum / CumsumV2 |
| Cumsum | op_host / tiling 层 | 待运行后填写 | 待运行后填写 | 编译后需确认 `cumsum_tiling*.gcno` 存在 |

### 8.3 功能场景覆盖统计

| 算子 | 主要覆盖维度 | 覆盖说明 |
|---|---|---|
| Add | API 变体 | 覆盖 Add、Adds、InplaceAdd、InplaceAdds、AddV3、InplaceAddV3 |
| Add | dtype | 覆盖 float、half、BF16、整数、bool、double、complex 等路径 |
| Add | shape | 覆盖普通 shape、broadcast、非连续、空 tensor |
| Add | 异常路径 | 覆盖 nullptr、shape mismatch、非法 cast、rank 超限等 |
| Cumsum | API 变体 | 覆盖标准 API 与 V2 API |
| Cumsum | V2 参数 | 覆盖 exclusive / reverse 组合 |
| Cumsum | dim | 覆盖正 dim、负 dim、dim 越界 |
| Cumsum | shape | 覆盖小 shape、空 tensor、大 shape、长 axis |
| Cumsum | 精度 | 覆盖正负混合、长序列误差累积、半精度误差 |

---

## 9. 问题发现与风险分析

### 9.1 Add

| 风险点 | 说明 | 测试处理 |
|---|---|---|
| alpha 小数用于整数输出 | 可能出现非法 cast 或不支持路径 | 设计 negative case，期望返回失败 |
| bool + scalar + alpha | bool guard 与普通数值加法语义不同 | 单独设计 bool guard 用例 |
| mixed dtype | 可能触发 promote / cast 分支 | 设计 BF16/FP16 到 FP32 输出场景 |
| rank 超限 | 高 rank 可能被接口拒绝 | 设计 rank > 8 negative case |
| 非连续 tensor | stride 与 storage offset 易出现索引错误 | 构造自定义 storageShape / strides |
| V3 API | scalar-tensor 路径与普通 Add 不同 | 单独覆盖 AddV3 与 InplaceAddV3 |

### 9.2 Cumsum

| 风险点 | 说明 | 测试处理 |
|---|---|---|
| dim 越界 | 接口应返回错误而不是执行 | 设计 dim out of range negative case |
| out shape 不一致 | 输出 shape 应与输入一致 | 设计 shape mismatch negative case |
| V2 dtype 不一致 | V2 self/out dtype 应一致 | 设计 dtype mismatch negative case |
| reverse/exclusive 边界 | 头尾位置容易出现 off-by-one 错误 | 覆盖四种参数组合 |
| 长 axis 误差累积 | 浮点累加误差随长度增加 | 使用大 shape 和正负混合数据 |
| 空 tensor | 容易触发异常内存访问 | 设计空 tensor 用例 |

---

## 10. 总结

本次总决赛提交围绕 Add 与 Cumsum 两个进阶算子设计端到端测试用例。Add 测试重点覆盖 6 类 API 变体、alpha 参数、多 dtype、broadcast、inplace、V3、bool guard 和异常路径；Cumsum 测试重点覆盖标准 API、V2 API、`exclusive` / `reverse` 参数组合、负维度、大 shape、空 tensor、接口校验和累积误差分析。

测试代码均包含 host 侧期望值计算与实际输出比对逻辑，可用于验证算子功能正确性，并通过 `--cov` 与 `gcov -b` 生成行覆盖率和分支覆盖率。后续在远程 Ascend 910_93 真机环境运行后，应将实际 `gcov` 输出补充到报告的覆盖率统计表中，并保留 `.gcda` / `.gcno` 文件用于最终评分。
