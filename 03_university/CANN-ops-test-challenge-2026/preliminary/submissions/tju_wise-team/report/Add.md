# 测试报告

## 一、测试策略
### 1.1 总体测试覆盖维度设计

| 维度 | 覆盖内容 |
| ---- | -------- |
| 数据类型 | FLOAT32 / FLOAT16 / BF16 / INT64 / INT32 / INT16 / INT8 / UINT8 / BOOL / COMPLEX64 |
| 计算模式 | TensorTensor / TensorScalar / ScalarTensor |
| API 类型 | 普通接口 / Inplace 接口 |
| shape | 同 shape / 广播 / 非法广播 / 空 Tensor |
| 数值范围 | 正常值 / 负值 / 零值 / 小数值 / 近边界整数值 |
| 特殊路径 | dtype 提升 / cast 失败 / 非 ND format / fallback 路径 |
| 异常输入 | nullptr / 非法 shape / 非法 out shape / 不支持 dtype |

---

### 1.2 覆盖策略说明

**1. API 覆盖策略**
- 覆盖全部API aclnnAdd、aclnnAdds、aclnnAddV3、 aclnnInplaceAdd、aclnnInplaceAdds、aclnnInplaceAddV3

**2. 数据类型覆盖策略**
- 常规成功路径覆盖 FLOAT32 / FLOAT16 / BF16 / INT32 / INT64 / UINT8 / INT8
- 通过 mixed dtype 用例覆盖类型提升与不同执行分支
- 通过 BOOL、输出类型不匹配、alpha 不可转换等场景覆盖错误分支

**3. 分支覆盖策略**
- 针对 `alpha == 1`、`alpha != 1`、Axpy、fallback 等执行路径设计专门用例
- 针对 `CheckPromoteType`、`CheckShape`、`CheckNotNull`、empty tensor、non-ND format 等逻辑设计失败探针
- 针对 `AddV3` 的 scalar promote、complex self、floating other 等条件组合补充专门用例
- 针对 `add_tiling_arch35.cpp` 的 dtype 分发与 `CheckDtype` 失败路径补充对应数据组合

**4. 边界值策略**
- 增加少量负值、零值、小量浮点值、接近 `INT32` 边界的测试

## 二、测试用例设计
### 2.1 测试文件

- 统一测试文件：`examples/test_aclnn_add.cpp`
- 当前共整理并执行 `70` 个 `RecordCase`

### 2.2 用例分组

#### 用例 1-23：Add / Adds 常规与异常覆盖
- **覆盖函数**：`aclnnAdd`、`aclnnAdds`
- **输入**：FLOAT/INT/BF16/F16/BOOL 组合、广播 shape、非法 out shape、空 Tensor、非 ND format
- **设计目的**：
  - 覆盖普通 TensorTensor / TensorScalar 主路径
  - 覆盖 dtype promote、empty tensor、broadcast fail、null 参数、format warning
  - 补充 mixed dtype 的多个方向，如 `FLOAT16 + FLOAT`、`FLOAT + BF16`、`BF16 + FLOAT`、`FLOAT + FLOAT16`

#### 用例 24-31：Inplace Add / Inplace Adds 覆盖
- **覆盖函数**：`aclnnInplaceAdd`、`aclnnInplaceAdds`
- **输入**：FLOAT、DOUBLE、INT8、mixed dtype、非法广播 shape
- **设计目的**：
  - 覆盖 inplace 正常执行路径
  - 覆盖 `broadcastShape != otherShape`
  - 覆盖 mixed dtype 下不支持 inplace from other 的失败分支
  - 覆盖 `AddInplace` 在 `op_api/add.cpp` 中原先几乎未执行的判断分支

#### 用例 32-52：AddV3 主路径与失败路径覆盖
- **覆盖函数**：`aclnnAddV3`
- **输入**：scalar self + tensor other，多种 scalar dtype、other dtype、alpha dtype、输出 dtype
- **设计目的**：
  - 覆盖 `alpha == 1`、`Axpy`、fallback 三条执行路径
  - 覆盖 `self = COMPLEX64`、`self = DOUBLE && out = FLOAT`、`self = FLOAT && other = INT32`
  - 覆盖 `other = FLOAT16 / BF16` 的 floating 分支
  - 覆盖 `alpha cast fail`、`out cast fail`、`other bool not support`、invalid shape、null 参数

#### 用例 53-70：覆盖率补充与边界值补充
- **覆盖内容**：负值、零值、小量浮点、接近 `INT32` 边界值
- **设计目的**：
  - 补充基础数值健壮性
  - 适度覆盖不同输入组合下的稳定性

## 三、测试结果
### 3.1 总体覆盖率

本次测试以 `coverage_html6/coverage_html/index.html` 为准，整体覆盖率如下：

| 指标 | 已覆盖 | 总数 | 覆盖率 |
| ---- | ------ | ---- | ------ |
| Lines | 440 | 529 | **83.2%** |
| Functions | 53 | 55 | **96.4%** |
| Branches | 730 | 2462 | **29.7%** |

### 3.2 分层覆盖率

| Directory | Line Coverage | Functions | Branches |
| --------- | ------------- | --------- | -------- |
| `op_api` | 358 / 439 = **81.5%** | 41 / 43 = **95.3%** | 667 / 2304 = **28.9%** |
| `op_host/arch35` | 82 / 90 = **91.1%** | 12 / 12 = **100.0%** | 63 / 158 = **39.9%** |

### 3.3 关键文件覆盖率

| 文件 | 行覆盖率 | 函数覆盖率 | 分支覆盖率 |
| ---- | -------- | ---------- | ---------- |
| `op_api/aclnn_add.cpp` | 251 / 303 = **82.8%** | 25 / 26 = **96.2%** | 453 / 1594 = **28.4%** |
| `op_api/aclnn_add_v3.cpp` | 74 / 77 = **96.1%** | 9 / 9 = **100.0%** | 164 / 446 = **36.8%** |
| `op_api/add.cpp` | 33 / 59 = **55.9%** | 7 / 8 = **87.5%** | 50 / 264 = **18.9%** |
| `op_host/arch35/add_tiling_arch35.cpp` | 82 / 90 = **91.1%** | 12 / 12 = **100.0%** | 63 / 158 = **39.9%** |

---

## 四、结果分析
### 4.1 主要成果

1. `aclnn_add_v3.cpp` 行覆盖率已提升到 **96.1%**，核心执行路径和大部分可达失败路径已覆盖。
2. `add_tiling_arch35.cpp` 行覆盖率达到 **91.1%**，dtype 分发和 `CheckDtype` 失败路径已有明显补充。
3. `aclnn_add.cpp` 已覆盖普通计算、empty tensor、null 参数、broadcast、promote、format warning、Axpy/fallback 等主要路径。

### 4.2 重点补充的覆盖点

- `Add`：
  - mixed dtype 多方向分支
  - invalid out shape
  - null 参数
  - non-ND format
  - 负值 / 零值 / 小数值 / 近边界整型值

- `InplaceAdd`：
  - broadcast fail
  - `broadcastShape != otherShape`
  - mixed float16 / bf16 非法 inplace
  - 非浮点整型成功路径

- `AddV3`：
  - `alpha != 1` 下 Axpy 与 fallback
  - scalar self 为 `COMPLEX64`、`DOUBLE`
  - other 为 `FLOAT16 / BF16`
  - `alpha cast fail` / `out cast fail`
  - null 参数 / invalid shape / empty tensor

- `add_tiling_arch35.cpp`：
  - `INT32 / INT64 / UINT8` dtype 路由
  - mixed dtype 输出非法
  - output dtype 不匹配
  - unsupported dtype

---

## 五、未覆盖分析
### 5.1 主要未覆盖代码类型

| 类别 | 说明 |
| ---- | ---- |
| 平台相关分支 | `SoC / arch / IsRegBase` 相关逻辑 |
| 防御性错误分支 | launcher 失败、内部资源异常等 |
| 宏/日志分支 | `OP_CHECK`、`OP_LOG` 展开的部分分支 |
| AddInplace 部分路径 | 无 |

### 5.2 未完全覆盖原因分析

- `op_api/add.cpp`
  - 原因是部分 `AiCore/AiCpu` 选择、`AddInplace` 内部条件、以及平台相关判断仍较难通过黑盒完全压满
  
- `op_api/aclnn_add.cpp`
  - 行覆盖已经较高，但分支数较大，存在较多平台相关或防御性分支
  - 继续提升需要结合平台特定场景

---

## 六、总结

本次测试围绕 `Add` 算子族进行了覆盖率导向补充，统一在 `test_aclnn_add_minimal_generic_runners_all_cases_recordcase.cpp` 中组织测试，共覆盖 `Add / Adds / AddV3 / InplaceAdd / InplaceAdds / InplaceAddV3` 主要接口，并补充了 `op_host/arch35` 的相关执行路径。

从结果看，整体覆盖率达到：

- 行覆盖率：**83.2%**
- 函数覆盖率：**96.4%**
- 分支覆盖率：**29.7%**

其中：
- `aclnn_add_v3.cpp` 与 `add_tiling_arch35.cpp` 覆盖效果较好；
- `aclnn_add.cpp` 主要功能路径已基本覆盖；
- `add.cpp` 仍是后续提升分支覆盖率的重点目标文件。
