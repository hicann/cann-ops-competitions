## 个人信息

- 姓名：刘天羽
- 学号：25303050126
- 联系邮箱：lewis2007s@163.com
- CANNJudge 账号：Kilo

## CANNJudge 提交说明

- 比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
- 最终提交时间：2026 年 6 月 5 日
- 两道题完成情况：
  - **ClipByValue**：已通过（5 个测试点全部 Pass）
  - **Lerp**：已实现并通过测试
  - **Addcmul**：因编译问题尚未完成，未包含在本次提交中

---

## 算子实现简介

### 一、ClipByValue 算子

#### 1.1 算子功能

基于 PyTorch `torch.clamp` 语义，将输入张量 `x` 的每个元素裁剪到 `[min, max]` 区间：

```
y[i] = clamp(x[i], min, max) = min(max(x[i], min), max)
```

- 若 `x[i] < min`，输出 `min`
- 若 `x[i] > max`，输出 `max`
- 否则保持 `x[i]` 原值

#### 1.2 技术方案

**Host 侧设计：**

- **TilingFunc**：根据 UB 大小、数据类型和向量指令限制，动态计算 `tileLength`、`perCoreElements` 和 `blockDim`
- **多核切分**：按 cache line（64B）对齐分配各 Core 的计算任务，充分利用 AI Core 并行算力
- **UB 内存管理**：根据 Queue 数量和缓冲份数计算 tile 上限，预留 8KB 指令空间

**Kernel 侧设计：**

- **Double Buffer 流水线**：CopyIn（GM→UB）→ Compute（UB 向量计算）→ CopyOut（UB→GM），隐藏数据传输延迟
- **向量指令**：`Maxs(x, min)` 取上界 → `PipeBarrier<PIPE_V>()` → `Mins(temp, max)` 取下界，两步完成裁剪
- **标量尾块**：对无法 32B 对齐的剩余元素，使用 `GetValue/SetValue` 逐元素处理，half 类型转 float 以保证比较精度
- **缓存刷新**：`DataCacheCleanAndInvalid<..., CACHELINE_OUT>(yGm)` 确保结果对 CPU 可见

#### 1.3 数据类型支持

通过 TilingKey 模板机制支持三种数据类型：

| 类型 | 说明 |
|------|------|
| `float16` | 半精度浮点，2 字节 |
| `float32` | 单精度浮点，4 字节 |
| `int32` | 32 位有符号整数 |

#### 1.4 测试结果（CANNJudge）

| 测试点 | 结果 | 输出错误占比 | 用时 | 最优用时 |
|--------|------|-------------|------|----------|
| 1 | Pass | 0.00% | 3.74μs | 3.00μs |
| 2 | Pass | 0.00% | 14.86μs | 13.78μs |
| 3 | Pass（修复后） | 0.00% | 7.08μs | 6.02μs |
| 4 | Pass | 0.00% | 3.20μs | 2.30μs |
| 5 | Pass | 0.00% | 9.54μs | 8.30μs |

#### 1.5 精度说明

- **float32**：相对误差 < 1e-4
- **float16**：相对误差 < 1e-3
- **int32**：精确匹配，无误差

---

### 二、Lerp 算子

#### 2.1 算子功能

基于 PyTorch `torch.lerp` 语义，对两个形状相同的张量 `start` 和 `end`，按标量权重 `weight` 逐元素计算线性插值：

```
y[i] = start[i] + weight × (end[i] - start[i])
```

- `weight = 0` 时输出 `start`
- `weight = 1` 时输出 `end`
- `weight = 0.5` 时输出两者的均值

#### 2.2 技术方案

**与 ClipByValue 的关键区别：**

| 对比项 | ClipByValue | Lerp |
|--------|-------------|------|
| 输入张量 | 1 个 | 2 个（start, end） |
| Queue 数量 | 2（in + out） | 3（start_in + end_in + out） |
| 向量步骤 | 2 步（Maxs + Mins） | 3 步（Sub + Muls + Add） |
| PipeBarrier | 1 次 | 2 次 |

**Kernel 侧三步计算流水线：**

1. `Sub(yLocal, endLocal, startLocal)` — 计算 `end - start`
2. `PipeBarrier<PIPE_V>()` → `Muls(yLocal, yLocal, weight)` — 乘以权重
3. `PipeBarrier<PIPE_V>()` → `Add(yLocal, startLocal, yLocal)` — 加回 start

使用就地更新策略（复用 `yLocal`），节省 UB 空间。

#### 2.3 数据类型支持

| 类型 | 说明 |
|------|------|
| `float16` | 半精度浮点 |
| `float32` | 单精度浮点 |

（注意：Lerp 不支持 int32，因为涉及浮点乘加运算）

#### 2.4 精度说明

- **float32**：相对误差 < 1e-4
- **float16**：相对误差 < 1e-3

---

### 三、关键性能优化方法

#### 3.1 Double Buffer 流水线

传统串行模式中，数据搬运和计算交替进行，存在大量等待时间。Double Buffer 通过 Pipe/Queue 机制，将 CopyIn、Compute、CopyOut 三个阶段的执行时间重叠，有效隐藏了数据搬运延迟。

#### 3.2 多核并行切分

- 根据数据总长度和可用 AI Core 数量，计算最优核数 `blockDim`
- `perCoreElements` 对齐到 cache line（64B）边界，提升访存效率
- 最后 1 个 Core 自动处理可能不足一份 `perCoreElements` 的剩余数据
- 对无数据可处理的 Core 进行早期退出，避免空转

#### 3.3 UB 内存最优利用

- tileLength 在 UB 限制内取最大值，减少流水线轮次
- 根据数据类型动态调整：16bit 最多 32640 元素/tile，32bit 最多 16320 元素/tile
- tileLength 对齐到 32B（dataBlockElems）边界，满足向量指令要求

#### 3.4 向量化与标量混合

- 主体数据使用向量指令批量处理（32B 对齐部分）
- 尾块使用标量方式逐元素处理（< 32B 的剩余部分）
- 标量路径中 half 类型转 float 以保证比较精度

---

### 四、遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `AscendC::Dup` 未定义 | CANN 8.5.0 中 API 改名 | 改为 `AscendC::Duplicate` |
| `AscendC::PIPE_V` 未定义 | 全局枚举无需命名空间 | 改为 `PIPE_V` |
| half 类型不能直接比较 | AI Core 不支持 half 标量比较 | 先 `static_cast<float>()` 再比较 |
| 越界访问导致崩溃 | `uint64_t` 下溢产生极大正数 | 加 `if (start >= length)` 前置判断 |
| TilingContext GetInputShape 类型错误 | 返回 `StorageShape*` 非 `Shape*` | 改用 `GetRequiredInputTensor()->GetShapeSize()` |

---

### 五、文件结构

```
25303050126_liutianyu/
├── README.md
├── AscendC_算子开发教程_ClipByValue与Lerp.md   # 详细教学文档
├── 提交成功截图.png
└── code/
    ├── ClipByValue/
    │   ├── CMakeLists.txt
    │   ├── op_host/
    │   │   ├── CMakeLists.txt
    │   │   └── clip_by_value.cpp
    │   └── op_kernel/
    │       ├── CMakeLists.txt
    │       ├── clip_by_value.cpp
    │       ├── clip_by_value_tiling.h
    │       └── tiling_key_clip_by_value.h
    └── Lerp/
        ├── CMakeLists.txt
        ├── op_host/
        │   ├── CMakeLists.txt
        │   └── lerp.cpp
        └── op_kernel/
            ├── CMakeLists.txt
            ├── lerp.cpp
            ├── lerp_tiling.h
            └── tiling_key_lerp.h
```

---

### 六、构建说明

1. 安装 CANN 开发环境（Ascend-cann-toolkit 8.5.0+）
2. 进入算子目录（如 `code/ClipByValue/`）
3. 执行构建：
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=./output
   make -j$(nproc)
   make install
   ```
4. 生成的算子库文件位于 `build/output/` 目录
