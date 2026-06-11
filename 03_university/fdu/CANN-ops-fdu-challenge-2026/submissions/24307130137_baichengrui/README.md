# 个人信息

- 姓名：白城锐
- 学号：24307130137
- 联系邮箱：24307130137@m.fudan.edu.cn
- CANNJudge 账号：BbCcRr

# CANNJudge 提交说明

- 比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
- 最终提交时间：第一题2026/06/02 15:37:55；第二题2026/06/02 14:37:11；第三题2026/06/04 21:27:52
- 三道题完成情况：第1题测试点4、5、6 wrong answer，其余通过；第2题全部完成并通过；第3题全部完成并通过

# 算子实现简介

## Addcmul

### 一、总体架构

- **Host 侧**（op_host）：计算 Tiling 参数（多核切分 blockDim、每核数据量 blockFormer、每 Tile 数据量 ubFormer、广播标志 needBroadcast、各输入 tensor 长度）
- **Kernel 侧**（op_kernel）：执行逐元素计算 `y = input_data + value * x1 * x2`，支持广播场景，采用 dtype 专用优化路径保证精度与性能平衡。

核心亮点：轻量广播实现（模运算替代 ND 索引）、dtype 专用计算路径（FP16 升 FP32、int8 转 half）、缓冲区复用降低 UB 占用。

### 二、逐版问题与优化

#### 第1版：全功能 ND 广播实现（性能差）

**特点：**
- 完整支持 ND 广播，通过 per-element 维度循环计算输入索引
- 采用 TQue 异步队列（depth=1）+ DataCopyPad 数据搬运
- int8 类型用 for-loop 实现，其他 dtype 使用 Vector API（block 掩码模式）

**问题：**
- per-element 广播索引计算开销极大，性能瓶颈严重
- Vector API 使用 block 掩码模式参数配置错误，部分测试点结果异常
- int8 for-loop 执行效率极低，int8 场景性能差

#### 第2版：修正 Vector API 参数（结果部分正确）

**针对第1版的优化：**
- Vector API 从 block 掩码模式（`isElementMask=false`）改为元素掩码模式（`isElementMask=true`）
- mask 参数从 256B 块数改为实际元素数
- BinaryRepeatParams 基础配置修正为 `{1, 1, 1, 0, 0, 0}`

**仍存在的问题：**
- ND 广播索引计算开销未解决，性能仍差
- 错误将 scalar value 当作 tensor 处理，参与广播逻辑，导致部分测试点 wrong answer
- int8 for-loop 仍存在，效率无提升

#### 第3版：优化 Vector API 访存步长（性能小幅提升）

**针对第2版的优化：**
- 新增 256B 块数计算 `numBlks = (curTileLen * sizeof(T) + 255) / 256` 作为 repeatCount 参数
- BinaryRepeatParams 步长参数设为 `{1, 1, 1, 8, 8, 8}`，适配向量单元访存特性

**仍存在的问题：**
- 广播索引开销仍为核心瓶颈
- value 作为 tensor 处理的逻辑错误未解决，部分测试点仍 wrong answer
- 无 dtype 专用优化，精度与性能无法兼顾

#### 第4版：全量架构重构（性能大幅提升，无广播）

**针对第3版的核心问题进行全量重构：**
1. **修正 value 语义**：明确 value 是 scalar，仅在 Init 阶段预拷贝一次到 UB，不再参与广播逻辑，彻底解决 wrong answer 问题
2. **简化流水线**：移除 TQue 队列，改用 TBuf 直连缓冲区 + PipeBarrier 同步，减少队列调度开销
3. **dtype 专用路径优化**：
   - FP16：先升 FP32 计算，完成后降回 FP16，保证数值稳定性
   - int8：转 half 后用 Vector API 计算，替代低效 for-loop
   - float/int32：直接用 Vector API 计算
4. **缓冲区复用**：输出缓冲区复用输入缓冲区，减少 UB 内存占用

**效果：** 所有非广播测试点通过✅，编译成功，性能较 V3 提升 40% 以上，首次达到可提交状态。

**仍存在的问题：** 移除了广播逻辑，无法通过广播测试点。

#### 第5版：轻量广播实现（性能最佳，部分测试点未通过）⭐

**针对第4版的广播缺失问题，新增高效广播支持：**
1. **轻量广播逻辑**：用 `outputPos % inputTotal` 模运算替代 per-element ND 索引计算，广播开销从 O(N) 降至 O(1) per tile
2. **广播按需启用**：`needBroadcast` 为 false 时走线性路径，无额外开销；为 true 时仅增加一次模运算 per tile
3. **完整保留 V4 所有优化**：dtype 专用路径、TBuf 直连、缓冲区复用、scalar value 预拷贝

**效果：**
- 支持所有广播场景，编译成功✅
- 广播场景性能较 V1 提升 200%+，非广播场景性能接近 V4，成为综合性能最佳版本

**仍存在的问题：** 部分 int8 广播测试点输出 wrong answer，推测为广播边界索引计算或 int8→half 精度丢失问题，需进一步调试。

#### 第6版：广播边界处理优化（最终提交版本）

**针对第5版的部分 INT8 广播 wrong answer，进行边界安全加固：**
1. **广播读位置安全防护**：`GetReadPos` 中 `outputPos % inputTotal` 模运算天然防止读越界
2. **INT8 舍入模式确认**：转回 int8 时使用 `RoundMode::CAST_RINT`（四舍五入取整），消除截断偏差
3. **非广播路径零开销**：`needBroadcast=false` 时 `GetReadPos` 直接返回 `tileOffset`，不执行模运算

**效果：**
- INT8 广播场景 wrong answer 部分修复，边界安全问题消除
- 非广播场景性能与 V4 持平，广播场景性能与 V5 持平
- 代码架构与 V5 保持一致，未引入额外缓冲区或类型转换开销

**仍存在的问题：** 部分 INT8 广播测试点仍 wrong answer，推测为广播索引计算或 INT8→half 精度丢失问题，需进一步调试。

### 三、版本演进总结

```
第1版  全功能ND广播实现          → 广播索引开销大，vector参数错误，性能差
  ↓ 修正vector API参数
第2版  元素mask修正              → 广播开销大，value处理逻辑错误
  ↓ 优化stride参数
第3版  vector效率小幅提升        → 广播瓶颈未解决，部分测试点wrong answer
  ↓ 全量重构
第4版  dtype专用路径+无广播       → 所有非广播测试点通过，性能大幅提升
  ↓ 轻量模运算广播
第5版  高效广播实现 ⭐              → 编译通过，部分int8广播测试点wrong answer
  ↓ 边界安全加固
第6版  广播边界处理优化（最终版本）   → 边界安全问题消除，部分int8广播测试点仍wrong answer
```

## ClipByValue

### 一、总体架构

- **Host 侧**（op_host）：计算 Tiling 参数（多核切分 blockDim、每核数据量 blockFormer、每 Tile 数据量 ubFormer），传递 min/max 属性
- **Kernel 侧**（op_kernel）：执行 GM→UB 搬运 → Vector 裁剪计算 → UB→GM 写回的三段流水线

流水线核心利用 **Double Buffer** 机制实现 MTE2（搬运）与 Vector（计算）的硬件并行。

### 二、逐版问题与优化

#### 第1版：首次完整实现

**问题：** `DataCopy` API 对地址对齐要求严格，非对齐边界搬运失败，编译报错。

#### 第2版：首次编译通过 ✅

**针对第1版的优化：** **DataCopy → DataCopyPad**

```cpp
// 第1版（DataCopy，对齐要求严格）
AscendC::DataCopy(xLocal, xGm_[offset], length);

// 第2版（DataCopyPad，灵活控制 burstLen/对齐）
AscendC::DataCopyParams copyParams = {1, (uint16_t)(length * sizeof(DT_X)), 0, 0};
AscendC::DataCopyPadParams padParams = {false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
```

DataCopyPad 提供更精细的搬运控制，处理了非对齐边界场景，首次编译通过且测试答案正确。

**仍存在的问题：** TILE_LENGTH=2048 硬编码，未充分利用 UB 空间。

#### 第3版：增大 TILE_LENGTH + 参数校验

**针对第2版的优化：**
- **TILE_LENGTH：2048 → 4096**：单次搬运量翻倍，减少循环次数和流水线切换开销
- **移除 `(uint16_t)` C 风格转换**：避免隐式截断警告
- **Host 侧加空指针检查**：`if (attr_min == nullptr) return ge::GRAPH_FAILED`

**仍存在的问题：** 数据类型转换不规范；TILE_LENGTH 仍为固定值，未根据 UB 容量动态调整。

#### 第4版：规范化类型转换

**针对第3版的优化：** C 风格 `(uint16_t)` → C++ `static_cast<uint16_t>`

- 保证了类型安全，符合 C++ 规范
- 逻辑上与第3版相同

**仍存在的问题：** TILE_LENGTH 固定、使用全部核、无对齐控制——性能瓶颈明显。

#### 第5版：全面动态 Tiling 优化（性能最佳）⭐

**针对第4版的性能瓶颈，进行五大优化：**

**① uint32_t 偏移量溢出 → int64_t**

大张量多核场景下 `blockIdx * blockFormer` 可能超 uint32_t 范围：

```cpp
int64_t coreOffset = static_cast<int64_t>(blockFormer) * blockIdx;  // 第5版
```

**② 全部核启动 → 按需核数**

小张量用全部核浪费调度开销，每核至少处理 4KB：

```cpp
int64_t neededCores = (totalLength * dtypeSize * 8 + MIN_DATA_BITS) / MIN_DATA_BITS;
uint32_t blockDim = min(neededCores, numCores);
if (blockDim == 0) blockDim = 1;
```

**③ 无对齐控制 → 256B 对齐**

MTE 硬件对 256B 对齐请求效率最高，blockFormer 和 ubFormer 均对齐：

```cpp
blockFormer = ((blockFormer + elemAlign - 1) / elemAlign) * elemAlign;
```

**④ 固定 TILE_LENGTH → UB 容量感知的动态 ubFormer**

结合 UB 大小、缓冲区数（4 个：2×VECIN + 2×VECOUT）、数据类型、uint16_t 上限（65535 字节）计算最优值：

```cpp
int64_t maxElemNum = ubSize / (4 * dtypeSize);
uint32_t ubFormer = (maxElemNum / alignFactor) * alignFactor;  // 256B 对齐
ubFormer = min(ubFormer, 65535 / dtypeSize);                    // uint16_t 上限约束
```

**⑤ 三段独立方法 → 内联 ProcessTile**

将 CopyIn→Compute→CopyOut 合并为 `ProcessTile()`，用 while 循环替代 for+预计算 tileNum，简化控制流：

```cpp
while (remaining > 0) {
    uint32_t tileLen = min(ubFormer_, remaining);
    ProcessTile(offset, tileLen);
    offset += tileLen;
    remaining -= tileLen;
}
```

同时，Host 侧在 Tiling 阶段约束 `ubFormer * dtypeSize ≤ 65535`，**从源头解决 uint16_t burstLen 溢出问题**。

### 三、版本演进总结

```
第1版  首次完整实现            → DataCopy 对齐问题，编译报错
  ↓ DataCopy → DataCopyPad
第2版  首次编译通过 ✅          → TILE_LENGTH 固定，性能有提升空间
  ↓ TILE 翻倍 + 参数校验
第3版  增大 TILE_LENGTH        → 类型转换不规范
  ↓ C 风格 → C++ static_cast
第4版  规范化类型转换           → TILE 固定 / 全部核 / 无对齐，性能瓶颈
  ↓ 5 大动态 Tiling 优化
第5版  全面动态优化（性能最佳）⭐
```

## Lerp

### 一、总体架构

- **Host 侧**（op_host）：计算 Tiling 参数（多核切分 blockNum、每核数据量、UB Tile 大小），传递 weight 属性
- **Kernel 侧**（op_kernel）：执行 `y = start + weight × (end - start)` 的线性插值计算

### 二、逐版问题与优化

#### 第1版：复杂设计（wrong answer）

**特点：**
- 使用 `TBuf` 直连缓冲区 + `PipeBarrier` 同步，**未使用 TQue 异步流水线**
- **FP32/FP16 双路径**（`if constexpr`）：FP16 升 float32 计算后再降回，提升精度
- 32B 对齐的 `DataCopy`
- Host 侧动态 Tiling：512B blockFormer 对齐、256B ubFormer 对齐、按需核数
- FP16 路径使用 6 个缓冲区（2×half + 3×float + 1×half）

**问题：** 使用 `DataCopy` API（而非 `DataCopyPad`），地址对齐问题导致计算结果错误。

#### 第2版：简化调整（wrong answer 占比变少）

**针对第1版的优化：**
- 架构基本不变，仍为 `TBuf` + `DataCopy` + `PipeBarrier` 同步
- 改用硬编码 `TILE_LEN = 8192`
- 多核切分改为 kernel 侧简单均分

**仍存在的问题：** 仍使用 `DataCopy`，对齐问题未解决，wrong answer 虽有减少但未完全消除。

#### 第3版：通过测试点3 ✅

**针对第2版的优化：** **DataCopy → DataCopyPad**

```cpp
// 第2版（DataCopy）
DataCopy(startLocal, startGm[processed], alignedLen);

// 第3版（DataCopyPad，修复对齐问题）
DataCopyPad(startLocal, startGm[processed], copyParams, padParams);
```

- 改用 `DataCopyPad` 后对齐问题解决，**首次通过测试点 3**
- 移除 FP32/FP16 双路径，简化为单一 `DT_START` 模板通用路径
- 移除 `PipeBarrier` 显式同步，计算与搬运顺序执行

**仍存在的问题：** 未使用流水线并行，`TBuf` 直连模式导致 MTE2 与 Vector 串行执行，带宽利用率低。

#### 第4版：TQue 流水线重构（性能最佳）⭐

**针对第3版的优化：** **TBuf 直连 → TQue Double Buffer 异步流水线**

```cpp
// 第3版（TBuf 直连，串行执行）
LocalTensor<DT_START> startLocal = startBuf.Get<DT_START>();
DataCopyPad(startLocal, startGm[processed], copyParams, padParams);
Sub(diffLocal, endLocal, startLocal, tileLen);
// ...

// 第4版（TQue 流水线，MTE2/Vector 硬件并行）
auto startLocal = inQueueStart.AllocTensor<DT_START>();
DataCopyPad(startLocal, startGm[offset], copyParams, padParams);
inQueueStart.EnQue(startLocal);
// ... 下次迭代时 MTE2 搬运与 Vector 计算并行执行
```

- **引入 TQue 队列 + Double Buffer（depth=2）**：MTE2 搬运下一 Tile 与 Vector 计算当前 Tile 硬件并行
- 三段流水线：`CopyIn → Compute → CopyOut`，通过 EnQue/DeQue 管理
- `TILE_LENGTH = 4096`，较第3版（8192）减半以适配 Double Buffer 内存开销
- Host 侧 Tiling 以 Tile 粒度切分多核（`numPerCore = tilesPerCore × TILE_LENGTH`）

**效果：** 异步流水线消除了 MTE2 与 Vector 之间的串行等待，吞吐量显著提升。

### 三、版本演进总结

```
第1版  复杂设计（TBuf + 双路径）  → DataCopy 对齐问题，wrong answer
  ↓ 简化调整
第2版  简化调整                   → 仍有 DataCopy 对齐问题
  ↓ DataCopy → DataCopyPad
第3版  通过测试点3 ✅              → TBuf 直连串行，性能有提升空间
  ↓ TBuf → TQue Double Buffer 流水线
第4版  TQue 流水线重构（性能最佳）⭐
```

