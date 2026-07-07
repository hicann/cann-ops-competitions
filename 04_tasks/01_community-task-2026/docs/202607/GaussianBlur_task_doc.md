# 7月社区任务-GaussianBlur算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950PR
- **开源仓地址**：https://gitcode.com/cann/ops-cv（预计 2026 年社区开放，验收前以社区通知为准）
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C + C++（OpenCV 适配层）
- **对标参考**：[OpenCV cv::GaussianBlur](https://github.com/opencv/opencv)（`modules/imgproc`，建议 ≥ 4.8）

## 任务概述

参考 OpenCV `imgproc` 模块中的 `cv::GaussianBlur`，在昇腾 NPU 上基于 **Ascend C Kernel + OpenCV C++ 适配层** 实现功能与接口完全对齐的高斯滤波算子，完成算子设计、开发、测试全流程，测试通过后合入昇腾 CV 算子开源仓。

**验收口径**：以 **OpenCV C++ 层 `cv::GaussianBlur()`** 的功能、精度、性能为唯一验收基准；**aclnn C++ 接口为必选交付项**（须与 OpenCV 层前向结果一致）；`cv_hal` 替换接口为可选。

### 功能定义

高斯滤波对输入图像做**可分离二维卷积**（先沿 X 方向、再沿 Y 方向），等价于与二维高斯核卷积：

\[
\text{dst}(x,y) = \sum_{i,j} \text{src}(x+i,\, y+j) \cdot G_{\sigma_x}(i) \cdot G_{\sigma_y}(j)
\]

其中 \(G_\sigma\) 由 `getGaussianKernelBitExact` 生成的一维高斯核（softfloat 位精确），经定点化后用于 `CV_8U` / `CV_16U` 位精确路径。

**实现要点（须与 OpenCV 一致）**：

1. 由 `sigmaX`/`sigmaY`/`ksize` 生成一维核 `kx`、`ky`（`createGaussianKernels`）
2. 调用可分离卷积 `sepFilter2D`（或等价 NPU 两趟 1D 卷积）
3. 各通道**独立**处理；支持 in-place（内部必要时 clone）

### 典型应用场景

图像去噪、边缘检测预处理（Canny/Sobel 前）、SSIM/光流、目标检测与视觉 pipeline 基础算子。

---

## 核心开发要求及验收标准

### 1. OpenCV C++ 层接口（**必选，逐字对齐**）

须与 `opencv2/imgproc.hpp` 声明完全一致：

```cpp
CV_EXPORTS_W void GaussianBlur(
    InputArray src,
    OutputArray dst,
    Size ksize,
    double sigmaX,
    double sigmaY = 0,
    int borderType = BORDER_DEFAULT,
    AlgorithmHint hint = cv::ALGO_HINT_DEFAULT);
```

**Legacy C API**（可选，建议复用 C++ 路径）：

```cpp
// cvSmooth(..., CV_GAUSSIAN, ...) 内部等价于：
cv::GaussianBlur(src, dst, Size(param1, param2), param3, param4, BORDER_REPLICATE);
```

**HAL 替换接口**（可选，便于 OpenCV 集成）：

```cpp
int cv_hal_gaussianBlur(
    const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step,
    int width, int height, int depth, int cn,
    size_t margin_left, size_t margin_top,
    size_t margin_right, size_t margin_bottom,
    size_t ksize_width, size_t ksize_height,
    double sigmaX, double sigmaY, int border_type);
```

参考：`modules/imgproc/src/hal_replacement.hpp`。

### 2. 参数与约束

| 参数 | 类型 | 约束 |
|------|------|------|
| `src` | InputArray | 非空；深度 **CV_8U / CV_16U / CV_16S / CV_32F / CV_64F**；通道数 1～4（及 OpenCV 支持的多通道） |
| `dst` | OutputArray | 与 `src` **同尺寸、同 type**；自动 `create` |
| `ksize` | Size | `width`、`height` 均为**正奇数**，或 **0**（由 sigma 自动推算）；允许 `width ≠ height` |
| `sigmaX` | double | ≥ 0；与 `ksize` 共同决定核 |
| `sigmaY` | double | ≤ 0 时设为 `sigmaX`；均为 0 且 ksize 给定时使用 binomial 预定义核 |
| `borderType` | int | 见 BorderTypes；**不支持 BORDER_WRAP**；默认 `BORDER_DEFAULT` |
| `hint` | AlgorithmHint | `ALGO_HINT_DEFAULT` / `ACCURATE` / `APPROX`；影响是否走 HAL 近似路径 |

#### 2.1 ksize 与 sigma 推算规则（须复现）

| 条件 | 行为 |
|------|------|
| `ksize.width == 0 && sigmaX > 0` | `ksize.width = round(sigmaX × (8U?3:4) × 2 + 1) \| 1` |
| `ksize.height == 0 && sigmaY > 0` | 同上（用 sigmaY） |
| `ksize.width == 1 && ksize.height == 1` | **直接 copyTo**，不做卷积 |
| `sigmaX == sigmaY == 0` 且 ksize 为 3/5/7/9 | 8U 路径可使用 **binomial 位精确核**（见 `getGaussianKernelBitExact`） |

#### 2.2 边界与 ROI

| 场景 | 行为 |
|------|------|
| borderType = BORDER_REPLICATE | 仅支持BORDER_REPLICATE即可 |
| AlgorithmHint = ALGO_HINT_DEFAULT | 仅支持ALGO_HINT_DEFAULT即可 |

#### 2.3 数据类型分级

| 级别 | depth | 实现路径 | 性能验收 |
|------|-------|----------|----------|
| **L1** | CV_32F | Ascend C 浮点 separable conv | 纳入 |


### 3. 算子约束限制

1. **通道独立**：多通道不得混通道卷积；
2. **BORDER_WRAP 禁止**：调用时若指定须与 OpenCV 一致报错或拒绝；
3. **in-place**：须支持；与 OpenCV 一样可在 bit-exact 路径内部 clone；
4. **非连续 Mat / step**：须支持任意合法 `step`（HAL 接口含 step）；
5. **仅验收前向**（GaussianBlur 本身无 backward）。

### 4. 功能验收用例（须全部通过）

| 编号 | 场景 | 参考来源 |
|------|------|----------|
| TC-03 | 1/2/3/4 通道 256×128 | 同上 |
| TC-04 | borderTypes + ROI | `test_filter.cpp` `Imgproc_GaussianBlur.borderTypes` |
| TC-05 | 空 src 抛异常 | `test_filter.cpp` issue 16857 |
| TC-06 | 仅 sigma 推 ksize（Size()） | `test_filter.cpp` regression_11303 |
| TC-07 | 32F 大分辨率 | 同上 |
| TC-08 | ksize 1×1 copy | `smooth.dispatch.cpp` |
| TC-09 | 非方形 ksize（宽≠高） | 自行构造 |
| TC-10 | BORDER_ISOLATED ROI | 自行构造 |
| TC-11 | ALGO_HINT_ACCURATE vs APPROX | `test_smooth_bitexact.cpp` |

**真值与精度**：详见「精度要求」；功能比对以 **OpenCV CPU**（同版本）为标杆。

### 5. 性能要求

整体性能以 **OpenCV CUDA（A100）或 OCL GPU** 为参考，验收取最优实现。

| 分档 | 场景 | depth | ksize | sigma | **达标基线（≥ A100 OpenCV）** |
|------|------|-------|-------|-------|-------------------------------|
| S1 | 1024×1024 | CV_32FC1 | 5×5 | 1.2 | **0.45×** |

> 图像算子受 HBM 带宽与 separable 两趟读写影响，小图固定开销可能导致比值偏低，自验证报告须分档说明。

### 6. 精度要求

算子计算精度需严格满足《[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)》，采用 [AscendOpTest](https://gitcode.com/HIT1920/AscendOpTest) 测试。

**真值生成方式**：以 **OpenCV GPU**（同版本、`cv::GaussianBlur`）为标杆；NPU 路径通过 OpenCV 适配层调用获取结果并比对。

**单标杆不满足时**：采用 [ATK](https://gitcode.com/AscendTest/ATK) 双标杆比对（`cv_fused_double_benchmark`），以更高精度的 CPU 实现为真值，同时评估同精度 CPU 与 NPU 算子实现相对于该真值的误差；满足条件为 NPU/同精度 CPU 的**最大相对误差比例 ≤ 2**、**平均相对误差比例 ≤ 1.2**、**均方根误差比例 ≤ 1.2**。

**说明**：ATK双标杆测试需要A100环境，请开发者自行准备。

精度自测用例参考[自测用例目录](./self_test_case/GaussianBlur/)。

| 数据类型 / 场景 | 精度策略 |
|-----------------|----------|
| **CV_32F（L1）** | 对标 OpenCV CPU；大 sigma/大核可用 ATK 浮点阈值 |

### 7. 接口分层要求

```
OpenCV cv::GaussianBlur（必选，功能/精度/性能验收基准）
    → aclnn 层（必选）：aclnnGaussianBlur GetWorkspaceSize + 执行入口
    → Ascend C Kernel（必选）：sepFilter2D 两趟 1D 卷积 + 边界处理
    → cv_hal_gaussianBlur（可选，便于 OpenCV HAL 集成）
```

#### 7.1 aclnn C++ 层接口（**必选**）

须按 CANN 标准 aclnn 风格封装，提供 **GetWorkspaceSize + 执行入口** 两阶段调用，语义与 OpenCV `cv::GaussianBlur` 对齐。建议接口原型如下：

```C
/**
 * @brief 获取 GaussianBlur 算子所需 workspace 大小
 */
aclnnStatus aclnnGaussianBlurGetWorkspaceSize(
    const aclTensor *src,
    const aclIntArray *ksize,       /* [width, height]，0 表示由 sigma 推算 */
    double sigmaX,
    double sigmaY,
    int64_t borderType,
    const aclTensor *dst,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief GaussianBlur 计算入口
 */
aclnnStatus aclnnGaussianBlur(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclTensor *src,
    const aclIntArray *ksize,
    double sigmaX,
    double sigmaY,
    int64_t borderType,
    aclTensor *dst,
    aclrtStream stream);
```

| 要求项 | 说明 |
|--------|------|
| 调用流程 | GetWorkspaceSize →（可选 preprocess）→ aclnnGaussianBlur |
| 参数语义 | 与 OpenCV `ksize`/`sigmaX`/`sigmaY`/`borderType` 一致 |
| 结果一致性 | aclnn 与 OpenCV 适配层前向 **bit-exact（8U/16U）或 ATK 阈值内（32F）** |
| 交付物 | aclnn 头文件、Host 实现、C++ 调用示例与测试 |

#### 7.2 Kernel 实现建议

- **32F**：NPU float32  separable conv，累加顺序差异允许在 ATK 阈值内；
- **边界**：Tile 级 halo 交换，支持 `BORDER_REPLICATE` / `REFLECT` / `REFLECT101` / `CONSTANT` 等 OpenCV 支持的模式。

### 8. 文档规范要求

1. 设计文档按[社区模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)填写；
2. 重点描述：**与 OpenCV 接口对照、aclnn 接口设计、核生成与定点化、separable 两趟调度、边界 halo、8U bit-exact 路径、CV_64F L2 策略、多路径性能对比**；
3. 自验证报告须含 OpenCV gtest 执行日志、**aclnn C++ 调用测试**、AscendOpTest/ATK 日志、性能截图；
4. README 含 OpenCV 调用示例、**aclnn C++ 调用示例**、支持 depth/ksize/border 表、L1/L2 分级说明。

---

## 验收交付件

1, 自测用例、测试结果报告、测试步骤指导文档

2, 算子代码的私仓邀请链接、代码仓路径、分支、算子目录

## PR 申请合入

测试通过后，在昇腾AMCT开源仓提交 PR 申请，申请将开发完成的代码合入 https://gitcode.com/cann/ops-cv/tree/master/gaussian_blur 。

---

## 参考资料

1. OpenCV 源码：`modules/imgproc/include/opencv2/imgproc.hpp`（L1575）
2. 实现：`modules/imgproc/src/smooth.dispatch.cpp`（`GaussianBlur`）、`smooth.simd.hpp`（`GaussianBlurFixedPoint`）
3. 测试：`modules/imgproc/test/test_smooth_bitexact.cpp`、`test/test_filter.cpp`
4. HAL：`modules/imgproc/src/hal_replacement.hpp`（`cv_hal_gaussianBlur`）
5. 精度标准：[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)

---

## 环境获取

1. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](pics/yunkaifa.png)

2. 使用 hidevlab WebIDE 算力（[https://hidevlab.huawei.com/online-develop-intro?from=hiascend](https://hidevlab.huawei.com/online-develop-intro?from=hiascend)）

   ![环境截图](pics/zaixiankaifa1.png)  

3. 如需额外环境资源，请联系昇腾小助手。


## 特别注意事项

1. **核系数须与 `getGaussianKernelBitExact` 一致**，不得自行用 float32 生成后量化（除非文档化且通过 bit-exact 用例）；
2. **sigma=0 的 binomial 核**（3/5/7/9）为常见 fast path，须专门优化；
3. **aclnn 接口为必选交付**，须与 OpenCV 层前向结果一致，并提供 C++ 调用示例与测试；
4. **性能验收取最优实现**，分档指标为达标基线；
5. 开发前阅读[【社区任务】流程及注意事项](https://gitcode.com/org/cann/discussions/39)。
