# MaxPool2dWithMaskBackward 算子设计文档

作者：Lxa2156。版本：2026-09-17。本文为设计评审稿；性能和双平台验收尚未完成。

## 1 需求背景

9月社区任务要求以 Ascend C 实现最大池化反向算子，交付至 ops-nn experimental/pooling，覆盖 A2 与 310P、CANN 9.0/9.1。最大池化的反向将梯度累加到正向选中的输入位置；重叠窗口需要兼顾数值顺序、并行写入和访存效率。

## 2 需求分析与拆解

输入为 gradOutput、self、indices，以及 kernelSize、stride、padding、dilation、ceilMode；输出 gradInput 与 self 同形状。self 只提供形状与类型。逻辑布局 NCHW，支持 FP32/FP16，BF16 仅 A2。任务 golden 限定 dilation=1、kernel 面积至少为2。非连续 aclTensor 的规范化和回写在正式 host 接口接入时实现。

indices 虽以 INT8 容器传递，任务数据在其起始地址连续存放小端 INT32 argmax。前缀下标为 (n*C+c)*Ho*Wo+oh*Wo+ow，值是通道内 ih*W+iw；不能按外壳通道跨度寻址。外壳形状为 [N,C,Kh*Kw,(ceil(Ho*Wo/16)+1)*32]。

已核对官方 ops-nn 9.0.0（commit fcebf031d193d641d2d1472a539bcc387b1e5f09）：A2 非1×1的 aclnnMaxPool2dWithMaskBackward 路径将同地址 indices 解释为 INT32，与任务 golden 一致；另一条 UINT16/NC1HWC0 路径的 bit mask 编码不同。310P 的正式前向联调和验收编码仍需确认，不能根据数据内容猜测格式。当前实现不宣称支持空张量、1×1、NaN/Inf 的完整泛化。

## 3 详细设计

### 3.1 数学定义与确定性

对 plane=n*C+c、输入位置 t=ih*W+iw，计算 dx[plane,t]=sum(dy[plane,q] if indices[plane,q]==t)，按 q 从小到大累加。只枚举窗口覆盖 t 的输出位置。高度范围为 [max(0,ceil((ih+padH-kernelH+1)/strideH)),min(Ho,floor((ih+padH)/strideH)+1))，宽度同理。先遍历 oh 再遍历 ow，使同一目标的加法顺序与 golden scatter 一致。所有 dtype 使用 FP32 累加，最后一次转换为输出类型。

### 3.2 Host 与 tiling 规划

复用现有 ACLNN 原型。Host 校验 shape、dtype、属性长度、stride/padding、输出尺寸和 indices 字节容量；偏移和乘积采用 int64 并防溢出，H*W 满足 INT32 索引范围。AutoContiguous 与输出回写由正式接口处理。不能将 CPU 包装层对非连续 numpy 输入的支持等同于 aclTensor 支持。

Tiling 参数包括 N、C、H、W、Ho、Wo、Kh、Kw、strideH/W、padH/W、输出元素数、tile大小、blockDim。按平台、dtype、窗口重叠程度和UB占用区分 tiling key。当前直调验证桥使用11个 int64 参数，独立设备缓冲区传参，尚未完成正式注册、tiling 和 ACLNN 封装。

### 3.3 当前正确性内核

将 gradInput 展平为256元素工作块，core b 处理 b、b+blockNum 等块。每个输入元素仅由一个核写入，不使用原子累加；未选中的位置写0，无需全局清零。尾块按真实元素数处理。标量 GM 写入后每64字节执行 DataCacheCleanAndInvalid；分配对齐冗余空间。FP16 转 FP32 累加；BF16 使用16位存储解码、FP32累加和 round-to-nearest-even 转回，避免当前编译器标量 bfloat16 转换限制。

代码包含共享 gather_core.h、Ascend C 三个 dtype 入口、ACL资源与事件计时桥、CANN CMake构建，以及固定种子的140条 Python验证入口。当前版本是标量正确性基线，GM重复读取较多，不能作性能达标承诺。

### 3.4 性能优化路线

无重叠窗口评估直接scatter与每核私有输出块；小重叠窗口采用UB缓存indices和梯度、复用连续tile；大kernel采用输入块归属的局部累加和单核写回。先保持确定性和FP32累加，再比较不同tile和核数，避免原子累加改变FP16/BF16舍入结果。优化版本均需与当前标量内核及独立golden对照。

## 4 可维可测分析

### 4.1 已完成验证

2026-09-17在 HiDevLab Ascend910B3 / CANN 9.0.0 / aarch64 环境完成 Ascend C内核与ACL启动库构建。配套140条完整尺寸用例全部真机通过：FP32 59条、FP16 44条、BF16 37条，与任务golden逐元素相等，最大绝对误差均为0。输入由配套golden和固定种子20260916+case编号生成。每条预热1次、事件计时3次，单方kernel event时间范围0.01372–61.53827 ms；未与TBE同口径比较，不能视为性能验收通过。

CPU侧140条同序参考全部通过；另有9类边界测试、48次独立标量池化反向对照、100个原生随机调度与scatter对照。ASan/UBSan曾在本机运行未返回，已中止，不计为通过。真机当前测试使用golden生成的indices前缀，尚未完成正式ACLNN正向联调或非连续aclTensor测试。

### 4.2 验收与兼容性

A2要求TBE耗时/候选耗时至少0.95；TBE小于100微秒且候选劣化超过30%的场景提供profiler图和分析。310P参考A2耗时乘5，超过该基准30%同样分析。测量记录预热、重复次数、芯片/CANN版本、kernel时间与数据搬运时间，并提供相同方式的标杆结果。

310P另需YOLOv11 / DOTAv1模型与A2的mAP50差不超过0.01。后续覆盖CANN9.0/9.1、FP32/FP16双平台、仅A2的BF16，以及非连续输入输出、ceil尾窗、缺省stride、重复最大值和重复运行确定性。

### 4.3 待完成事项

正式host注册/tiling/ACLNN接入；UB向量化优化与TBE比较；310P编码确认、编译和精度；YOLOv11模型验证；整理性能及精度报告、完成设计评审后提交验收。当前只提交设计评审，不声明上述待办已完成。

## 5 参考

- 任务讨论：https://gitcode.com/cann/ops-nn/discussions/9
- 社区流程：https://gitcode.com/org/cann/discussions/39
- 官方反向接口：https://gitcode.com/cann/ops-nn/blob/9.0.0/pooling/max_pool3d_grad_with_argmax/op_api/aclnn_max_pool2d_with_indices_backward.cpp
- 官方正向接口：https://gitcode.com/cann/ops-nn/blob/9.0.0/pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool2d_with_indices.cpp
