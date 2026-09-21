# 功能基线设计

输入 CHW/NCHW，输出 y 与 INT8 indices，三种数据类型；不新增算法优化策略。
二维窗口扫描根据 max_pool_with_argmax_v3 裁剪；具体参考见 reference_map.md。

Host 共用 Parse 校验 pair 属性、dilation=1、padding、正维度、shape 溢出和 mask 容量。
InferShape 与 Tiling 使用相同的尺寸函数。Tiling 另外校验调用方传入的输出 shape/dtype，
用 CANN TILING_DATA_FIELD_DEF 记录 12 个 int64 字段，SaveToBuffer 序列化。
key=1/2/3 对应 FP16/FP32/BF16；blockDim=1，workspace=0。
OpDef 添加 ascend910b 配置，保留动态具体 shape 能力，未知 shape 图模式不作支持声明。

Kernel 是标准五参数入口，通过 GET_TILING_DATA 获取数据。输入 GlobalTensor 只读，
二维扫描每个合法位置；严格大于时更新，保留第一次最大值。FP16 无损转 FP32，
BF16 uint16 左移恢复精确 FP32，y 保留选中输入的原始位表示。没有 FP32→FP16 降精度。

两个 64 字节 UB 缓冲分别存 16 个 y 和 16 个 int32 index。
循环覆盖 maskWords：前 total 个槽写 argmax，其余槽清零。只有前 total 个槽有 y 写回。
DMA 起点按 16 元素分组对齐，最后不足一组用 DataCopyPad 写实际字节数。
S_MTE3 保证标量 UB 写在搬运前完成，MTE3_S 保证搬运结束后才复用 UB。
单核避免跨核同步，不需要 Host 预先清零 indices，也不依赖系统同名池化实现。

indices 的原始 INT8 shape 只定义容器大小，其前缀连续跨 N/C 存放小端 int32，
每个值是通道内 ih*W+iw。容量不足时返回失败，不截断索引或更改验收 shape。

构建：顶层 CMake 只调用 Python 标准库构建脚本。脚本使用现有 CANN msopgen 模板，
合并本地三份 host .cpp 成模板的一份同名源文件（不改变逻辑），复制头文件和 kernel，
设置 ascend910b/vendor=mpwm，调用模板 build.sh。aclnn 由 CANN 工具生成。
项目不修改 ops-nn 全局 CMake、不复制 3D 实现、不依赖 arch35 或内部 opdev 库。

验证：C++ 样例按仓库 aclnn 两阶段模式执行，缓冲区用 0xA5 预填。
Python 使用原 golden，比对 y 原始字节和全部 indices 字节；BF16 用位表示输入及
无损 FP32 golden 视图，不要求 ml_dtypes。设备执行结果写 JSON，失败退出码非零。
