# 接口与需求映射附录

[返回设计正文](design.md)。本附录保留完整接口、异常行为和需求到证据的映射。

## 接口与异常行为

类模板为 `aclco::HyperLogLog<Key>`，Key仅允许`std::int32_t`和`std::int64_t`。各方法使用下列参数顺序，stream默认值为nullptr（ACL默认流）：

| 接口 | 参数与返回值 | 语义 |
| --- | --- | --- |
| CreateWithSketchSizeKB | uint32_t kb, aclrtStream stream → HyperLogLog | 支持8/16/32/64/128/256 KiB |
| CreateWithPrecision | uint32_t p, aclrtStream stream → HyperLogLog | p=13..18 |
| CreateWithStandardDeviation | double sd, aclrtStream stream → HyperLogLog | 选择满足1.04/sqrt(m)<=sd的最小支持容量 |
| Create | uint32_t kb, aclrtStream stream → HyperLogLog | KB工厂的别名 |
| Destroy | 无参数，返回void | 显式释放；重复释放安全，ACL释放错误可报告 |
| Clear | aclrtStream stream → void | 全部寄存器归零，保留存储供复用 |
| Add | void* keys, Extent<size_t> count, aclrtStream stream → void | 连续Device输入；count=0时允许nullptr |
| Merge | const HyperLogLog& other, aclrtStream stream → void | 同Key和precision的逐寄存器max；允许自合并 |
| Estimate | aclrtStream stream → uint64_t，方法为const | 近似基数，空sketch为0；不改变寄存器 |

普通接口返回前同步指定流。配置非法或不兼容合并报invalid_argument；输入字节数/地址溢出或超出ACL分配范围报length_error；对已销毁或移动后对象执行读写/配置查询报logic_error。非空Add检查nullptr、对齐、Device内存属性及分配范围；ACL调用失败以runtime_error保留错误码。void*不携带运行时dtype元数据，元素类型由Key模板及调用者的内存内容共同确定。

复制被禁用，移动构造/赋值转移所有权；析构不抛异常。所属ACL上下文必须在输入及容器释放后才销毁。调用者须提供有效流，并对同一容器的Host方法及外部device-ref操作建立顺序依赖；本设计不承诺无序并发Clear/Destroy与Add的语义。

## 需求对应表

源码路径均相对于尚未发布的冻结源码仓；证据编号在[证据索引](evidence-index.md)解释。

| target要求 | 接口 / 实际行为 | 实现文件 | 原始验证 / 补充证据 |
| --- | --- | --- | --- |
| §2.1 I32/I64 | HyperLogLog<int32_t/int64_t>，其他Key编译期拒绝 | include/hyperloglog.h | E1六类各含两种dtype；E5 |
| 三种构造与Create | KB、precision、标准差工厂；Create为KB别名 | include/detail/hyperloglog/hyperloglog.inl | create_test.cpp，38组合；E1 |
| 单字节与容量 | m=KB×1024=2^p，KB六档，p13–18 | hyperloglog.inl；kernels.h（均在detail/hyperloglog） | 原始全部容量；E4显式内存审计 |
| Destroy/生命周期 | 禁复制、可移动、RAII；显式Destroy可报告释放失败 | include/hyperloglog.h；hyperloglog.inl | destroy_test.cpp，24组合；E4移动/销毁后使用 |
| Clear及复用 | 同步归零，存储复用 | hyperloglog.inl；kernels.h::Clear | clear_test.cpp，180组合 |
| Add空、重复、分批 | 空输入无更新但同步；非空按rank max | hyperloglog.inl；kernels.h::AddTiled | add_test.cpp，726组合；E4全sketch oracle |
| Merge兼容性 | 同Key与precision；逐字节max；允许自合并 | hyperloglog.inl；kernels.h::Merge | merge_test.cpp，192组合；E4 |
| Estimate精度、重复性 | 固定整数直方图归约及确定的finalizer | kernels.h；finalizer.h；tuning.h | estimate_test.cpp，120组合；E4全状态 |
| §2.4异常参数 | 配置/合并invalid_argument；失效对象logic_error；范围length_error；ACL错误runtime_error | hyperloglog.inl | create/add/merge原始断言；E4扩展参数检查 |
| §3.2精度 | 误差≤3×1.04/sqrt(m)，空值0 | tests/common/hll_test_common.h | E1原始数据与断言；不以CPU oracle代替NPU |
| §3.3性能 | 6操作×2dtype×6容量逐行≤baseline/0.4 | tests/performance/hyperloglog/；原测试框架 | E2；附录72行 |
| §2.2纯头文件 | 两个公开头文件，detail实现；CMake注册原测试 | CMakeLists.txt；tests/CMakeLists.txt | E3新构建16目标、多TU与API示例 |

**需要评审确认的契约边界**：target通用表要求空/未初始化句柄报错，而C++拥有型接口的重复Destroy、移动后Destroy实际为幂等释放；其他需有效存储的方法调用Validate。原始测试通过不自动消除这项文字口径差异，该差异在本设计中明确列出，待评审确认。void*也无法动态识别缓冲实际dtype，类型合法性由模板和调用者负责。非法流错误通过ACL调用/同步传播，不声称有额外的预启动流有效性检测或覆盖所有损坏流句柄。
