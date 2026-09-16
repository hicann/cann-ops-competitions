# DynamicMap 容器设计文档

## 需求

任务要求在 `ops-collections` 中参考 cuCollections `dynamic_map` 实现 Ascend C/C++ DynamicMap 容器，支持：

- 构造、析构、Insert、Erase、Find、Contains、Reserve、InsertOrAssign。
- key/value dtype：`uint16_t`、`uint32_t`、`uint64_t`、`float`。
- 接口入参顺序尽量与 cuCollections 一致；device iterator 可用起始地址加元素个数替代。
- 性能目标：I16/I32 整体达到 A100 的 0.7 倍，I64 整体达到 A100 的 0.5 倍。

当前状态：代码、功能测试入口和性能测试入口已完成编译闭环；真实 Atlas 950 功能/性能验证仍受外部硬件阻塞，不能写验收就绪。

## 设计

DynamicMap 复用现有 StaticMap 能力，不改变 StaticMap/StaticSet 的 public API：

| 层级 | 设计 |
| --- | --- |
| host 容器 | `aclco::DynamicMap<Key, T, ...>` 持有多个 `StaticMap` 子表，管理容量、Reserve、size 和 device 元数据 |
| device 引用 | kernel 侧按子表地址和容量构造 `StaticMapRef`，跨子表执行查找/插入/删除 |
| SIMT kernel | 新增 DynamicMap 专用 Insert/InsertOrAssign/Find/Contains/Erase kernel，风格对齐既有 StaticMap kernel |

核心语义：

- Insert：跨全部子表查重；不存在则插入当前目标子表；返回新插入数量。
- InsertOrAssign：命中则更新 value；未命中则插入；返回新插入数量。
- Erase：跨全部子表删除；返回成功删除数量。
- Find：命中输出 value；未命中输出 `emptyValue`。
- Contains：命中输出 `true`，否则输出 `false`。
- Reserve：按目标容量提前创建子表，子表容量按倍数增长，目标负载约 0.60。

约束：

- `emptyKey` 是保留哨兵，不能作为有效 key 插入。
- `float` key 语义沿用 StaticMap 的 bitwise 比较，不做近似比较。
- 同批重复 key 或并发读写的最终值不作为确定性承诺；验收 case 不依赖 unspecified 行为。

## 文件与测试入口

计划交付文件：

| 类型 | 路径 |
| --- | --- |
| 容器入口 | `include/dynamic_map.h` |
| 实现 | `include/detail/dynamic_map/dynamic_map.inl` |
| kernel | `include/detail/dynamic_map/kernels.h` |
| 功能测试 | `tests/dynamic_map/` |
| 性能测试 | `tests/performance/dynamic_map/` |
| README | `docs/` 或仓根 README 中的 DynamicMap 章节 |

功能测试应覆盖：

- 构造/析构、空输入、Insert、Erase、Find、Contains、Reserve、InsertOrAssign。
- 4 种 dtype 的 key/value 组合。
- 重复 key、删除后重插、扩容、多子表查找、不同匹配率。

性能测试入口按任务书 8 类操作组织：

```text
Create / Destroy / Insert / Erase / Find / Contains / Reserve / InsertOrAssign
```

性能报告必须绑定真实 Atlas 950 环境、代码 commit、构建命令、原始日志/CSV、重复次数和任务书阈值对比。

## 兼容性

- 新增 DynamicMap，不修改 StaticMap/StaticSet 行为。
- 复用既有 `aclco` namespace、ACL stream 和 device memory 管理方式。
- 构建系统只增加 DynamicMap 测试和性能测试目标。
- 既有 StaticMap/StaticSet/utility 回归必须继续编译通过。

## 当前阻塞

| 阻塞 | 说明 |
| --- | --- |
| Atlas 950 真机缺失 | 950 simulator 下 ccec 程序 `aclInit=507000`，上游 static_map 同样失败，不能替代真机验收 |
| 功能套件未上板 | 需真实 Atlas 950 跑完整功能测试并归档日志 |
| 性能矩阵未实测 | 需真实 Atlas 950 跑 8 类性能入口并逐项对比任务书阈值 |
| 交付材料未定版 | fork、README、自验证报告、设计评审需绑定最终 commit 和真实证据 |

## 结论口径

当前只能写：

```text
DynamicMap 代码、功能测试入口和性能测试入口已完成设计与编译闭环；真实 Atlas 950 功能和性能验证仍受外部环境阻塞，不能升级为验收就绪。
```
