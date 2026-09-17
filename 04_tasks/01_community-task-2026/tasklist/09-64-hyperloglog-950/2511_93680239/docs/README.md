# HyperLogLog 设计评审入口

**先读 [设计正文 design.md](design.md)。** 正文按执行顺序介绍生命周期、字节布局、hash/rank、Add的分区与UB窗口、并发CAS、DMA合并、Estimate、同步及内存。

- [接口与需求映射](requirements.md)：完整接口、异常契约、实现文件和证据对应。
- [性能附录](appendix-performance.md)：同一轮72行性能及逐行余量。
- [证据索引](evidence-index.md)：源码版本、实现哈希、结果和原日志定位。
- [复现步骤](reproduce.md)：构建、功能与性能的独立复现入口。
- [官方清单对应](review-checklist.md)：模板及检查项覆盖。

原始功能1280/1280、一次完整串行性能72/72及干净构建/补充运行通过均为作者自测。设计评审、平台验收和实现代码合入是后续独立流程；源码与原始证据访问入口待补充。
