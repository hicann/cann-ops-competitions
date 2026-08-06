# 2026 HCCL 华北赛区决赛代码归档

- 学校：南开大学
- 队伍名称：春和景明( ˝ᗢ̈˝ ) 的队伍
- 参赛账号：春和景明( ˝ᗢ̈˝ )（@2302_80894008）
- 冻结版本：V42 Legal Pure-Layer
- 原始提交包：`SUBMIT_CHAMPION_V42_LEGAL_PURE_LAYER_417342c.zip`
- 提交包 SHA-256：`982fb185bd0065afcdcc7e81deca39069033ec4c692fb6177d17f5dee4d6ec7c`
- 官方工程模板：`allgather_ccu_problem_264_template.zip`
- 工程模板 SHA-256：`5ceed74f5f481bf901bedd7f961c88dc190503e2bf8d58443def12079f6b3723`

V42 是本地比赛材料中最后一个同时生成正式 ZIP 与独立 SHA-256 清单的决赛冻结包。本目录以原始官方模板补齐为完整可编译工程，再覆盖 V42 的六个参赛源码文件；六个源码文件均与原始提交包逐字节一致，其余 13 个文件保持官方模板内容。

归档前已复核 ZIP 完整性、六文件哈希、纯 CCU 分层与单 IO Die 约束。该说明只记录归档来源和本地结构校验，不新增或修改比赛算法，也不主张未由官方平台确认的成绩。

| 文件 | SHA-256 |
| --- | --- |
| `include/custom.h` | `47571e83769b602f08bc97eeeb0e3045cac96be127ae86f26b976920574cef8f` |
| `op_host/allgather.cc` | `4c4f146e4e74700a4ebc9d164ee4e518ccf5c6caa1b9bb990568ea1a256d06b7` |
| `op_host/exec_op.h` | `ef7d73a0cb4535e64d116f6c8fee0b25e771ca5d559e7116ec5819353b339147` |
| `op_host/exec_op.cc` | `823ef8019b936e43eb3855aff3f06841b3c5f1fecf88f78b977d8a681dccbf23` |
| `op_kernel_ccu/ccu_kernel.h` | `9c5ace83430b632fc7f8ee20d6564a13eb73f1a30ebd0568e97931ffe665f406` |
| `op_kernel_ccu/ccu_kernel.cc` | `dbe0e52dce4e4654d979871cd003afbebf544e724a2e10ac946f4bc199f42504` |
