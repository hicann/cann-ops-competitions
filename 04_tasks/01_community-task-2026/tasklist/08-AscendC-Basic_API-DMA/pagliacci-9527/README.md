# Ascend C Basic API 指针化扩展（DMA）

社区任务：Ascend C Basic API 中 DMA（数据搬运 / 缓存）类接口的指针化扩展。贡献者：pagliacci-9527。

## 目录

```text
pagliacci-9527/
├── docs/
│   └── design.md
└── README.md
```

代码合入 [asc-devkit](https://gitcode.com/cann/asc-devkit) 的 `include/basic_api` 与 `impl/basic_api`。

## 范围

`DataCopy`、`DataCopyPad`、`DataCopyL1ToUB`、`DataCachePreload`、`DataCacheCleanAndInvalid`。通过 `GetUnderlyingPtr` 扩展指针入参，复用原 `*Impl`。
