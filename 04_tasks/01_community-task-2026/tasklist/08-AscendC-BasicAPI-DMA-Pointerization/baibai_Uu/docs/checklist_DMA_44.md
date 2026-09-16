# DMA Basic API 官方重载全量勾选（44/44）

来源：官方 DMA 接口表。改造状态：设计阶段（未改实现）。  
`[ ]` = 未改 / `[x]` = 已指针化并通过自测。粘贴原文 `Std::` 记为 `std::`。

## 汇总

| API | 文件 | 条数 |
|-----|------|------|
| DataCopy | `kernel_operator_data_copy_intf.h` | 27 |
| DataCopyPad | `kernel_operator_data_copy_intf.h` | 11 |
| DataCopyL1ToUB | `kernel_operator_data_copy_intf.h` | 2 |
| DataCacheCleanAndInvalid | `kernel_operator_cache_intf.h` | 3 |
| DataCachePreload | `kernel_operator_cache_intf.h` | 1 |
| **合计** | | **44** |

---

## DataCopy（27）

| # | 状态 | 签名要点 |
|---|------|----------|
| 1 | [ ] | SFINAE `bfloat16←float`；`DataCopy(Local, Local, DataCopyParams, DataCopyEnhancedParams)` |
| 2 | [ ] | SFINAE `int16←int32`；`__inout_pipe__(V)`；Enhanced |
| 3 | [ ] | SFINAE `int8←int32`；`__inout_pipe__(V)`；Enhanced |
| 4 | [ ] | SFINAE `uint8←int32`；`__inout_pipe__(V)`；Enhanced |
| 5 | [ ] | SFINAE `float←half`；`__inout_pipe__(V)`；Enhanced |
| 6 | [ ] | SFINAE `half←float`；Enhanced |
| 7 | [ ] | SFINAE `half←int32`；`__inout_pipe__(V)`；Enhanced |
| 8 | [ ] | `enableSmallC0`；`__inout_pipe__(MTE2)`；`(Local, Global, Dn2NzParams)` |
| 9 | [ ] | `enableSmallC0`；`__inout_pipe__(MTE2)`；`(Local, Global, Nd2NzParams)` |
| 10 | [ ] | `(Global, Local, DataCopyCO12DstParams)` |
| 11 | [ ] | `(Local, Local, DataCopyCO12DstParams)` |
| 12 | [ ] | `(Local\<T\>, Local\<U\>, DataCopyParams)` |
| 13 | [ ] | `dim`+`NdDmaConfig`；`(Local, Global, MultiCopyParams)` |
| 14 | [ ] | `__inout_pipe__(MTE2)`；`(Local, Global, DataCopyParams, Enhanced)` |
| 15 | [ ] | `__inout_pipe__(MTE2)`；`(Local, Global, Nd2NzParams)` |
| 16 | [ ] | `__inout_pipe__(MTE2)`；SliceInfo[] |
| 17 | [ ] | `__inout_pipe__(MTE2)`；count |
| 18 | [ ] | `__inout_pipe__(MTE3)`；`(Global, Local, DataCopyParams, Enhanced)` |
| 19 | [ ] | `__inout_pipe__(MTE3)`；`(Global, Local, DataCopyParams)` |
| 20 | [ ] | `__inout_pipe__(MTE3)`；`Nz2NdParamsFull` |
| 21 | [ ] | `__inout_pipe__(MTE3)`；SliceInfo[] |
| 22 | [ ] | `__inout_pipe__(MTE3)`；count |
| 23 | [ ] | `(Local, Local, DataCopyParams, Enhanced)` 同类型 |
| 24 | [ ] | `(Local, Local, DataCopyParams)` 同类型 |
| 25 | [ ] | `(Local, Local, Nd2NzParams)` |
| 26 | [ ] | `(Local, Local, count)` |
| 27 | [ ] | `__inout_pipe__(MTE2)`；`(Local, Global, DataCopyParams)` |

完整签名见 [checklist_DataCopy.md](./checklist_DataCopy.md)。

---

## DataCopyPad（11）

| # | 状态 | 签名 |
|---|------|------|
| 1 | [ ] | `template <typename T, typename U, typename std::enable_if<std::is_same<PrimT<T>, U>::value && (!std::is_same<T, U>::value), bool>::type = true> __aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<U>& padParams);` |
| 2 | [ ] | `template <typename T, PaddingMode mode = PaddingMode::Normal> __aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams);` |
| 3 | [ ] | `template <typename T, PaddingMode mode = PaddingMode::Normal> __aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams);` |
| 4 | [ ] | `template <typename T, PaddingMode mode = PaddingMode::Normal> __aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams);` |
| 5 | [ ] | `template <typename T, PaddingMode mode = PaddingMode::Normal> __aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams);` |
| 6 | [ ] | `template <typename T> __aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams);` |
| 7 | [ ] | `template <typename T> __aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams);` |
| 8 | [ ] | `template <typename T> __aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams);` |
| 9 | [ ] | `template <typename T> __aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams);` |
| 10 | [ ] | `template <typename T> __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const Nd2NzParams& nd2nzParams);` |
| 11 | [ ] | `template <typename T> __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams, const Nd2NzParams& nd2nzParams);` |

---

## DataCopyL1ToUB（2）

| # | 状态 | 签名 |
|---|------|------|
| 1 | [ ] | `template <typename T, uint8_t subBlockId = 0> __aicore__ inline void DataCopyL1ToUB(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& repeatParams);` |
| 2 | [ ] | `template <typename T, uint8_t subBlockId = 0> __aicore__ inline void DataCopyL1ToUB(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count);` |

---

## DataCacheCleanAndInvalid（3）

| # | 状态 | 签名 |
|---|------|------|
| 1 | [ ] | `template <typename T, CacheLine entireType, DcciDst dcciDst> __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst);` |
| 2 | [ ] | `template <typename T, CacheLine entireType, DcciDst dcciDst> __aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst);` |
| 3 | [ ] | `template <typename T, CacheLine entireType> __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst);` |

---

## DataCachePreload（1）

| # | 状态 | 签名 |
|---|------|------|
| 1 | [ ] | `template <typename T> __aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t>& src, const T cacheOffset);` |

---

## 统一改造规则

1. 仅将 `LocalTensor<*>` / `GlobalTensor<*>` 改为可经 `GetUnderlyingPtr` 萃取的模板入参（或保留 Tensor 重载并新增指针重载，以实现评审结论为准；设计默认方案 B）。
2. 保留：`__inout_pipe__`、SFINAE、`PaddingMode`、`enableSmallC0`、`subBlockId`、`CacheLine`、`DcciDst`、`NdDmaConfig` 等非操作数模板参数。
3. 不修改 `*Impl` 数值语义；不改 VECTOR/CUBE 他册。
