# Roll

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品|√|

说明：当前工程在 `op_host/roll_def.cpp` 中注册 `ascend910b` 对应的 AICore 配置，支持数据类型包括 `UINT8`、`INT8`、`BOOL`、`BFLOAT16`、`FLOAT16`、`FLOAT`、`INT32`、`UINT32`，其中 `BOOL` 沿原生 `Roll` AICore 路径执行。

## 功能说明

- 算子功能：沿指定维度循环滚动张量元素。

- 计算公式：

$$
y = Roll(x, shifts, dims)
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入tensor。</td>
      <td>UINT8、INT8、BOOL、BFLOAT16、FLOAT16、FLOAT、INT32、UINT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>UINT8、INT8、BOOL、BFLOAT16、FLOAT16、FLOAT、INT32、UINT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shifts</td>
      <td>属性</td>
      <td>指定每个维度上的循环滚动步数。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dims</td>
      <td>属性</td>
      <td>指定需要滚动的维度，默认空列表表示按展平后一维处理。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持 `ND` 格式。
- 输入与输出 shape、dtype 需要保持一致。
- `dims` 参数对象必传，可为空数组；`dims` 非空时长度必须与 `shifts` 一致。
- `dims` 为空时，`shifts` 长度必须为 `1`。
- 输入 rank 支持 `0` 到 `8` 维；对非空、非 `0` 维、非空 Tensor 输入，非空 `dims` 的取值范围为 `[-rank, rank - 1]`。
- `0` 维 Tensor 仅允许 `dims` 为空且 `shifts` 长度为 `1`。
- 空 Tensor 在完成基础参数合法性校验后直接返回，不继续进入普通 `dims` 取值范围校验。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_roll.cpp](./examples/test_aclnn_roll.cpp) | 通过[test_aclnn_roll](./docs/aclnnRoll.md)接口方式调用Roll算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| boxw987 | 个人开发者 | Roll | 2026/04/30 | Roll算子适配社区任务交付 |
