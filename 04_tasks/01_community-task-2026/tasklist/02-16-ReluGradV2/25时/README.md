# ReluGradV3 (Select 优化版)

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品</term> |    √     |

## 功能说明

- 算子功能：对应 Relu 操作的反向传播梯度。

- 计算公式：

$$
backprops[i] = mask[i] \ ?\ gradients[i] : 0
$$

其中 mask 为 0/1 门控。数学上等价于与 gradients 逐元素相乘，但实际计算对大 tile 采用 Compares+Select 直接选择，无需乘法。

## 实现方式

采用 **Compares+Select + Mul+Cast 双路径**：

| 数据类型 | 大 tile（≥128 half） | 小 tile（<128 half） |
|----------|---------------------|---------------------|
| half/float | Compares+Select（mask→half → Compares≠0 → Select） | Mul+Adds 回退 |
| bf16 | Cast grad→float + Compares+Select | Mul+Cast 回退 |
| int8/uint8/int32 | — | Mul+Cast |

Select 路径仅需 3 条 Vector 指令（Cast + Compares + Select），省去中间 Mul 和多层 Cast，UB 占用较纯 Mul+Cast 降低 30%+。同时 mask=0 时直接输出 0，天然规避 NaN/INF 通过乘法传播的问题。

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>gradients</td>
      <td>输入</td>
      <td>传递给对应 Relu 操作的反向传播梯度</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>作为输入传递给对应 ReluV2 操作的特征，0/1 门控</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>公式中的输出张量</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- Compares/Select 要求操作数 256B 对齐（910B 上等价 128 个 half 元素），大 tile 阈值由此确定。
- Select 模式 1/2 内部需占用 8KB UB。

## 调用说明

| 调用方式  | 调用样例                                                    | 说明                                                                   |
| --------- | ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_relu_grad_v3.cpp](./examples/test_aclnn_relu_grad_v3.cpp) | 通过 `aclnnReluGradV3` 接口方式调用 ReluGradV3 算子。 |

## 对标说明

本算子对标 TBE 的 `ReluGradV2` 算子。