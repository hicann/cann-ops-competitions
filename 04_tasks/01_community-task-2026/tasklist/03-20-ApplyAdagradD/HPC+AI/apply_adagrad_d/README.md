# ApplyAdagradD

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term> |    √     |

## 功能说明

- 算子功能：实现AdagradD优化器功能，计算公式如下：
- 计算公式：

$$
g_t = grad
$$

$$
accum_{t}=\begin{cases}
accum_{t-1}+g_{t}^{2}
& \text{ if } update\_slots = true\\
accum_{t-1}
& \text{ if } update\_slots = false
\end{cases}
$$

$$
\theta_{t}=\theta_{t-1}-\frac{\eta \cdot g_t}{\sqrt{accum_t}}
$$

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
      <td>var</td>
      <td>输入</td>
      <td>公式中的参数张量 θ<sub>t-1</sub>。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>公式中的累积梯度平方张量 accum<sub>t-1</sub>。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>公式中的学习率 η。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>公式中的梯度张量 g<sub>t</sub>。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>属性</td>
      <td>是否更新accum，为true时更新 accum<sub>t-1</sub>+g<sub>t</sub><sup>2</sup>，为false时保持 accum<sub>t-1</sub> 不变。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var_out</td>
      <td>输出</td>
      <td>公式中的更新后参数张量 θ<sub>t</sub>。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_out</td>
      <td>输出</td>
      <td>公式中的更新后累积梯度平方张量 accum<sub>t</sub>。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|------|------|------|
| aclnn调用 | [test_aclnn_apply_adagrad_d](./examples/test_aclnn_apply_adagrad_d.cpp) | 通过[aclnnApplyAdagradD](./docs/aclnnApplyAdagradD.md)接口方式调用ApplyAdagradD算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| HPC+AI | 个人开发者 | ApplyAdagradD | 2026/04/22 | ApplyAdagradD算子适配开源仓 |
