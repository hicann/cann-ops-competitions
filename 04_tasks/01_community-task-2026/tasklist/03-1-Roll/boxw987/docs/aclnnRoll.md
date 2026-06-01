# aclnnRoll

[📄 查看源码](../)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品</term> |    √     |

说明：当前工程在 `op_host/roll_def.cpp` 中注册 `ascend910b` 对应的 AICore 配置，支持数据类型包括 `BFLOAT16`、`FLOAT16`、`FLOAT`、`INT8`、`UINT8`、`INT32`、`UINT32`、`BOOL`，其中 `BOOL` 沿原生 `Roll` AICore 路径执行。

## 功能说明

- 接口功能：沿给定尺寸和维度移动Tensor中的数据。

- 举例：

  ```
  x = tensor([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
  经过roll(x, 1, 0)计算后，（在dim=0的维度上向下整体移动1行）

  x = tensor([[7, 8],
              [1, 2],
              [3, 4],
              [5, 6]])
  ```

## 函数原型

每个算子采用 aclnn 两段式接口，必须先调用“aclnnRollGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRoll”接口执行计算。

```cpp
aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* x,
    const aclIntArray* shifts,
    const aclIntArray* dims,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnRoll(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnRollGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度（shape）</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入的原始数据。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、UINT8、INT32、UINT32、BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shifts（aclIntArray*）</td>
      <td>输入</td>
      <td>指定每个维度上要滚动的步数。</td>
      <td>当dims非空时数组长度必须与dims保持一致；当dims为空时长度必须为1；对于0维输入同样只允许长度为1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dims（aclIntArray*）</td>
      <td>输入</td>
      <td>指定要滚动的维度。</td>
      <td>`aclnn`接口中参数对象必传，但数组可为空；对非空、非0维、非空Tensor输入，数组非空时长度必须与shifts保持一致，取值范围在[-x.dim(), x.dim() - 1]之内；对于0维输入仅允许空数组；空Tensor在完成基础参数合法性校验后直接返回，不继续进入普通dims取值范围校验。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>滚动处理后的输出数据。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、UINT8、INT32、UINT32、BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小；该输出指针本身不可为nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程；该输出指针本身不可为nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体含义参见 CANN aclnn 返回码说明。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>第一段接口传入的x、shifts、dims、out、workspaceSize或executor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>x的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>当dims非空时，shifts和dims的size不保持一致。</td>
    </tr>
    <tr>
      <td>当dims为空时，shifts数组长度不等于1。</td>
    </tr>
    <tr>
      <td>对非空、非0维、非空Tensor输入，dims的数值不在[-x.dim(), x.dim() - 1]之内。</td>
    </tr>
    <tr>
      <td>x的维度超过8维。</td>
    </tr>
    <tr>
      <td>out和x的shape不一致</td>
    </tr>
  </tbody>
  </table>

## aclnnRoll

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace（void*）</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t）</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRollGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor*）</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream（aclrtStream）</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体含义参见 CANN aclnn 返回码说明。

## 约束说明

- aclnnRoll默认确定性实现。
- 仅支持`ND`格式。
- 输入与输出shape、dtype需要保持一致。
- `dims`参数对象必传，可为空数组；`dims`非空时长度必须与`shifts`一致。
- `dims`为空时，`shifts`长度必须为`1`。
- 输入rank支持`0`到`8`维；对非空、非`0`维、非空Tensor输入，非空`dims`的取值范围为`[-rank, rank - 1]`。
- `0`维Tensor仅允许`dims`为空且`shifts`长度为`1`。
- 空Tensor在完成基础参数合法性校验后直接返回，不继续进入普通`dims`取值范围校验。

## 调用示例

完整调用样例请参考本交付目录下的 [README](../README.md) 与 [样例源码](../examples/test_aclnn_roll.cpp)。
