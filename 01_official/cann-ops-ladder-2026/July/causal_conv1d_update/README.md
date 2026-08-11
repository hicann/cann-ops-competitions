# causal_conv1d_update算子

**难度系数：2.0**

## 一、赛题背景

Causal Conv1D（因果一维卷积）是时序建模与自回归生成任务中的核心序列计算算子，广泛应用于大语言模型、状态空间模型以及时间序列预测等深度学习场景。

本题要求采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能、高兼容性、高适配性的 Causal Conv1D 算子。

## 二、算子功能描述

causal_conv1d_update算子的计算公式：
$$
y[c, t] = \text{bias}[c] + \sum_{j=0}^{K-1} \text{weight}[c, j] \cdot x_{\text{ext}}[c, t-j]
$$

其中：

* K=width，卷积核宽度，通常为 2/3/4
* $x_{ext}$ 为扩展后输入：当 $t-j < 0$ 时，从conv_state 读取历史；若无历史则为 0
* 可选融合 SiLU：$y = y \times \text{sigmoid}(y) = \frac{y}{1 + e^{-y}}$

## 三、核心定义与约束

### 3.1 参考实现
```python
import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1

def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    PyTorch reference implementation of causal_conv1d.

    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,)
        initial_states: (batch, dim, width - 1)
        final_states_out: (batch, dim, width - 1)
        return_final_states: bool
        activation: str

    Returns:
        out: (batch, dim, seqlen)
        final_states_out: (batch, dim, width - 1) if return_final_states
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    Args:
        x: Input tensor
        conv_state: (..., dim, state_len)
        weight: (dim, width)
        bias: (dim,)
        activation: str
        conv_state_indices: (batch,) int32
        num_accepted_tokens: (batch,) int32
        query_start_loc: (batch + 1,) int32
        pad_slot_id: int

    Returns:
        out: same shape as x
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)

    feature_dim = x.shape[-1] if query_start_loc is None else x.shape[1]
    if weight.shape[0] != feature_dim and weight.shape[1] == feature_dim:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    dim, width = weight.shape
    if dim != feature_dim:
        raise RuntimeError(
            f"causal_conv1d_update: weight dim mismatch, feature_dim={feature_dim}, weight.shape={tuple(weight.shape)}"
        )

    if conv_state.shape[-2] != dim and conv_state.shape[-1] == dim:
        # Accept both (..., dim, state_len) and (..., state_len, dim) inputs.
        conv_state = conv_state.transpose(-1, -2)
    if conv_state.shape[-2] != dim:
        raise RuntimeError(
            f"causal_conv1d_update: conv_state dim mismatch, "
            f"expected dim={dim}, conv_state.shape={tuple(conv_state.shape)}"
        )

    state_len = width - 1
    if conv_state.shape[-1] < state_len:
        raise RuntimeError(
            f"causal_conv1d_update: conv_state too short, need >= {state_len}, got {conv_state.shape[-1]}"
        )

    out = x.clone()

    def _select_state(i: int) -> torch.Tensor | None:
        if conv_state_indices is not None:
            idx = int(conv_state_indices[i].item())
            if idx == pad_slot_id:
                return None
            state = conv_state[idx]
        else:
            state = conv_state[i]
        return state

    def _run_one(seq_tokens: torch.Tensor, state: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # seq_tokens: [L, dim] -> [1, dim, L]
        x_ref = seq_tokens.transpose(0, 1).unsqueeze(0)
        init_state = state[..., offset : offset + state_len].unsqueeze(0)
        out_ref, final_state = causal_conv1d_ref(
            x_ref,
            weight,
            bias,
            initial_states=init_state,
            return_final_states=True,
            activation=activation,
        )
        state[..., :state_len].copy_(final_state.squeeze(0))
        # [1, dim, L] -> [L, dim]
        return out_ref.squeeze(0).transpose(0, 1)

    def _run_spec_decoding(seq_tokens: torch.Tensor, state: torch.Tensor, offset: int, seqlen: int) -> torch.Tensor:
        # Spec decoding mode: read history from offset position
        # hist: [1, dim, width - 1] from state[..., offset : offset + width - 1]
        # x_cat: [1, dim, (width - 1) + seqlen]
        hist = state[..., offset : offset + state_len].unsqueeze(0)
        x_ref = seq_tokens.transpose(0, 1).unsqueeze(0)
        x_cat = torch.cat([hist, x_ref], dim=-1)
        
        out_ref = F.conv1d(x_cat, weight.unsqueeze(1), bias, padding=0, groups=dim)
        out_ref = out_ref[..., :seqlen]
        if activation is not None:
            out_ref = F.silu(out_ref)
        
        # Update state: sliding window
        # new_state = concat(old_state[offset+1:], x)[-state_len:]
        # state_len = width - 1
        # new_state = x_cat.squeeze(0)[:, -(state_len):]
        # state[..., :state_len].copy_(new_state)
        if x_cat.shape[-1] > state.shape[-1] + 1:
            x_cat = x_cat[..., :(state.shape[-1] + 1)]
        state[..., :x_cat.shape[-1]-1].copy_(x_cat.squeeze(0)[:, 1:])
        
        return out_ref.squeeze(0).transpose(0, 1)

    if query_start_loc is None:
        if x.dim() == 2:
            batch = x.shape[0]
            for i in range(batch):
                state = _select_state(i)
                if state is None:
                    continue
                seq_tokens = x[i : i + 1]
                if num_accepted_tokens is not None:
                    accepted = int(num_accepted_tokens[i].item())
                    if accepted <= 0:
                        continue
                    # Spec decoding mode: use offset to read history
                    offset = accepted - 1
                    out_i = _run_spec_decoding(seq_tokens, state, offset, seq_tokens.shape[0])
                else:
                    out_i = _run_one(seq_tokens, state)
                out[i : i + out_i.shape[0]] = out_i
        else:
            batch = x.shape[0]
            for i in range(batch):
                state = _select_state(i)
                if state is None:
                    continue
                seq_tokens = x[i]
                if num_accepted_tokens is not None:
                    accepted = int(num_accepted_tokens[i].item())
                    if accepted <= 0:
                        continue
                    if accepted >= state.shape[1]-width + 2:
                        accepted = state.shape[1]-width + 2
                    # Spec decoding mode: use offset to read history
                    offset = accepted - 1
                    out_i = _run_spec_decoding(seq_tokens, state, offset, seq_tokens.shape[0])
                else:
                    out_i = _run_one(seq_tokens, state)
                out[i, : out_i.shape[0]] = out_i
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        for i in range(batch):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            state = _select_state(i)
            if state is None:
                continue
            seq_tokens = x[start:end]
            if num_accepted_tokens is not None:
                accepted = int(num_accepted_tokens[i].item())
                if accepted <= 0:
                    continue
                if accepted >= state.shape[1]-width + 2:
                    accepted = state.shape[1]-width + 2
                # Spec decoding mode: use offset to read history
                offset = accepted - 1
                out_i = _run_spec_decoding(seq_tokens, state, offset, seq_tokens.shape[0])
            else:
                out_i = _run_one(seq_tokens, state)
            out[start : start + out_i.shape[0]] = out_i

    return out.to(original_x_dtype)

```

### 3.2 输入输出与属性总览

| 类型 | 参数名 | 是否必选 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 描述 |
|------|--------|----------|------|----------|--------------|----------|------|
| INPUT | x | 必选 | tensor | (num_tokens, dim) or (batch, seqlen, dim) | bfloat16 | ND | 输入张量 |
| INPUT | conv_state | 必选 | tensor | (num_cache_lines, state_len, dim) | bfloat16 | ND | 状态缓存 |
| INPUT | weight | 必选 | tensor | (width, dim) | bfloat16 | ND | 卷积权重 |
| INPUT | bias | 可选 | tensor | (dim) | bfloat16 | ND | 偏置向量 |
| INPUT | query_start_loc | 可选 | tensor | (batch+1) | int32 | ND | varlen模式累积序列长度 |
| INPUT | num_accepted_tokens | 可选 | tensor | (batch) | int32 | ND | Speculative Decoding接受token数 |
| INPUT | conv_state_indices | 可选 | tensor | (batch) | int32 | ND | 批次到缓存行的映射 |
| ATTR | activation | 可选 | string | - | string | - | 激活函数，"silu"或None，默认None |
| ATTR | pad_slot_id | 可选 | int | - | int32 | - | 填充槽标识，默认-1 |
| OUTPUT | output | 必选 | tensor | 同输入x | bfloat16 | ND | 输出张量 |

### 3.3 关键输入约束

* 参数解释

- **x** (Tensor)：必选参数，输入张量，数据类型支持 bfloat16，维度为 [batch, seqlen, dim] 的 3 维 Tensor或 [batch, dim]，[num_tokens, dim] 的 2 维 Tensor。数据格式支持 ND。
- **weight** (Tensor)：必选参数，表示卷积权重，数据类型支持 bfloat16，维度为 [width, dim] 的 2 维 Tensor，数据格式支持 ND。
- **conv_state** (Tensor)：必选参数，表示状态缓存，数据类型支持 bfloat16，维度为 [num_cache_lines, state_len, dim] 的 3 维 Tensor，数据格式支持 ND。
- **bias** (Tensor)：可选参数，表示偏置向量，默认值为None，数据类型支持 bfloat16，维度为 [ dim ] 的 1 维 Tensor，数据格式支持 ND。
- **query_start_loc** (Tensor)：可选参数，表示varlen模式累积序列长度，默认值为None，数据类型支持 int32，维度为 [ batch + 1 ] 的 1 维 Tensor，数据格式支持 ND。
- **conv_state_indices** (Tensor)：可选参数，表示批次到缓存行的映射，默认值为None，数据类型支持 int32，维度为 [ batch ] 的 1 维 Tensor，数据格式支持 ND。
- **num_accepted_tokens** (Tensor)：可选参数，表示Speculative Decoding 场景接受的 token 数，默认值为None，数据类型支持 int32，维度为 [ batch ] 的 1 维 Tensor，数据格式支持 ND。
- **activation** (string)：可选参数，表示应用的激活函数，"silu"或None，默认值为None。
- **pad_slot_id** (int)：可选参数，表示无效的cache id，默认值为-1。

* 约束说明

- batch 支持 1~128。
- seqlen 支持 1~6。
- dim 需要满足整除 16，最大支持 12288。
- width 支持 2，3，4。
- state_len 要求大于等于 width - 1 + seqlen - 1。
- num_cache_lines：num_cache_lines >= batch。
- num_accepted_tokens 当 x=(batch, dim) 或者 width!=4 时为None。元素取值小于等于seqlen，大于等于1。
- query_start_loc 当输入x为[num_tokens, dim]时不能为None，满足query_start_loc[0] = 0，表示各batch的seqlens累计和，要求序列严格单调递增。
- conv_state_indices：元素值域范围为[0, num_cache_lines-1]。

### 3.4 输出严格要求

- **精度要求**: 要求与参考实现的精度误差满足浮点数精度要求

## 四、规则要求

- 算子实现需遵循Ascend C编程规范
- 需支持上述所有输入数据类型和维度组合
- 需正确处理可选输入为None的场景
