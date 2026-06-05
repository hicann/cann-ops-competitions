# 算子说明书：BatchToSpace

## 1 算子说明

算子功能：`BatchToSpace` 是一个数组重排算子，用于将批处理维度中的数据块重新排列到空间维度上。该操作是 `SpaceToBatch` 的逆操作，通常用于神经网络中的反卷积或上采样操作后的数据重排。通过将批处理维度中的小块数据按照指定块大小放置到空间位置的对应块中，实现从批次维度到空间维度的数据转换。

## 2 输入输出规格

- **输入规格**

   | 输入 |类型 | 形状 | 数据类型 | 含义 |
   |------|------|------|----------|------|
   | x |张量 |`[batch, height, width, depth]` | float/float16 | 输入张量，待重排的批处理数据 |
   | crops |属性| `[4]` | int32 | 裁剪参数，格式为 `[crop_top, crop_bottom, crop_left, crop_right]`，标杆内部reshape为[2,2]后使用 |
   | block_size | 属性 |标量 | int32 | 块大小，用于将批次维度中的数据重新排列到空间维度 |

- **输出规格**

   | 输出 | 类型|形状 | 数据类型 | 含义 |
   |------|------|------|----------|------|
   | y |张量 |`[batch / (block_size^2), height × block_size - crop_top - crop_bottom, width × block_size - crop_left - crop_right, depth]` | float/float16 | 输出张量，重排并裁剪后的结果 |

**形状约束：**
- `batch` 必须能被 `block_size * block_size` 整除
- `crops` 为1维数组，长度为4，标杆内部reshape为[2,2]，满足 `crops[i][0] + crops[i][1] < block_size * input_shape[i+1]`（其中 i=0,1 对应空间维度）
- 输出张量的空间维度计算：
  - `out_height = height * block_size - crops[0][0] - crops[0][1]`
  - `out_width = width * block_size - crops[1][0] - crops[1][1]`
- 输出批处理维度计算：`out_batch = batch // (block_size * block_size)`

## 3 功能示例

```python
import tensorflow as tf
import numpy as np
def impl(x, crops, block_size):
    crops = np.array(crops, dtype=np.int32).reshape(2, 2)
    return tf.raw_ops.BatchToSpace(input=x, crops=crops, block_size=block_size).numpy()
# 示例1：基本用法（无裁剪）
# 输入形状：[4, 2, 2, 1]，block_size=2
x = tf.reshape(tf.range(16, dtype=tf.float32), [4, 2, 2, 1]).numpy()
crops = np.array([0, 0, 0, 0], dtype=np.int32)
block_size = 2
y = impl(x, crops, block_size)
# 输出形状：[1, 4, 4, 1]
# 结果：将4个2x2块重排成1个4x4矩阵
# [[ 0,  1,  4,  5],
#  [ 2,  3,  6,  7],
#  [ 8,  9, 12, 13],
#  [10, 11, 14, 15]]
# 示例2：带裁剪的用法
x = tf.reshape(tf.range(16, dtype=tf.float32), [4, 2, 2, 1]).numpy()
crops = np.array([1, 1, 1, 1], dtype=np.int32)  # 每个维度裁剪1行/列
block_size = 2
y = impl(x, crops, block_size)
# 输出形状：[1, 2, 2, 1]（4x4裁剪后得到2x2）
# 结果：保留中心2x2区域
# [[ 3,  6],
#  [11, 14]]
# 示例3：非均匀裁剪
x = tf.reshape(tf.range(16, dtype=tf.float32), [4, 2, 2, 1]).numpy()
crops = np.array([0, 2, 1, 1], dtype=np.int32)  # 底部裁剪2行，左右各裁剪1列
block_size = 2
y = impl(x, crops, block_size)
# 输出形状：[1, 2, 2, 1]（高度：2*2-0-2=2，宽度：2*2-1-1=2）
# 结果：保留特定区域
```

**算子逻辑说明**：
 
1. **数据重排阶段**：   
   - 将输入 x 的形状从 `[batch, height, width, depth]` 重塑为 `[block_size, block_size, batch // (block_size^2), height, width, depth]`
   - 通过转置操作将块数据移动到空间位置：`[batch // (block_size^2), height, block_size, width, block_size, depth]`
   - 重塑为 `[batch // (block_size^2), height × block_size, width × block_size, depth]`
2. **裁剪阶段**：   
   - crops 从1D crop_top, crop_bottom, crop_left, crop_right 标杆内部reshape为2D `[crop_top, crop_bottom, crop_left, crop_right]`
   - 根据 crops 参数裁剪掉四周的边界值
   - 裁剪后输出：`[:, crops[0][0]:height × block_size - crops[0][1], crops[1][0]:width × block_size - crops[1][1], :]`
   
## 4 任务要求
- 针对不同 Data type 和不同 shape 维度，设计算子逻辑，保证算子精度正确
- 充分发挥系统带宽能力，算子性能更优
- 探索输入tensor的切分方式，找到不同输入shape场景下的最优解