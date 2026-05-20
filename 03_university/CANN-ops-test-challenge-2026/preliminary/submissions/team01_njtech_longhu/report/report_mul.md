# Mul 算子测试报告

## 覆盖率结果

| 文件 | 行覆盖率 | 有效行数 |
|------|---------|---------|
| `op_api/aclnn_mul.cpp` | **80.18%** | 328 |
| `op_api/mul.cpp` | **80.77%** | 52 |
| `op_host/arch35/mul_tiling_arch35.cpp` | **89.22%** | 102 |

## 测试设计思路

### 1. Tiling 层 dtype 组合覆盖（16种）

`mul_tiling_arch35.cpp` 中的 `DTYPE_MAP` 包含 16 种 dtype 组合，每种对应不同的 tiling 函数分支：

| 测试用例 | dtype 组合 | 覆盖分支 |
|---------|-----------|---------|
| `tiling_float_float_float` | FLOAT×FLOAT→FLOAT | `MulOp<float>` |
| `tiling_fp16_fp16_fp16` | FP16×FP16→FP16 | `MulXfp16Op<half>` |
| `tiling_bf16_bf16_bf16` | BF16×BF16→BF16 | `MulXfp16Op<bfloat16_t>` |
| `tiling_int8_int8_int8` | INT8×INT8→INT8 | `MulInt8Op` |
| `tiling_uint8_uint8_uint8` | UINT8×UINT8→UINT8 | `MulUint8Op` |
| `tiling_bool_bool_bool` | BOOL×BOOL→BOOL | `MulBoolOp` |
| `tiling_int16_int16_int16` | INT16×INT16→INT16 | `MulOp<int16_t>` |
| `tiling_int32_int32_int32` | INT32×INT32→INT32 | `MulOp<int32_t>` |
| `tiling_int64_int64_int64` | INT64×INT64→INT64 | `MulOp<int64_t>` |
| `tiling_double_double_double` | DOUBLE×DOUBLE→DOUBLE | `MulDoubleOp` |
| `tiling_complex64_complex64_complex64` | COMPLEX64×COMPLEX64→COMPLEX64 | `MulOp<int64_t>` |
| `tiling_fp16_fp32_fp32` | FP16×FLOAT→FLOAT | `MulMixFpOp<half,float,float>` |
| `tiling_fp32_fp16_fp32` | FLOAT×FP16→FLOAT | `MulMixFpOp<float,half,float>` |
| `tiling_bf16_fp32_fp32` | BF16×FLOAT→FLOAT | `MulMixFpOp<bfloat16_t,float,float>` |
| `tiling_fp32_bf16_fp32` | FLOAT×BF16→FLOAT | `MulMixFpOp<float,bfloat16_t,float>` |
| `tiling_complex128_complex128_complex128` | COMPLEX128×COMPLEX128→COMPLEX128 | AiCpu 路径 |

### 2. op_api 层关键分支覆盖

**aclnnMul（tensor×tensor）：**
- `isMixDataType=true` 路径：fp16×fp32、bf16×fp32
- `isMixDataType=false` + 同dtype + `IsMulSupportNonContiguous=true`：float×float
- `isMixDataType=false` + 需要 Cast：int32×int32
- 空 tensor（`IsEmpty` 分支）

**aclnnMuls（tensor×scalar）：**
- `canUseMuls=true`：bf16 tensor × float scalar、fp16 tensor × float scalar
- `canUseMuls=false` + `IsMulSupportNonContiguous=true`：float × float scalar
- `canUseMuls=false` + 需要 Cast：int32 × int scalar

**aclnnInplaceMul：**
- `IsRegBase() && isMixDataType`：fp16×fp32
- 普通路径：float×float

**aclnnInplaceMuls：**
- `canUseMuls=true`：bf16 × float scalar
- 普通路径：float × float scalar

### 3. Shape 覆盖

- 广播：`[2,3]×[3]`、`[1]×[4,4]`、`[4,1]×[1,4]`
- 1D、4D tensor
- 空 tensor（shape 含 0 维）
- 大 tensor（64×64）

### 4. 边界值

- 零值、负数、float 极大/极小值
- NaN、±Inf

### 5. 异常输入（参数校验分支）

- nullptr self / other / out
- 不兼容广播的 shape
- nullptr scalar

### 6. 结果验证

所有 float/int32 类型测试均包含 CPU 端期望值计算和数值比对：
- FLOAT32：atol=1e-5, rtol=1e-5
- FLOAT16：atol=1e-3, rtol=1e-3
- BF16：atol=1e-2, rtol=1e-2
- INT32：精确匹配
