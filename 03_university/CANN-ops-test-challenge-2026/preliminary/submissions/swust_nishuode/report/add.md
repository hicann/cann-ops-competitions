# Add 算子测试报告

## 1. 测试目标

本次测试面向 CANN ops-math 仓库中的 **Add 算子**，目标是在官方示例代码基础上补充端到端执行与结果校验测试，尽可能覆盖 Add 算子的主要执行路径，并提升题目要求范围内的代码覆盖率。

Add 算子的计算语义为：

```text
y = x1 + alpha * x2
```

其中 `alpha` 为缩放标量；当输入 shape 不一致时，按广播规则进行逐元素计算。

---

## 2. 测试范围

根据题目说明，本题覆盖率重点统计以下 4 个文件：

- `op_api/aclnn_add.cpp`
- `op_api/aclnn_add_v3.cpp`
- `op_api/add.cpp`
- `op_host/arch35/add_tiling_arch35.cpp`

因此本次测试设计重点围绕以下维度展开：

1. **API 变体覆盖**
   - `aclnnAdd`
   - `aclnnAdds`
   - `aclnnInplaceAdd`
   - `aclnnInplaceAdds`
   - `aclnnAddV3`
   - `aclnnInplaceAddV3`

2. **alpha 分支覆盖**
   - `alpha = 1`
   - `alpha = 0`
   - `alpha < 0`
   - 非整数浮点 `alpha`

3. **shape 覆盖**
   - 同 shape
   - broadcast
   - 更高 rank 的 broadcast
   - 较大 tensor

4. **dtype 覆盖**
   - `FLOAT`
   - `FLOAT16`
   - `BF16`
   - `INT32`
   - 混合 dtype（如 `FLOAT16 + FLOAT -> FLOAT`、`BF16 + FLOAT -> FLOAT`）

5. **异常路径覆盖**
   - `nullptr` 输入
   - 非法 dtype 组合
   - 错误输出 shape

---

## 3. 测试实现说明

### 3.1 实现方式

测试代码采用题目示例的官方骨架风格，使用正式头文件与两段式接口调用：

- 通过 `aclnnXXXGetWorkspaceSize(...)` 获取 workspace 大小与 executor
- 分配 workspace 后调用 `aclnnXXX(...)`
- 同步 stream
- 从 device 拷贝输出到 host
- 在 CPU 端计算期望值并进行数值比对

### 3.2 结果校验

所有执行型测试均包含结果校验逻辑。

- 对浮点类型：采用 `atol + rtol * |expected|` 方式比较
- 对整型类型：采用精确相等比较
- 对 `FLOAT16` / `BF16`：在 host 侧做对应精度量化后再比对

### 3.3 日志输出

每个测试用例输出：

- `[PASS]`：测试通过
- `[FAIL]`：测试失败

程序末尾输出总通过数、失败数以及整体结果。

---

## 4. 关键测试点

本次测试覆盖的关键用例如下：

### 4.1 Add / Adds
- `FLOAT` 同 shape 加法
- `FLOAT` + `alpha=0`
- 高 rank broadcast
- 较大 tensor
- `FLOAT16` / `BF16` 执行路径
- `INT32` 执行路径
- tensor + scalar 路径（`Adds`）

### 4.2 混合 dtype
- `FLOAT16 + FLOAT -> FLOAT`
- `FLOAT + FLOAT16 -> FLOAT`
- `BF16 + FLOAT -> FLOAT`
- `FLOAT + BF16 -> FLOAT`

### 4.3 原地接口
- `InplaceAdd`
- `InplaceAdds`
- `InplaceAddV3`

### 4.4 V3 接口
- `AddV3`：scalar + tensor
- `InplaceAddV3`：按头文件签名补充 V3 原地路径

### 4.5 异常输入
- `self == nullptr`
- 不支持的 dtype 组合
- 错误 out shape
- `AddV3` 的空 scalar 输入

---

## 5. 环境处理

根据题目说明，测试前对官方示例环境进行了如下处理：

1. 将 `test_aclnn_inplace_add.cpp` 替换为占位脚本，避免其在模拟器环境中崩溃，影响 `test_aclnn_add.cpp` 的执行。
2. 避免使用 `ACL_DOUBLE` 作为主要执行类型，优先使用 `ACL_FLOAT` 及题目支持的其他 dtype，减少模拟器下 AICPU 路径导致的问题。

---

## 6. 运行方式

### 6.1 编译

```bash
bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov
```

### 6.2 安装

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

### 6.3 执行测试

```bash
bash build.sh --run_example add eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

### 6.4 查看覆盖率

```bash
find build -name "*.gcda" | grep add

gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add.cpp.gcda
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/aclnn_add_v3.cpp.gcda
gcov -b build/math/abs/CMakeFiles/ophost_math_opapi_obj.dir/__/add/op_api/add.cpp.gcda
gcov -b build/math/add/CMakeFiles/ophost_math_tiling_obj.dir/op_host/arch35/add_tiling_arch35.cpp.gcda
```

---

## 7. 当前结果说明

当前测试代码已完成以下核心能力：

- 覆盖 Add 题目要求的主要 API 变体
- 包含真实执行与 CPU 端结果校验
- 包含多 dtype / 多 shape / 多 alpha / broadcast / 异常路径测试
- 已定位题目真正关心的 4 个覆盖率统计文件

最终是否达到 **90% 以上覆盖率**，仍以本地 `gcov` 实测结果为准；若仍有不足，需要继续根据 4 个目标文件的未覆盖分支补充更有针对性的 case。

---

## 8. 总结

本次测试不是仅调用接口做参数检查，而是补充了 **“真实执行 + 结果验证 + 多路径覆盖 + 异常路径覆盖”** 的完整测试逻辑。相比官方示例，这一版本更有利于覆盖：

- API 层调度逻辑
- V3 版本独立逻辑
- dtype / broadcast 相关分支
- tiling 分发逻辑

后续若要继续提升覆盖率，可重点针对 `gcov -b` 输出中未命中的分支继续添加专门测试用例。
