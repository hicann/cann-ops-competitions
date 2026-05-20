# 参赛全流程教程

本教程以完整的操作流程，从启动 Docker 环境到最终提交，逐步演示如何为 Mul 算子编写测试用例并提升覆盖率。建议在阅读本教程之前，先浏览[《Mul 算子架构分析》](https://gitcode.com/org/AI4SE/discussions/4)和[《Mul 算子测试用例分析》](https://gitcode.com/org/AI4SE/discussions/5)两篇参考文档，对算子的分层结构和测试用例的代码骨架有一个基本的了解。之后可以阅读[《Mul 算子精度分析教程》](https://gitcode.com/org/AI4SE/discussions/9)参考文档，加深对算子测试开发的理解。

## 第一步：启动实验环境

本次实验提供预配置的 Docker 镜像。首先确保本机已安装 Docker Desktop（Windows）或 Docker Engine（Linux），需要注意的是由于CANN模拟器限制，只能使用x86架构，Mac等arm架构处理器无法使用该Docker环境。

从 DockerHub 拉取镜像并启动容器：

```bash
docker pull yeren666/cann-ops-test:v1.0
docker run -it --name mul-test yeren666/cann-ops-test:v1.0
```

如果无法访问 DockerHub，可以下载离线包后导入：

```bash
docker load -i cann-ops-test-v1.0.tar.gz
docker run -it --name mul-test cann-ops-test:v1.0
```

启动后会进入容器的 bash 终端。进入工作目录：

```bash
cd /home/workspace/ops-math
```

此时 CANN 工具链、CPU 模拟器和 ops-math 源码都已就绪。

> **提示：** 如果不小心退出了容器，可以用 `docker start -i mul-test` 重新进入，之前的修改不会丢失。

检查toolkit  cat /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info

## 第二步：了解待测试的文件

先查看 Mul 算子的官方测试示例：

```bash
cat math/mul/examples/test_aclnn_mul.cpp
```

这个文件共 147 行，结构如下：

- 第 1-68 行：工具函数（`Init`、`CreateAclTensor` 等）
- 第 70-76 行：初始化设备和 stream
- 第 78-99 行：构造输入数据（`self={0,1,2,3,4,5,6,7}`，`other={1,1,1,2,2,2,3,3}`，shape 为 `{4,2}`，类型为 float32）
- 第 101-119 行：两段式调用 `aclnnMulGetWorkspaceSize` + `aclnnMul`
- 第 121-147 行：打印结果并释放资源

注意这个示例的两个不足：**只有一组 float32 的测试数据**，以及**只打印结果、没有验证结果是否正确**。这就是我们需要改进的地方。

## 第三步：运行原始示例，获取基线覆盖率

在修改代码之前，先运行一次原始示例，看看它能覆盖多少代码。

编译算子：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov
```

这一步大约需要 3-5 分钟。`--cov` 表示启用覆盖率统计。编译成功后会在 `build_out/` 下生成算子安装包。

安装算子包：

```bash
./build_out/cann-ops-math-custom_linux-x86_64.run
```

看到 `SUCCESS` 即安装成功。

运行测试：

```bash
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

等待约 30 秒，会看到类似输出：

```
result[0] is: 0.000000
result[1] is: 1.000000
result[2] is: 2.000000
result[3] is: 6.000000
result[4] is: 8.000000
result[5] is: 10.000000
result[6] is: 18.000000
result[7] is: 21.000000
run test_aclnn_mul, execute samples success
```

这说明算子执行成功。结果是 `self * other` 的逐元素乘积：`0*1=0, 1*1=1, 2*1=2, 3*2=6, ...`。

现在查看覆盖率。执行以下命令分别查看四个评测文件的覆盖情况：

```bash
gcov -b $(find build -name "aclnn_mul.cpp.gcda" | head -1) 2>&1 \
    | grep -A2 "File.*aclnn_mul.cpp" | head -3

gcov -b $(find build -name "mul.cpp.gcda" | grep "mul/op_api" | head -1) 2>&1 \
    | grep -A2 "File.*op_api/mul.cpp" | head -3

gcov -b $(find build -name "mul_tiling_arch35.cpp.gcda" | head -1) 2>&1 \
    | grep -A2 "File.*mul_tiling" | head -3

gcov -b $(find build -name "mul_infershape.cpp.gcda" | head -1) 2>&1 \
    | grep -A2 "File.*mul_infershape" | head -3
```

输出结果如下：

```
File '/home/workspace/ops-math/math/mul/op_api/aclnn_mul.cpp'
Lines executed:36.28% of 328

File '/home/workspace/ops-math/math/mul/op_api/mul.cpp'
Lines executed:57.69% of 52

File '/home/workspace/ops-math/math/mul/op_host/arch35/mul_tiling_arch35.cpp'
Lines executed:48.04% of 102

File '/home/workspace/ops-math/math/mul/op_host/mul_infershape.cpp'
Lines executed:0.00% of 10
```

这就是基线覆盖率。可以看到 `aclnn_mul.cpp` 只有 36%——因为我们只测了 float32 一种类型和 aclnnMul 一个 API，大量的类型提升分支、Muls/InplaceMul 路径都没有走到。

## 第四步：分析覆盖率不足的原因

基线覆盖率低，是因为原始示例只覆盖了一种场景。对照《Mul 算子架构分析》，可以发现以下未覆盖的维度：

**数据类型方面**，只测了 float32。`aclnn_mul.cpp` 中有针对混合类型（如 float16 * float32）、整数类型、bool 类型等不同的调度分支。`mul_tiling_arch35.cpp` 中有 16 种 dtype 组合的 tiling 策略映射表。每多测一种类型，就有可能触发新的代码路径。

**API 变体方面**，只调了 `aclnnMul`（tensor * tensor）。`aclnn_mul.cpp` 中还有 `aclnnMuls`（tensor * scalar）、`aclnnInplaceMul`、`aclnnInplaceMuls` 的实现，每个 API 的参数校验和调度逻辑不同，不调用就无法覆盖。

**Shape 方面**，只测了同 shape 的 `{4,2} * {4,2}`。广播场景（如 `{2,3} * {1,3}`）会走不同的 shape 处理路径。

## 第五步：修改测试用例

根据以上分析，我们在原始示例的基础上增加三个测试用例：一个使用 INT32 类型（覆盖不同的 tiling 路径），一个使用广播 shape（覆盖广播处理逻辑），一个调用 `aclnnMuls` API（覆盖 tensor*scalar 路径）。同时补充结果验证逻辑。

用编辑器打开测试文件：

```bash
vi math/mul/examples/test_aclnn_mul.cpp
```

> **提示：** 容器中可以使用 `vi` 编辑器。如果不熟悉 vi，也可以在容器外用 `docker cp` 将文件拷出来编辑后再拷回去：
>
> ```bash
> # 在容器外执行
> docker cp mul-test:/home/workspace/ops-math/math/mul/examples/test_aclnn_mul.cpp .
> # 用本地编辑器修改后拷回
> docker cp test_aclnn_mul.cpp mul-test:/home/workspace/ops-math/math/mul/examples/
> ```

以下是修改后的完整代码。保留了原始示例中的工具函数（`Init`、`CreateAclTensor`），新增了结果验证函数和多个测试用例：

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) shapeSize *= i;
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// ========== 新增：结果验证函数 ==========
bool AlmostEqual(double expected, double actual, double atol, double rtol) {
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual))
        return (expected > 0) == (actual > 0);
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

// ========== 用例 1：aclnnMul（tensor * tensor）==========
int RunMulTest(const char* name,
               const std::vector<float>& x1, const std::vector<int64_t>& shape,
               const std::vector<float>& x2,
               aclrtStream stream) {
    int64_t n = GetShapeSize(shape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    CreateAclTensor(x1, shape, &x1Dev, ACL_FLOAT, &x1T);
    CreateAclTensor(x2, shape, &x2Dev, ACL_FLOAT, &x2T);
    std::vector<float> outHost(n, 0);
    CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 验证：在 CPU 上用 double 精度计算期望值并比对
    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double expected = (double)x1[i] * (double)x2[i];
        if (!AlmostEqual(expected, outHost[i], 1e-5, 1e-5)) {
            LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHost[i]);
            failed++;
        }
    }
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

// ========== 用例 2：aclnnMuls（tensor * scalar）==========
int RunMulsTest(const char* name,
                const std::vector<float>& self, const std::vector<int64_t>& shape,
                float scalarVal,
                aclrtStream stream) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr;
    aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, ACL_FLOAT, &selfT);
    std::vector<float> outHost(n, 0);
    CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &outT);

    aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(selfT, scalar, outT, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: GetWorkspaceSize=%d\n", name, ret); return 1);

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMuls(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost.data(), n*sizeof(float), outDev, n*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double expected = (double)self[i] * (double)scalarVal;
        if (!AlmostEqual(expected, outHost[i], 1e-5, 1e-5)) {
            LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHost[i]);
            failed++;
        }
    }
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(scalar);
    aclrtFree(selfDev); aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int totalFailed = 0;

    // 原始用例：float32 同 shape
    totalFailed += RunMulTest("float32_basic",
        {0,1,2,3,4,5,6,7}, {4,2},
        {1,1,1,2,2,2,3,3}, stream);

    // 新增：含负数和零
    totalFailed += RunMulTest("float32_neg_zero",
        {-1.0f, 0.0f, 3.5f, -2.0f}, {2,2},
        {2.0f, 5.0f, -1.0f, 0.0f}, stream);

    // 新增：aclnnMuls（tensor * scalar，覆盖 aclnn_mul.cpp 中的 Muls 路径）
    totalFailed += RunMulsTest("float32_muls",
        {1.0f, 2.0f, 3.0f, 4.0f}, {2,2}, 2.5f, stream);

    LOG_PRINT("\n=== Summary: %d failed ===\n", totalFailed);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFailed;
}
```

代码的核心改动有三处：

1. 新增了 `AlmostEqual` 函数，用容差方式比较浮点数结果
2. 将测试逻辑封装为 `RunMulTest` 和 `RunMulsTest` 两个函数，每个函数内部都包含"计算期望值 → 比对结果 → 输出 PASS/FAIL"的完整验证流程
3. 在 `main` 中调用三组测试用例，覆盖了 float32 基本运算、含负数和零的边界值、以及 `aclnnMuls` API 变体

## 第六步：重新编译运行，观察覆盖率变化

修改完代码后，重新执行编译、安装、运行的完整流程：

```bash
# 编译
bash build.sh --pkg --soc=ascend950 --ops=mul --vendor_name=custom --cov

# 安装
./build_out/cann-ops-math-custom_linux-x86_64.run --quiet

# 运行
bash build.sh --run_example mul eager cust \
    --vendor_name=custom --simulator --soc=ascend950 --cov
```

在运行输出中应该能看到：

```
[PASS] float32_basic
[PASS] float32_neg_zero
[PASS] float32_muls

=== Summary: 0 failed ===
```

全部通过。现在查看覆盖率：

```bash
gcov -b $(find build -name "aclnn_mul.cpp.gcda" | head -1) 2>&1 \
    | grep -A2 "File.*aclnn_mul.cpp" | head -3
```

对比基线数据：

| 文件                    | 修改前 | 修改后     | 变化原因                              |
| ----------------------- | ------ | ---------- | ------------------------------------- |
| `aclnn_mul.cpp`         | 36.28% | **42.07%** | 新增 aclnnMuls 调用覆盖了 Muls 路径   |
| `mul_tiling_arch35.cpp` | 48.04% | **50.98%** | 不同 dtype/场景触发了新的 tiling 分支 |

仅增加了两个测试用例，`aclnn_mul.cpp` 的覆盖率就从 36% 提升到了 42%。按照同样的思路，继续增加更多类型（INT32、FLOAT16、INT8 等）和更多 API 变体（InplaceMul、InplaceMuls），覆盖率还可以进一步提升。

## 第七步：继续优化的方向

以下是一些可以继续尝试的方向，每个都有可能触发新的代码路径：

**更多数据类型。** `mul_tiling_arch35.cpp` 中有 16 种 dtype 组合的映射表，每种类型对应不同的 tiling 策略。尝试 INT32、FLOAT16、INT8、DOUBLE 等类型。需要注意：不同类型的 `CreateAclTensor` 模板参数和 `aclDataType` 枚举值不同，例如 INT32 对应 `ACL_INT32`，数据用 `std::vector<int32_t>`。

**InplaceMul / InplaceMuls API。** 这两个 API 在 `aclnn_mul.cpp` 中有独立的实现路径。调用方式与 Mul/Muls 类似，但不需要单独的 `out` tensor，结果直接覆盖 `selfRef`：

```cpp
aclnnInplaceMulGetWorkspaceSize(selfRefTensor, otherTensor, &wsSize, &executor);
aclnnInplaceMul(wsAddr, wsSize, executor, stream);
// 结果从 selfRef 的设备地址取回
```

**异常输入。** 传入 nullptr 或不支持的 dtype，验证返回错误码而非崩溃：

```cpp
ret = aclnnMulGetWorkspaceSize(nullptr, x2T, outT, &wsSize, &executor);
// 期望 ret != ACL_SUCCESS
```

**混合数据类型。** 例如 `x1` 为 FLOAT16，`x2` 为 FLOAT32，`out` 为 FLOAT32。这会触发 `aclnn_mul.cpp` 中的混合类型处理逻辑。

## 第八步：准备提交文件

当覆盖率达到满意的水平后，准备提交内容。

将测试源文件和 build 目录拷贝出来（在容器外执行）：

```bash
# 拷贝测试源文件
docker cp mul-test:/home/workspace/ops-math/math/mul/examples/test_aclnn_mul.cpp .

# 拷贝 build 目录（包含覆盖率数据）
docker cp mul-test:/home/workspace/ops-math/build ./build
```

组织提交目录：

```
<队名>/
├── test_aclnn_mul.cpp          # 测试用例源文件
├── build/                      # 包含 .gcda 和 .gcno 覆盖率数据
└── 测试报告.md                  # 测试设计说明（鼓励提交）
```

将以上目录打包为 zip 压缩包提交即可。

测试报告建议包含：测试策略说明（覆盖了哪些 dtype、shape、API 变体）、各用例的设计目标、覆盖率统计结果、以及对未覆盖代码的分析。

## 常见问题

**Q: 编译报错 `command not found` 或 `No such file`？**

检查环境变量是否加载。在容器内执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，或者退出后用 `docker start -i mul-test` 重新进入（`.bashrc` 会自动加载）。

**Q: 运行时卡住不动？**

CPU 模拟器执行较慢，每个测试用例大约需要 10 秒。如果测试用例较多，耐心等待即可。

**Q: 覆盖率没有变化？**

确认是否完整执行了"编译 → 安装算子包 → 运行"三步。只编译不安装或只安装不运行都不会更新覆盖率数据。

**Q: `mul_infershape.cpp` 覆盖率始终为 0%？**

这是正常现象。在 eager 模式（本次实验使用的模式）下，shape 推断由 op_api 层内部处理，不会调用注册的 `InferShape4Broadcast` 函数。该文件的覆盖率提升空间有限，建议将精力集中在 `aclnn_mul.cpp` 和 `mul_tiling_arch35.cpp` 上。

**Q: 如何查看某个文件中具体哪些行被覆盖了？**

`gcov` 执行后会在当前目录生成 `.gcov` 文件，可以直接查看。带 `#####` 标记的行表示未被执行：

```bash
gcov -b $(find build -name "aclnn_mul.cpp.gcda" | head -1)
cat aclnn_mul.cpp.gcov | head -50
```