# 8月社区任务 - ELU 算子开发任务书

## 1. 任务概述

### 1.1 开发目标
基于 **Ascend C C API** 接口，开发纯 **Vector-Core** 算子 `ELU` （Exponential Linear Unit）。
*   **核心约束**：全程**禁止**调用 Cube-Core 单元。
*   **技术挑战**：算子内部包含指数运算，指令链路较长。开发者需合理排布向量运算指令及访存逻辑，以优化性能。
*   **数学公式**：
    $$
    ELU(x) = 
    \begin{cases} 
    scale \cdot x, & \text{if } x > 0 \\
    \alpha\cdot \operatorname{scale}\cdot\left(e^ {x \cdot\operatorname{inputScale}} - 1\right), & \text{if } x \le 0 
    \end{cases}
    $$
    > *注：α 表示超参数*

### 1.2 易用性反馈
*   在开发过程中，请总结遇到的易用性问题（如：文档理解歧义、接口Bug、API设计不合理等）。
*   请将这些问题以 **Issue** 形式提交至 `asc-devkit` 开源仓库（Issue 提交规范见参考资料（https://gitcode.com/cann/asc-devkit/wiki/05_%E7%A0%94%E5%8F%91%E5%8D%8F%E4%BD%9C%E8%A7%84%E8%8C%83.md ））。

### 1.3 交付流程
完成代码编写、单元测试、性能调优后，提交 **PR** 到仓库 `master` 分支的 `examples/02_simd_c_api/03_c_api/02_reg_vector_compute` 目录下（PR 提交规范见参考资料（https://gitcode.com/cann/asc-devkit/wiki/05_%E7%A0%94%E5%8F%91%E5%8D%8F%E4%BD%9C%E8%A7%84%E8%8C%83.md ）），并附带完整的测试报告。

---

## 2. 核心开发要求

### 2.1 功能实现
1.  **数据类型支持**：必须支持 `float` 、 `float16` 两种数据类型。
2.  **泛化能力**：必须实现算子的泛化功能，支持各类合法输入场景。验收阶段将采用泛化数据进行测试。
3.  **对齐标准**：功能逻辑需与 Ascend C 官方算子库（https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/atlasoxol_09_0053.html ）中的 `elu` 实现完全对齐。

### 2.2 参数定义

| 参数名 | 类型 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | ---- |
| `dst` | Output | 目的操作数的起始地址 | FLOAT / FLOAT16 | ND |
| `src` | Input | 源操作数的起始地址 | FLOAT / FLOAT16 | ND |
| `alpha` | Input | 表示ELU激活函数的激活系数，公式中的α | FLOAT | ND |
| `scale` | Input | 表示ELU激活函数的缩放系数，公式中的`scale` | FLOAT | ND |
| `input_scale` | Input | 表示ELU激活函数的输入的缩放系数，公式中的`inputScale` | FLOAT | ND |

### 2.3 约束限制
*    **禁止 Host 侧串行计算**：严禁在Host侧使用 `for` 循环逐个元素运算，所有计算必须在 Device 侧批量完成。
*    **内存安全**：需合理分配 Local Memory，确保无缓冲区溢出风险。

以上约束所有case均需满足。

## 3. 自验证计划

测试硬件和软件要求：
- **适配硬件**：Ascend 950
- **CANN 版本**：CANN 9.0.0 ~ CANN 9.1.0

计划基于官方自验证样例（https://gitcode.com/cann/asc-devkit/tree/master/examples/02_simd_c_api/00_introduction/01_add/c_api_async_add ）完成修改：
1. 将`c_api_add.asc`中当前占位调用：

     ```cpp
     add_custom<<<numBlocks, 0, stream>>>(xDevice, yDevice, zDevice);
     ```
    替换为：
    ```cpp
    elu_custom<<<numBlocks, 0, stream>>>(...);
    ```
2. 以`script/gen_data.py`的形式验证数据精度是否达标：

    开发者可以参考现有样例（https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/03_basic_api/02_reg_vector_compute/abs/scripts ）中的验证方案。将`np.abs(x)`更换为`elu`接口即可。

3. 精度测试

    - 算子计算精度需满足 生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md#2-%E8%AF%AF%E5%B7%AE%E6%8C%87%E6%A0%87%E4%B8%8E%E9%80%9A%E8%BF%87%E6%A0%87%E5%87%86 ）。
    - 测试计划覆盖：
        - 数据大小范围：[-100,  100]
        - 数据Shape：
        ```cpp
        shape = 1
        shape = 32
        shape = 1024
        shape = 2048
        ```
4. 性能要求

    无

## 4. 交付件清单

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 标题格式：**[Requirement\|需求建议]: 【社区任务】xxx设计文档评审申请**；<br> 2. 设计文档模板：https://gitcode.com/cann/asc-devkit/issues/1222 ；<br> 3. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md ；<br> 4. 通过评审后的设计文档还需要在这个仓通过issue提交：https://gitcode.com/cann/asc-devkit/issues|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要清晰指导算子的使用方法、编译步骤及测试命令，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
|  4  |  易用性issue链接  |   若在开发中发现易用性问题，需附上对应的 Issue 链接，并上传至归档处（https://docs.qq.com/sheet/DS2RBdUdNc1pvRVps?tab=rx2vr7 ）  |
| 5 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 算子目录下需要提供readme文件，参考：https://gitcode.com/cann/ops-transformer/blob/master/attention/chunk_gated_delta_rule/README.md <br> |

![邀请示意](./pics/invite.jpeg)

## 5. 参考资料与环境

### 5.1 参考资料
1.  **文档**：Ascend C 算子开发文档（https://www.hiascend.com/document/detail/zh/canncommercial/850/funcguide/funcguide/funcguide_03_0001.html ）、算子开发接口文档（https://www.hiascend.com/document/detail/zh/canncommercial/850/funcguide/funcguide/funcguide_03_0002.html ）
2.  **课程**：Ascend C 在线课程（https://www.hiascend.com/edu/courses?activeTab=%E5%88%9D%E7%BA%A7,%E4%B8%AD%E7%BA%A7,%E9%AB%98%E7%BA%A7 ）
3.  **参考样例**：
    *   https://gitcode.com/cann/asc-devkit/tree/master/examples/02_simd_c_api/00_introduction/01_add/c_api_async_add
    *   https://gitcode.com/cann/ops-nn/blob/master/activation/elu/README.md
    *   https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/03_basic_api/02_reg_vector_compute/abs/scripts
4.  **PR/Issue提交规范**：
    - PR提交规范：https://gitcode.com/cann/asc-devkit/wiki/05_%E7%A0%94%E5%8F%91%E5%8D%8F%E4%BD%9C%E8%A7%84%E8%8C%83.md
    - Issue提交规范：请以"【AscendC CAPI社区任务】xxx" 格式提交你的Issue，Issue内容清晰完整。

### 5.2 环境获取

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend ；
   - 如果是新用户，在申请权限的时候需要备注使用的算力类型（A2、A3或者950）；
   - 如果是老用户且需要使用950算力，需要向昇腾CANN小助手反馈账号名（个人中心->基本信息，如下图所示），后台会添加账号至950使用白名单。

   ![环境截图](./pics/zaixiankaifa1.png)  
   ![账号名](./pics/account.png)  

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

   ![环境截图](./pics/yunkaifa.png)

3. 如需额外环境资源，请联系昇腾CANN小助手。

### 5.3 特别注意事项
1. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
2. 开发前请务必阅读【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39 。