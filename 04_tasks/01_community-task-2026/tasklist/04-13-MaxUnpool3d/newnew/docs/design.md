 # aclnnMaxUnpool3d 算子设计文档
 	 
 	 ## 一、 需求背景
 	 
 	 ### 1.1 需求来源
 	 
 	 通过社区任务完成开源仓算子贡献的需求，补充完善 Ascend C 算子库。为现有的 `aclnnMaxUnpool3d` 算子，self输入增加对bf16数据类型的支持，indices增加对int32的支持。
 	 
 	 ### 1.2 背景介绍
    当前支持情况
      - 算子功能：aclnnMaxPool在3d的逆运算，由outputSize决定outRef的D、H、W轴大小，并根据indices索引在outRef中填入self的元素值，其余位置都设置为0。
      - 计算公式：
      - 输入为4维，各维度分别为N、D、H、W，（其中N（Batch）表示批量大小、H（Height）表示特征图高度、W（Width）表示特征图宽度、D（Depth）表示特征图深度）时：

         $$
         outRef[N][indices[N][i]] = self[N][i]
         $$
      - 输入为5维，各维度分别为N、C、D、H、W，（其中C（Channels）表示特征图通道）时：

         $$
         outRef[N][C][indices[N][C][i]] = self[N][C][i]
         $$

         其中outRef、indices和self是最后两轴合为一轴，经过reshape得到的，i∈[0,D*H*W)。
    
      - 当前算子定义

- **参数说明：**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的self，表示待转换的目标张量。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与outRef的数据类型一致。</li>
          <li>shape与indices保持一致。</li>
          <li>当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8、DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的indices，表示输入self的元素在输出结果中的索引位置。</td>
      <td class="tg-0pky">
        <ul>
          <li>shape与self保持一致。</li>
          <li>当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。</li>
        </ul>
      </td>
      <td class="tg-0pky">INT64、INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示输出结果在D、H和W维度上的空间大小。</td>
      <td class="tg-0pky">size大小为3，三个元素乘积值需大于等于self在D、H和W维度上的size乘积值。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">stride（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示最大池化窗口在D、H和W维度上的步长大小。</td>
      <td class="tg-0pky">预留参数，当前版本不参与计算，需要传入size大小为3、值大于0的Host侧aclIntArray。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">padding（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示最大池化窗口在D、H和W维度上的填充值。</td>
      <td class="tg-0pky">预留参数，当前版本不参与计算，需要传入size大小为3的Host侧aclIntArray。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">outRef（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">公式中的outRef。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与self的数据类型一致。</li>
          <li>当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8、DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
  </tbody></table>
  
         现有aclnnMaxUnpool3d算子，已支持任务书要求的“Atlas A2 训练系列产品 / Atlas A3 系列产品”，数据类型不支持bf16。

      - 算子功能实现
         aclnnMaxUnpool3d 底层调用的是scatter_elements_v2算子。代码仓的scatter_elements_v2算子已经支持bf16，且indices已支持int32，int64，可运行在“Atlas A2 训练系列产品 / Atlas A3 系列产品”上。
 	 
 	 ## 二、 需求分析
 	 
 	 ### 2.1 外部组件依赖
 	 
 	 * 不涉及外部组件依赖
 	 
 	 ### 2.2 内部适配模块
 	 
 	 * 适配aclnn接口

    ### 2.3 算子原型
  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的self，表示待转换的目标张量。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与outRef的数据类型一致。</li>
          <li>shape与indices保持一致。</li>
          <li>当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16、INT16、INT32、INT64、INT8、UINT8、DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的indices，表示输入self的元素在输出结果中的索引位置。</td>
      <td class="tg-0pky">
        <ul>
          <li>shape与self保持一致。</li>
          <li>当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。</li>
        </ul>
      </td>
      <td class="tg-0pky">INT64、INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示输出结果在D、H和W维度上的空间大小。</td>
      <td class="tg-0pky">size大小为3，三个元素乘积值需大于等于self在D、H和W维度上的size乘积值。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">stride（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示最大池化窗口在D、H和W维度上的步长大小。</td>
      <td class="tg-0pky">预留参数，当前版本不参与计算，需要传入size大小为3、值大于0的Host侧aclIntArray。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">padding（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示最大池化窗口在D、H和W维度上的填充值。</td>
      <td class="tg-0pky">预留参数，当前版本不参与计算，需要传入size大小为3的Host侧aclIntArray。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">outRef（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">公式中的outRef。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与self的数据类型一致。</li>
          <li>当维度为4时，各维度依次表示N、D、H、W，当维度为5时，各维度依次表示N、C、D、H、W。</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8、DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
  </tbody></table>

 	 ## 三、 需求详细设计
 	 
 	 ### 3.1 使能方式
 	 
 	 | 上层框架 | 涉及的框架勾选 |
 	 | --- | --- |
 	 | TF训练/推理 |  |
 	 | Pytorch训练/推理 |  |
 	 | ATC推理 |  |
 	 | Aclnn直调 | ✅ |
 	 
 	 ### 3.2 详细设计
 	 本次任务，是在现有代码上进行修改，下面分别从op_api, op_host, op_kernel三个方面进行描述。
 	 #### 3.2.1 op_api：增加bfp16数据类型支持。
 		 
 	 #### 3.3.2 op_host：无需修改。
      底层scatter_elements算子已经支持bfp16，indices已经支持int32。

 	 #### 3.3.3 op_kernel：无需修改。    
      底层scatter_elements算子已经支持。
 	 
 	 #### 3.3.3 Ascend C 侧运行流程图
 	  ```mermaid
 	 graph TD
 	     %% Host ACLNN API层
 	     A[外部调用 aclnnMaxUnpool3dGetWorkspaceSize] --> B[入参合法性校验<br/>dtype/shape/空指针/边界校验]
 	     B --> C[计算并返回所需Workspace大小]
 	     C --> D[调用 aclnnMaxUnpool3d 正式执行入口]
 	     D --> E[Tiling参数推导 + 数据类型分支分发]
 	     E --> F[构造Kernel执行描述、下发任务至NPU Device]
 	 
 	     %% Ascend C Device 核侧标准流水线
 	     F --> G[Global Memory -> Local/UB搬运 CopyIn]
 	     G --> I[根据indices将self写入outRef]
 	     I --> J[UB -> Global Memory 结果CopyOut]
 	     J --> K[任务完成 Host侧同步返回结果]
 	 ```
 	 
 	 ### 3.3 支持硬件
 	 
 	 * **Atlas A2 训练系列产品 / Atlas A3 系列产品** (对应 `ascend910b` 及 `ascend910_93`)。
 	 
 	 ## 四、 验收及可维可测标准
 	 
 	 ### 4.1 精度与性能标准
 	 
 	 * **精度标准**：
 	 * 算子计算精度需满足 AscendOpTest 工具默认阈值。
 	 
 	 
 	 * **性能标准**：
    * 性能和fp16数据类型持平。Indices int32与int64持平。
 	 
 	 ## 五、 变更文件清单
 	 
 	 | 文件路径 | 变更类型 | 变更说明 |
 	 | --- | --- | --- |
 	 | `op_api/aclnn_max_unpool3d.cpp` | 修改 | 新增支持类型。|
 	 | `docs/aclnnMaxUnpool3d.md` | 修改 | 新增支持类型，修改适配硬件信息。 |
 	 | `examples/test_aclnn_max_unpool3d.cpp` | 新增 | 测试aclnnMaxUnpool3d接口。 |
