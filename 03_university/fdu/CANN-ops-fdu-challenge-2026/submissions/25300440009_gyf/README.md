## 个人信息
- 姓名：甘雨梵
- 学号：25300440009
- 联系邮箱：1653012886\@qq.com
- CANNJudge 账号：CtrlZ@\_@
## CANNJudge 提交说明
比赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
最终提交时间：2026-06-05 
三道题完成情况：
- Addcmul：
	所有测试点均已通过
- ClipByValue：
	所有测试点均已通过
- Lerp：
	所有测试点均已通过

## 算子实现简介
### Addcmul
实现公式：`y = input_data + x1 * x2 * value`
- **广播优化**：存在广播时，`block_size`限制为最小输入长度的约数，确保tile能完整覆盖广播维度
- **负载均衡**：`core_size`向下对齐，余数`core_remain`分配给最后一个核心。核数过多时自动减少，避免空转
- **类型特化**：int8类型提升为half计算后转回，使用转化为int16后左移8位再右移8位把前八位设置为符号位，模拟溢出；其他类型直接使用`Mul→Muls→Add`向量指令
- **双缓冲**：`BUFFER_NUM=2`实现数据加载与计算
### ClipByValue
实现公式：`y = clamp(x, min, max)`
- **实现方式**：`Mins(x, max)` → `Maxs(y, min)`，两条向量指令完成clip操作
- **分tile处理**：超UB容量时自动切分为多个tile
- **动态核数**：根据UB大小、数据类型和总量动态分配。NUM=4，buffer数较少
### Lerp
实现公式：`y = start + weight * (end - start)`
- **Tiling策略**：与Addcmul类似，`core_size`向下对齐到`8×ALIGN_NUM`，余数分配末核。buffer数量NUM=6
- **权重特化**：Host侧通过Attr读取weight，经TilingData传递至Kernel。Kernel侧判断weight是否为0或1，直接返回start或end，跳过计算
- **计算流程**：`Sub`计算差值 → `Muls`乘权重 → `Add`加回start
- **双缓冲**：`BUFFER_NUM=2`实现数据加载与计算
### 遇到的问题
- NUM的设定过度依赖于Kernel侧buffer的设定，导致有额外的处理空间被浪费，牺牲了性能但是换取了精确度
- Tiling过程中对于0的检查和处理
- Lerp 的 weight 从 Host 侧 float 经 TilingData 传递到 Kernel，转换为 half 时存在精度损失