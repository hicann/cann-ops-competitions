## 个⼈信息
- 姓名：涂家豪
- 学号：25303050257
- 联系邮箱：25303050257@m.fudan.edu.cn
- CANNJudge 账号：2662872838@qq.com Zzz


## CANNJudge 提交说明
⽐赛链接：https://cannjudge.cn/fdu-aiops/fdu-competition-2026
最终提交时间：2026/06/04 22:10:34
三道题完成情况（附提交成功截图）：
<div>
Addcmul：  
<img src="https://picui.ogmua.cn/s1/2026/06/05/6a22b077b094d.webp">  
ClipByValue：  
<img src="https://picui.ogmua.cn/s1/2026/06/05/6a22b077aa315.webp">
Lerp：  
<img src="https://picui.ogmua.cn/s1/2026/06/05/6a22b077ac38c.webp">
</div>

## 算⼦实现简介  
Addcmul:  
题目要求输入3个张量，`inputdata`，`x`，`y`和一个标量`value`，若x，y与inputdata形状不等需广播。  
在`TilingFunc`部分，获取AI Core的数量和UB内存大小。根据数据类型，预留出8个`Buffer`的空间（用于实现双缓冲队列`inQueueData`, `inQueueX1`, `inQueueX2`, `outQueueY`，每种2个。如果是`INT8`，还要额外预留2个`half`类型的转换缓存。  
总数据需32位向上对齐，采用 “大核与小核” 策略，尾块部分，平均分配给前面的几个`Core`。  
广播部分判断：对比`input_data`、`x1`、`x2`的总长度。如果它们长度不完全一致，将`needBroadcast`标志位设为1，指导`Kernel` 端执行不同的地址取址逻辑。  
在`kernal`侧初始化部分,确定当前Core是大核还是小核，计算出该Core需要处理的总数据量、偏移量，并为`Global Memory`和`Local Memory`申请空间。  
流水线处理部分，将`Core`的数据再次切分为多个`Tile`，通过一个循环逐步完成计算：  
`CopyIn`：从`Global Memory`将数据搬运到`Local Memory`中的输入队列。如果是`Broadcast`模式，通过取模运算动态确定读取起点。  
`Compute`：对于非`INT8`类型：调用矢量计算接口`Mul`、`Muls`、`Add`依次进行计算。
对于`INT8`类型：由于`Muls`接口限制，将其`Cast`转换至`half`进行乘加运算，最后再通过平移和`Cast`操作转回`INT8`。  
`CopyOut`：将计算完毕的`Local Memory`数据搬运回`Global Memory`。

ClipByValue:  
题目一共输入三个参数，一个`Tensor x`，两个标量`min`和`max`，最后输出一个和`x`形状、数据类型相同的`Tensor y`。对于两个张量，采用出入队列的方法，实现数据在host侧和kernal侧的传输，与上一题一致。  
对于两个标量，将其地址写入`tiling_data`结构体，实现数据从host侧到kernal侧的传输。  
在`TilingFunc`部分，传输数据块32位对齐，处理尾块时采用大小核设计，将尾块平均分给工作的核处理，同上。  
初始化和流水线处理部分，基本思路与上一题基本类似。只是`Compute`部分，调用的`min`和`max`函数接口不同。

Lerp:  
题目一共输入三个参数，两个`Tensor` `start`和`end`，一个标量`weight`，最后输出形状、数据类型与`start`相同的`Tensor y`。  
数据传输方式与`ClipByValue`基本一致。
在`TilingFunc`部分同样采用大小核设计。  
初始化和流水线处理部分，基本思路与上一题基本类似。只是`Compute`部分，调用的`Sub`,`Muls`和`Add`函数接口不同。