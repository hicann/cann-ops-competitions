## 团队信息

- 团队名称：didudidu
- 所属单位：南京航空航天大学
- 联系人：戎一飞
- 联系邮箱：2271047501@qq.com

## 算子信息

- 算子名称：batch_to_space
- 实现路径：code/batch_to_space

## 算子实现介绍

本实现将 BatchToSpace 拆分为 host 侧 shape 校验、tiling 选择与 kernel 侧多路径调度。
