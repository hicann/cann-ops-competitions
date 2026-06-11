// Tiling数据结构 — Host侧填充后下发给kernel侧, 指导kernel的分块与分核策略.
// 结构体大小受限于tiling buffer (通常几KB), 因此尽可能用uint32_t紧凑存储.
//
// 两种计算模式:
//   mode=0  快速路径: 所有输入均为标量或与输出同形(无广播).
//            各核按总元素数均匀切分, 沿连续地址搬运, 无跨行地址计算开销.
//   mode=1  行广播路径: 存在非平凡广播(某个输入在某维大小为1而输出>1).
//            将输出视为 outerSize × lastDim 的二维矩阵, 按行分核;
//            行内沿lastDim向量化搬运, 行间通过stride跳转.
//
// 维度合并优化:
//   对相邻且广播签名(三个输入的broadcast flag组合)相同的维度进行合并,
//   减少外层循环层级, 增大单次搬运的连续长度, 减少地址计算与循环开销.

#pragma once

#include <cstdint>

// 合并后支持的最大外层维度数.
// 实际网络中绝大多数张量经合并后 <= 8维; 超出则TilingFunc返回失败.
#define ADDCMUL_MAX_DIM 8

struct AddcmulTilingData {
    // ---- 通用字段 ----
    uint32_t totalLength;   // 广播后输出张量的元素总数; 0 表示空张量, kernel直接返回
    uint32_t tileLength;    // UB单次处理的最大元素数 (已对齐到256, 由UB容量和dtype反推)
    uint32_t mode;          // 0 = 快速路径, 1 = 行广播路径

    // ---- mode=0 专用 ----
    // 标记各输入是否为全局标量 (numel == 1).
    // 若为标量, kernel用标量单元参与计算, 避免从GM搬运, 节省带宽和UB空间.
    uint32_t scalarIn;
    uint32_t scalarX1;
    uint32_t scalarX2;

    // ---- mode=1 专用 ----
    uint32_t lastDim;       // 最内层维度大小 — 向量化搬运方向 (连续读写)
    uint32_t outerSize;     // 行数 = totalLength / lastDim (所有外层维度的乘积)
    uint32_t outerCount;    // 合并后的外层维度个数 (即合并后rank - 1)

    // 最内层广播标记: 该输入在最内层维度大小为1 ∧ 输出>1 时为1
    // kernel据此决定: 最内层循环中对该输入使用标量复制, 还是逐元素搬运
    uint32_t lastBcIn;
    uint32_t lastBcX1;
    uint32_t lastBcX2;

    // 合并后外层各维度的大小
    uint32_t outerShape[ADDCMUL_MAX_DIM];

    // 每个输入沿合并后各外层维度的stride (步长).
    // 非广播维度: stride = 该维右侧(含自身)的原始维度乘积 (行优先).
    // 广播维度:   stride = 0, 使得坐标×stride累加时该项恒为0,
    //            该输入始终从基地址读取同一个元素, 实现广播效果.
    uint32_t strideIn[ADDCMUL_MAX_DIM];
    uint32_t strideX1[ADDCMUL_MAX_DIM];
    uint32_t strideX2[ADDCMUL_MAX_DIM];
};
