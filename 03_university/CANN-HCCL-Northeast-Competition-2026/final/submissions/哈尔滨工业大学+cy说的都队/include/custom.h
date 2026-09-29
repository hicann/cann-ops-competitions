#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <memory>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// Scatter 算子的 CCU Kernel 静态参数：注册时传入，包含通道与拓扑信息
struct CcuScatterKernelArg : CcuKernelArgBase {
    uint32_t rankId = 0;
    uint32_t rankSize = 1;
    uint32_t root = 0;
};

// Scatter 算子单次 Kernel 下发（HcommCcuKernelLaunch）的任务参数布局，
// 以 uint64 数组形式传入，Kernel 内通过 LoadArg 按索引读取。
enum CcuScatterTaskArgIdx {
    CCU_SCATTER_TASK_ARG_INPUT = 0,        // 本 chunk 的输入基址（sendBuf chunk 偏移，仅 root 使用）
    CCU_SCATTER_TASK_ARG_OUTPUT = 1,       // 本 chunk 的输出基址（recvBuf chunk 偏移）
    CCU_SCATTER_TASK_ARG_TOKEN = 2,        // 本 rank 输出内存的访问 token
    CCU_SCATTER_TASK_ARG_SLICE_STRIDE = 3, // 相邻 rank 切片在 sendBuf 中的字节跨度（recvBytes）
    CCU_SCATTER_TASK_ARG_LENGTH = 4,       // 本 chunk 的字节数
    // ---- 中继（代理转发）用的中转区：本 rank 在 HCCL buffer 上划出的一段 ----
    CCU_SCATTER_TASK_ARG_STAGING = 5,       // 中转区基址
    CCU_SCATTER_TASK_ARG_STAGING_TOKEN = 6, // 中转区的访问 token
    CCU_SCATTER_TASK_ARG_PREFIX_LEN = 7, // 直发段长度：root 直接走 Clos 的那一段
    CCU_SCATTER_TASK_ARG_SUFFIX_LEN = 8, // 中继段长度：走 mesh → 助手 → Clos 的那一段
    CCU_SCATTER_TASK_ARG_NUM = 9,
};

// ---- 中继一次中继几个跨服对端 ----
// 【要加码只改 RELAY_DEST_NUM 这一个数。】
constexpr uint32_t RELAY_DEST_MAX = 8;
constexpr uint32_t RELAY_DEST_NUM = RELAY_DEST_MAX;

// ---- 中继 / 拉取式路径的切片阈值 ----
constexpr uint64_t MIN_RELAY_SLICE_BYTES = 8ULL * 1024 * 1024;

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;

    // ---- 中继（代理转发）计划：全 rank 由同一套规则算出，不依赖额外通信 ----
    uint32_t relayEnabled = 0;                       ///< 0 = 关闭（行为与 base371 逐指令一致）
    uint32_t relayCount = 0;                         ///< 实际中继了几片（<= RELAY_DEST_NUM）
    uint32_t relayDest[RELAY_DEST_MAX] = {0};        ///< 第 i 片被中继的跨服目的端
    uint32_t relayHelper[RELAY_DEST_MAX] = {0};      ///< 第 i 片由哪个本服 rank 代收代发
    uint32_t relayHelperRank = INVALID_VALUE_RANKID; ///< 本 rank 若是助手则为自身，否则 INVALID
    // 本服（mesh）那一组对应的 kernel 下标。助手要把它排到 Clos kernel 前面先跑。
    uint32_t relayMeshKernelIdx = 0;

    // ---- 中继量均摊 ----
    // 【每个】被中继的对端都走同一个比例 r = relayFracNum/relayFracDen：
    uint32_t relayFracNum = 0; ///< 每个中继对端走中继的比例分子
    uint32_t relayFracDen = 1; ///< 分母（恒 > 0）

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << relayEnabled;
        binaryStream << relayCount;
        binaryStream << relayDest;
        binaryStream << relayHelper;
        binaryStream << relayHelperRank;
        binaryStream << relayMeshKernelIdx;
        binaryStream << relayFracNum;
        binaryStream << relayFracDen;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> relayEnabled;
        binaryStream >> relayCount;
        binaryStream >> relayDest;
        binaryStream >> relayHelper;
        binaryStream >> relayHelperRank;
        binaryStream >> relayMeshKernelIdx;
        binaryStream >> relayFracNum;
        binaryStream >> relayFracDen;
    }
};

#endif // OPS_HCCL_CUSTOM_H
