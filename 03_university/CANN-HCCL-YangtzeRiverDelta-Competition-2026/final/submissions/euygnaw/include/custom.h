#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <memory>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

// 一个Kernel只绑定同一CCU Die上的Channel。
struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t root = 0;

    // 4x1大数据链式流水专用静态参数。INVALID_VALUE_RANKID表示不存在该方向。
    uint32_t prevRank = INVALID_VALUE_RANKID;
    uint32_t nextRank = INVALID_VALUE_RANKID;
    uint32_t prevChannelIndex = INVALID_VALUE_RANKID;
    uint32_t nextChannelIndex = INVALID_VALUE_RANKID;
    uint64_t pipelineChunkBytes = 0;
    uint32_t pipelineChunkCount = 0;
    uint64_t pipelineLastChunkBytes = 0;
    uint64_t directDataBytes = 0;
};

using BroadcastCcuKernelArg = CcuKernelArgBase;

struct CcuKernelInfo {
    char kernelFuncName[64]{};
    void *kernelFunc = nullptr;
    void *kernelArg = nullptr;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(const std::shared_ptr<T> &arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

constexpr uint32_t BCAST_ALG_DIRECT = 0;
constexpr uint32_t BCAST_ALG_SAG_SPLIT = 1;
constexpr uint32_t BCAST_ALG_SAG_FUSED = 2;
constexpr uint32_t BCAST_ALG_CHAIN_PIPELINE = 3;
constexpr uint32_t BCAST_ALG_8P4_PREFETCH_SAG = 4;
constexpr uint32_t BCAST_ALG_CHAIN_ACK_512 = 5;

// 同一Die组复用一条Thread。小数据使用directHandle；大数据使用
// scatter/allGather两个Kernel；4x1单Die场景使用fusedHandle。
struct CcuKernelLaunchEntry {
    uint32_t dieId = 0;
    uint32_t threadIndex = 0;
    // 非root场景：该Die组是否包含到root的Channel。用于SAG阶段重叠调度。
    uint32_t containsRootChannel = 0;

    // 保存该Die Kernel绑定的对端顺序。v3.2.1用于为不同Kernel构造
    // 与LoadArg数量完全匹配的紧凑TaskArg。
    uint32_t channelCount = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE]{};

    CcuKernelHandle directHandle = 0;
    CcuKernelHandle scatterHandle = 0;
    CcuKernelHandle allGatherHandle = 0;
    CcuKernelHandle fusedHandle = 0;
    CcuKernelHandle chainHandle = 0;
};

struct AlgResourceCtx {
    uint32_t algorithm = BCAST_ALG_DIRECT;
    ThreadHandle ccuThread = 0;
    CommBuffer localBuffer{nullptr, 0};
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelLaunchEntry> ccuKernelEntries;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << algorithm;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernelEntries;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> algorithm;
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernelEntries;
    }
};

#endif // OPS_HCCL_CUSTOM_H