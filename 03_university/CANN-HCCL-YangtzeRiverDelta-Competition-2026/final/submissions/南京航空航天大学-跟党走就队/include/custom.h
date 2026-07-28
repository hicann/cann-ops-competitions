#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

constexpr uint32_t MAX_DIE_KERNEL_NUM = 4;
constexpr uint32_t BCAST_PIPELINE_CHUNK_NUM = 24;
constexpr uint32_t ALG_RESOURCE_CTX_MAGIC = 0x42564335U; // "BVC5"
constexpr uint32_t ALG_RESOURCE_CTX_VERSION = 53U;

/*
 * EngineCtx热路径使用固定容量数组，避免每次ExecOp反序列化时为两个
 * std::vector重复申请堆内存。接口刻意保持vector常用子集，Host代码无需
 * 引入额外分支。
 */
template <typename T, uint32_t Capacity>
struct FixedArray {
    T values[Capacity]{};
    uint32_t count = 0;

    size_t size() const
    {
        return static_cast<size_t>(count);
    }

    bool empty() const
    {
        return count == 0;
    }

    void clear()
    {
        count = 0;
    }

    void reserve(size_t)
    {
    }

    void resize(size_t newSize)
    {
        count = static_cast<uint32_t>(newSize);
    }

    void push_back(const T &value)
    {
        values[count++] = value;
    }

    T &operator[](size_t index)
    {
        return values[index];
    }

    const T &operator[](size_t index) const
    {
        return values[index];
    }

    T *begin()
    {
        return values;
    }

    T *end()
    {
        return values + count;
    }

    const T *begin() const
    {
        return values;
    }

    const T *end() const
    {
        return values + count;
    }
};

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
};

using BroadcastCcuKernelArg = CcuKernelArgBase;

// Direct消息长度在Kernel注册时固化，运行时只下发base和token。
struct BroadcastDirectCcuKernelArg : CcuKernelArgBase {
    uint64_t directDataBytes = 0;
};

// 24片静态表只属于4x1链式Kernel，避免扩大Direct/SAG/8+4的公共KernelArg。
struct BroadcastChainCcuKernelArg : CcuKernelArgBase {
    uint64_t pipelineChunkOffsets[BCAST_PIPELINE_CHUNK_NUM]{};
    uint64_t pipelineChunkSizes[BCAST_PIPELINE_CHUNK_NUM]{};
};

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

// 同一Die组复用一条Thread。小数据使用directHandle；大数据使用
// scatter/allGather两个Kernel；4x1单Die场景使用chainHandle。
struct CcuKernelLaunchEntry {
    uint32_t dieId = 0;
    uint32_t threadIndex = 0;
    // 非root场景：该Die组是否包含到root的Channel。用于SAG阶段重叠调度。
    uint32_t containsRootChannel = 0;

    // 保存该Die Kernel绑定的对端顺序，用于构造与LoadArg严格匹配的TaskArg。
    uint32_t channelCount = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE]{};

    CcuKernelHandle directHandle = 0;
    CcuKernelHandle scatterHandle = 0;
    CcuKernelHandle allGatherHandle = 0;
    CcuKernelHandle fusedHandle = 0;
    CcuKernelHandle chainHandle = 0;
};

struct AlgResourceCtx {
    uint32_t magic = ALG_RESOURCE_CTX_MAGIC;
    uint32_t version = ALG_RESOURCE_CTX_VERSION;
    uint32_t algorithm = BCAST_ALG_DIRECT;
    uint32_t reserved = 0;
    ThreadHandle ccuThread = 0;
    CommBuffer localBuffer{nullptr, 0};
    FixedArray<ThreadHandle, MAX_DIE_KERNEL_NUM> threads;
    FixedArray<CcuKernelLaunchEntry, MAX_DIE_KERNEL_NUM> ccuKernelEntries;

    std::vector<char> Serialize() const
    {
        std::vector<char> result(sizeof(AlgResourceCtx));
        std::memcpy(result.data(), this, sizeof(AlgResourceCtx));
        return result;
    }

    bool DeSerialize(const void *data, uint64_t dataSize)
    {
        if (data == nullptr || dataSize != sizeof(AlgResourceCtx)) {
            return false;
        }
        std::memcpy(this, data, sizeof(AlgResourceCtx));
        return magic == ALG_RESOURCE_CTX_MAGIC &&
            version == ALG_RESOURCE_CTX_VERSION &&
            threads.count > 0 && threads.count <= MAX_DIE_KERNEL_NUM &&
            ccuKernelEntries.count > 0 &&
            ccuKernelEntries.count <= MAX_DIE_KERNEL_NUM;
    }
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "AlgResourceCtx must remain trivially copyable");

#endif // OPS_HCCL_CUSTOM_H
