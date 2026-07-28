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

// 极限提速：消灭 std::map 树形查询开销，转为 O(1) 的寄存器级匹配
inline uint64_t FastGetTypeSize(HcclDataType dataType) {
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8: case HCCL_DATA_TYPE_UINT8: return 1;
        case HCCL_DATA_TYPE_FP16: case HCCL_DATA_TYPE_BFP16: case HCCL_DATA_TYPE_INT16: case HCCL_DATA_TYPE_UINT16: return 2;
        case HCCL_DATA_TYPE_FP32: case HCCL_DATA_TYPE_INT32: case HCCL_DATA_TYPE_UINT32: return 4;
        case HCCL_DATA_TYPE_FP64: case HCCL_DATA_TYPE_INT64: case HCCL_DATA_TYPE_UINT64: return 8;
        default: {
            const auto it = SIZE_TABLE.find(dataType);
            return it != SIZE_TABLE.end() ? it->second : 0;
        }
    }
}

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

constexpr uint32_t MAX_DIE_KERNEL_NUM = 4;
constexpr uint32_t ALG_RESOURCE_CTX_MAGIC = 0x42564335U; // "BVC5"
constexpr uint32_t ALG_RESOURCE_CTX_VERSION = 82U;

template <typename T, uint32_t Capacity>
struct FixedArray {
    T values[Capacity]{};
    uint32_t count = 0;

    size_t size() const { return static_cast<size_t>(count); }
    bool empty() const { return count == 0; }
    void clear() { count = 0; }
    void reserve(size_t) {}
    void resize(size_t newSize) { count = static_cast<uint32_t>(newSize); }
    void push_back(const T &value) { values[count++] = value; }
    T &operator[](size_t index) { return values[index]; }
    const T &operator[](size_t index) const { return values[index]; }
    T *begin() { return values; }
    T *end() { return values + count; }
    const T *begin() const { return values; }
    const T *end() const { return values + count; }
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t root = 0;

    uint32_t prevRank = INVALID_VALUE_RANKID;
    uint32_t nextRank = INVALID_VALUE_RANKID;
    uint32_t prevChannelIndex = INVALID_VALUE_RANKID;
    uint32_t nextChannelIndex = INVALID_VALUE_RANKID;
    uint64_t pipelineChunkBytes = 0;
    uint32_t pipelineChunkCount = 0;
    uint64_t pipelineLastChunkBytes = 0;
};

using BroadcastCcuKernelArg = CcuKernelArgBase;

struct CcuKernelInfo {
    char kernelFuncName[64]{};
    void *kernelFunc = nullptr;
    void *kernelArg = nullptr;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(const std::shared_ptr<T> &arg) {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

constexpr uint32_t BCAST_ALG_DIRECT = 0;
constexpr uint32_t BCAST_ALG_SAG_SPLIT = 1;
constexpr uint32_t BCAST_ALG_CHAIN_PIPELINE = 2;
constexpr uint32_t BCAST_ALG_8P4_PREFETCH_SAG = 3;

struct CcuKernelLaunchEntry {
    uint32_t dieId = 0;
    uint32_t threadIndex = 0;
    uint32_t containsRootChannel = 0;
    uint32_t channelCount = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE]{};

    CcuKernelHandle directHandle = 0;
    CcuKernelHandle scatterHandle = 0;
    CcuKernelHandle allGatherHandle = 0;
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

    std::vector<char> Serialize() const {
        std::vector<char> result(sizeof(AlgResourceCtx));
        std::memcpy(result.data(), this, sizeof(AlgResourceCtx));
        return result;
    }

    bool DeSerialize(const void *data, uint64_t dataSize) {
        if (data == nullptr || dataSize != sizeof(AlgResourceCtx)) {
            return false;
        }
        std::memcpy(this, data, sizeof(AlgResourceCtx));
        return magic == ALG_RESOURCE_CTX_MAGIC && version == ALG_RESOURCE_CTX_VERSION;
    }
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "AlgResourceCtx must remain trivially copyable");

#endif // OPS_HCCL_CUSTOM_H