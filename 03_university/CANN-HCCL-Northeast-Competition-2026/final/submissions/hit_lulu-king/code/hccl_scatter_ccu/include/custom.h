#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <memory>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

// BEGIN_SPARSE_RATIO
inline uint64_t ScatterRelayElements(uint64_t count, uint32_t helpers, uint32_t remote)
{
    if (helpers == 0 || remote <= 4 || remote < helpers || remote - 4 >= helpers + 4) {
        return 0;
    }
    const uint64_t numerator = remote - 4, denominator = helpers + 4;
    // Compute floor(count * numerator / denominator) without overflowing.
    return (count / denominator) * numerator + (count % denominator) * numerator / denominator;
}
// END_SPARSE_RATIO

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t root = 0;
    uint64_t recvBytes = 0;
    uint32_t mode = 0;
    uint32_t kind = 0;
    uint32_t copyLocal = 0;
    uint32_t helperForRank[MAX_RANK_SIZE]{};
    uint64_t relayBytes = 0;
    uint64_t tileBytes = 0;
};

enum ScatterMode : uint32_t { SCATTER_DIRECT_PUSH = 0, SCATTER_PEER_PULL = 1 };

enum RelayKind : uint32_t {
    RELAY_ROOT = 1, RELAY_RECEIVER, RELAY_INIT_INPUT,
    RELAY_WAIT_0, RELAY_WAIT_1, RELAY_REUSE_0, RELAY_REUSE_1,
    RELAY_FINISH_INPUT, RELAY_INIT_OUTPUT, RELAY_FORWARD, RELAY_FINISH_OUTPUT,
    RELAY_RESIDENT_HELPER, RELAY_INPUT_ONESHOT, RELAY_OUTPUT_ONESHOT
};

using ScatterCcuKernelArg = CcuKernelArgBase;

struct CcuKernelInfo {
    char kernelFuncName[64]{};
    void *kernelFunc = nullptr;
    void *kernelArg = nullptr;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void SetKernelArg(const std::shared_ptr<T> &arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct CcuKernelLaunchEntry {
    CcuKernelHandle handle = 0;
    uint32_t threadIndex = 0;
    uint32_t channelCount = 0;
    uint32_t kind = 0;
};

struct AlgResourceCtx {
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelLaunchEntry> kernels;
    uint32_t mode = SCATTER_DIRECT_PUSH;
    uint32_t relayRole = 0; // 0: original path, 1: root, 2: helper, 3: receiver
    uint64_t relayBytes = 0;
    uint64_t tileBytes = 0;
    uint32_t targetCount = 0;
    uint32_t activeThreads = 0;
    struct ScratchBuffer {
        void *addr = nullptr;
        uint64_t size = 0;
    } scratch;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << threads;
        binaryStream << kernels;
        binaryStream << mode << relayRole << relayBytes << tileBytes << targetCount << activeThreads << scratch;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> threads;
        binaryStream >> kernels;
        binaryStream >> mode >> relayRole >> relayBytes >> tileBytes >> targetCount >> activeThreads >> scratch;
    }
};

#endif // OPS_HCCL_CUSTOM_H