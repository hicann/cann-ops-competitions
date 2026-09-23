/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO ANY DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include "binary_stream.h"
#include "common.h"

// CCU Kernel 内部错误检查宏（模板 log.h 未提供，CCU 侧返回 CcuResult）
#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuRet = (call); \
        if (ccuRet != CCU_SUCCESS) { \
            HCCL_ERROR("[%s]call trace: ccuRet -> %d", __func__, static_cast<int32_t>(ccuRet)); \
            return ccuRet; \
        } \
    } while (0)

// channel 上变量交换(XN)槽位编号与信号位（对齐官方示例，从 0 起）
constexpr uint32_t XN_ID_OUTPUT = 0;
constexpr uint32_t XN_ID_TOKEN = 1;
constexpr uint32_t DATA_SIGNAL_ID = 2;
constexpr uint32_t CKE_IDX_0 = 0;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t IO_DIE_NUM = 2;
// Each bank contains only offloaded tails, whose aggregate is < one tile.
constexpr uint64_t RELAY_TILE_BYTES = 4U * 1024U * 1024U;
constexpr uint64_t RELAY_BANK_BYTES = 8U * 1024U * 1024U;
constexpr uint64_t COARSE_SLICE_LIMIT = 64U * 1024U * 1024U;
constexpr uint64_t EXTENDED_COARSE_SLICE_LIMIT = 128U * 1024U * 1024U;
constexpr uint32_t INVALID_IO_DIE = 0xFFFFFFFFU;
// Low bits retain the original per-kernel pull mask. A high bit records
// the copy owner for non-16-rank routes without changing the legacy layout.
constexpr uint32_t ROOT_COPY_INDEX_SHIFT = 31U;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

// One kernel owns channels from exactly one IO Die.  The root has one kernel
// per active Die; a receiver has a single kernel containing its root channel.
struct ScatterCcuKernelArg : public CcuKernelArgBase {
    uint32_t legacySchedule = 0; // Preserve v5 CCU scheduling for the 2x8 topology.
    uint32_t isRoot = 0;
    uint32_t copyLocalSlice = 0;
    uint32_t gatherCount = 0;
    uint32_t persistent = 0;
    uint32_t receiveRelayCount = 0;
    uint32_t noRootAck = 0; // v16: small pulls skip the root completion receipt.
    uint32_t remoteIndex = 0;
    ChannelHandle receiveRelayChannels[MAX_RANK_SIZE]{};
    uint32_t receiveRelayIndices[MAX_RANK_SIZE]{};
    uint32_t gatherRanks[MAX_RANK_SIZE]{};
    uint32_t pushMask = 0;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
};

struct RelayState {
    uint32_t enabled = 0;
    uint32_t persistent = 0;
    uint32_t role = 0; // 0 root, 1 local relay, 2 remote receiver
    uint32_t index = 0;
    uint64_t scratch = 0;
    uint32_t receiveDie = INVALID_IO_DIE;
    uint32_t pipeline = 0;
    uint32_t residentReceiver = 0;
    uint32_t coarse = 0;
};

struct RelayPlan : RelayState {
    std::vector<uint32_t> relays;
    std::vector<uint32_t> remotes;
    std::vector<uint32_t> fusedIndices;
    std::vector<CcuKernelHandle> forwardKernels;
    std::vector<ThreadHandle> forwardThreads;
    std::vector<uint32_t> forwardOnMain;
    std::vector<std::vector<uint32_t>> forwardIndices;
};

struct ResourceState {
    ThreadHandle workerThread{};
    uint32_t directPush = 0;
    uint32_t rootMainIndex = 0;
    uint32_t directPullMask = 0;
};

// Cached resources contain no owning pointers or dynamically allocated lists.
// Registration may use vectors; repeated launches restore one fixed object.
template <typename T, size_t Capacity> struct ResourceList {
    uint32_t count = 0;
    T values[Capacity]{};

    size_t size() const { return count; }
    bool empty() const { return count == 0U; }
    const T *begin() const { return values; }
    const T *end() const { return values + count; }
    const T &operator[](size_t index) const { return values[index]; }
    const T &front() const { return values[0]; }
    bool Valid() const { return count <= Capacity; }

    bool Assign(const std::vector<T> &source)
    {
        if (source.size() > Capacity) { return false; }
        count = source.size();
        std::copy(source.begin(), source.end(), values);
        return true;
    }
};

template <typename T, size_t Outer, size_t Inner>
bool FreezeLists(const std::vector<std::vector<T>> &source,
    ResourceList<ResourceList<T, Inner>, Outer> &target)
{
    if (source.size() > Outer) { return false; }
    target.count = source.size();
    for (size_t i = 0; i < source.size(); ++i) {
        if (!target.values[i].Assign(source[i])) { return false; }
    }
    return true;
}

struct FrozenRelayPlan : RelayState {
    ResourceList<uint32_t, MAX_RANK_SIZE> relays;
    ResourceList<uint32_t, MAX_RANK_SIZE> remotes;
    ResourceList<uint32_t, MAX_RANK_SIZE> fusedIndices;
    ResourceList<CcuKernelHandle, IO_DIE_NUM> forwardKernels;
    ResourceList<ThreadHandle, IO_DIE_NUM> forwardThreads;
    ResourceList<uint32_t, IO_DIE_NUM> forwardOnMain;
    ResourceList<ResourceList<uint32_t, MAX_RANK_SIZE>, IO_DIE_NUM> forwardIndices;
};

struct FrozenResourceCtx : ResourceState {
    static constexpr uint32_t MAGIC = 0x53434355U;
    static constexpr uint32_t VERSION = 20U;
    uint32_t magic = MAGIC;
    uint32_t version = VERSION;
    ResourceList<uint32_t, IO_DIE_NUM> directWriteMasks;
    ResourceList<ResourceList<uint32_t, MAX_RANK_SIZE>, IO_DIE_NUM> directPeers;
    ResourceList<CcuKernelHandle, IO_DIE_NUM> ccuKernels;
    FrozenRelayPlan relay;

    bool Valid() const
    {
        if (magic != MAGIC || version != VERSION || !ccuKernels.Valid() || !directPeers.Valid()
            || !directWriteMasks.Valid() || directPeers.size() != ccuKernels.size()
            || directWriteMasks.size() != ccuKernels.size()
            || (!ccuKernels.empty() && rootMainIndex >= ccuKernels.size())
            || !relay.relays.Valid() || !relay.remotes.Valid() || !relay.fusedIndices.Valid()
            || !relay.forwardKernels.Valid() || !relay.forwardThreads.Valid()
            || !relay.forwardOnMain.Valid() || !relay.forwardIndices.Valid()
            || relay.forwardKernels.size() != relay.forwardThreads.size()
            || relay.forwardKernels.size() != relay.forwardOnMain.size()
            || relay.forwardKernels.size() != relay.forwardIndices.size()) {
            return false;
        }
        for (const auto &peers : directPeers) {
            if (!peers.Valid()) { return false; }
        }
        for (const auto &indices : relay.forwardIndices) {
            if (!indices.Valid()) { return false; }
        }
        return true;
    }

    bool Load(const void *source, uint64_t bytes)
    {
        if (source == nullptr || bytes != sizeof(*this)) { return false; }
        std::memcpy(this, source, sizeof(*this));
        return Valid();
    }
};

static_assert(std::is_trivially_copyable<FrozenResourceCtx>::value, "resource cache must be byte-copyable");

struct AlgResourceCtx : ResourceState {
    std::vector<uint32_t> directWriteMasks;
    std::vector<std::vector<uint32_t>> directPeers;
    std::vector<CcuKernelHandle> ccuKernels; // 已注册的 CCU Kernel 句柄
    RelayPlan relay;

    // 序列化
    std::vector<char> Serialize()
    {
        FrozenResourceCtx frozen{};
        static_cast<ResourceState &>(frozen) = static_cast<const ResourceState &>(*this);
        static_cast<RelayState &>(frozen.relay) = static_cast<const RelayState &>(relay);
        if (!frozen.directWriteMasks.Assign(directWriteMasks) || !FreezeLists(directPeers, frozen.directPeers)
            || !frozen.ccuKernels.Assign(ccuKernels) || !frozen.relay.relays.Assign(relay.relays)
            || !frozen.relay.remotes.Assign(relay.remotes) || !frozen.relay.fusedIndices.Assign(relay.fusedIndices)
            || !frozen.relay.forwardKernels.Assign(relay.forwardKernels)
            || !FreezeLists(relay.forwardIndices, frozen.relay.forwardIndices)
            || !frozen.relay.forwardThreads.Assign(relay.forwardThreads)
            || !frozen.relay.forwardOnMain.Assign(relay.forwardOnMain) || !frozen.Valid()) {
            return {};
        }
        std::vector<char> result(sizeof(frozen));
        std::memcpy(result.data(), &frozen, sizeof(frozen));
        return result;
    }
};

#endif // OPS_HCCL_CUSTOM_H
