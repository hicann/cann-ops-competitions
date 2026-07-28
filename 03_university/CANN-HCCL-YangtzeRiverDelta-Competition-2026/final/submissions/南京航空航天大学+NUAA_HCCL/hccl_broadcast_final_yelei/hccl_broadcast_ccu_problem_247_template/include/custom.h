/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <memory>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint64_t F2_SMALL_DIRECT_LIMIT = 512ULL * 1024ULL;
constexpr uint32_t F2_PREPARE_NOTIFY = 0;
constexpr uint32_t F2_SCATTER_NOTIFY = 1;
constexpr uint32_t F2_CROSS_NOTIFY = 2;
constexpr uint32_t F2_LOCAL_NOTIFY = 3;
constexpr uint32_t F2_PATH_NOTIFY_BASE = 1;
constexpr uint32_t F2_PATH_COUNT = 3;
constexpr uint32_t F2_CHUNK_COUNT = 8;
constexpr uint32_t F2_CHANNEL_NOTIFY_NUM = F2_PATH_NOTIFY_BASE + F2_PATH_COUNT;
constexpr uint32_t F6_MESH_CHANNEL_NOTIFY_NUM = 8;
constexpr uint64_t F6_MESH_SLICE_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint32_t F2_PIPELINE_CHUNK_COUNT = 8;
constexpr uint32_t F2_PIPELINE_MAX_RELAY_COUNT = 8;
constexpr uint32_t F2_PIPELINE_CHANNEL_NOTIFY_NUM = F2_PIPELINE_MAX_RELAY_COUNT;
constexpr uint32_t F4_FOREST_CHUNK_COUNT = 8;
constexpr uint32_t F4_FOREST_MAX_TREE_COUNT = 11;
constexpr uint32_t F4_FOREST_CHANNEL_NOTIFY_NUM = F4_FOREST_MAX_TREE_COUNT + 1;
constexpr uint32_t F4_INVALID_RANK = MAX_RANK_SIZE;
constexpr uint32_t F2_DIE_COUNT = 2;
constexpr uint32_t F2_WORKER_COUNT = 1;
constexpr uint32_t F2_THREAD_NOTIFY_NUM = 1;

enum F2TopologyMode : uint32_t {
    F2_TOPO_FLAT_FOUR = 0,
    F2_TOPO_TWO_SERVER = 1
};

enum F4AlgorithmMode : uint32_t {
    F4_ALG_F2_DEFAULT = 0,
    F4_ALG_CLOS_PIPELINE,
    F4_ALG_PIPELINED_FOREST,
    F6_ALG_MESH_SCATTER_ALLGATHER
};

enum F2KernelIndex : uint32_t {
    F2_KERNEL_PREPARE = 0,
    F2_KERNEL_DIRECT_0,
    F2_KERNEL_DIRECT_1,
    F2_KERNEL_SCATTER,
    F2_KERNEL_CROSS,
    F2_KERNEL_LOCAL,
    F2_KERNEL_PACKED,
    F2_KERNEL_CLOS_PIPELINE,
    F4_KERNEL_PIPELINED_FOREST,
    F6_KERNEL_MESH,
    F2_KERNEL_PER_DIE
};

constexpr uint32_t F2_KERNEL_COUNT = F2_DIE_COUNT * F2_KERNEL_PER_DIE;

struct CommBuffer {
    void *addr;
    uint64_t size;
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount = 0;
};

struct CcuKernelInfo {
    char kernelFuncName[64];
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

struct AlgResourceCtx {
    ThreadHandle ccuThread;
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint32_t> dieChannelCounts;
    uint32_t algorithmMode = F4_ALG_F2_DEFAULT;

    std::vector<char> Serialize()
    {
        BinaryStream stream;
        stream << ccuThread;
        stream << localBuffer;
        stream << threads;
        stream << ccuKernels;
        stream << dieChannelCounts;
        stream << algorithmMode;
        std::vector<char> result;
        stream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream stream(data);
        stream >> ccuThread;
        stream >> localBuffer;
        stream >> threads;
        stream >> ccuKernels;
        stream >> dieChannelCounts;
        stream >> algorithmMode;
    }
};

#endif // OPS_HCCL_CUSTOM_H
