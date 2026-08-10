/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

// GroupBroadcast 使用的 CCU Memory Slice 流水配置
constexpr uint64_t CUSTOM_CCU_MS_SIZE = 4096;
constexpr uint32_t CUSTOM_GROUP_BROADCAST_MS_INTERLEAVE = 8;
constexpr uint32_t CUSTOM_GROUP_BROADCAST_MS_PER_LOOP = 8;
constexpr uint64_t CUSTOM_GROUP_BROADCAST_BLOCK_SIZE =
    CUSTOM_CCU_MS_SIZE * CUSTOM_GROUP_BROADCAST_MS_PER_LOOP;
// CCU_SCHED 每个 IO Die 提供 16 个块式 LoopEngine、128 个块式 MS；
// 每路使用连续 8 个 MS 搬运 32KB，因此 16 路恰好用满可用资源
constexpr uint32_t CUSTOM_GROUP_BROADCAST_LOOP_COUNT = 16;

// 专用双轴算法中，Mesh 与 Clos 分别由不同 IO Die 上的 CCU Kernel 执行
constexpr uint32_t CUSTOM_AXIS_NONE = 0;
constexpr uint32_t CUSTOM_AXIS_MESH = 1;
constexpr uint32_t CUSTOM_AXIS_CLOS = 2;

constexpr uint64_t CUSTOM_AXIS_STAGE_DIRECT = 0;
constexpr uint64_t CUSTOM_AXIS_STAGE_1 = 1;
constexpr uint64_t CUSTOM_AXIS_STAGE_2 = 2;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// AllGather Mesh1D Mem2Mem 算法的 CCU Kernel 参数
struct CcuKernelArgAllGatherMesh1DMem2Mem : public CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    // 每个 channel 对应的对端 rankId，与 channels[] 同序
    uint32_t rankIds[MAX_RANK_SIZE];
    // 该 kernel 是否负责本 rank 槽位的本地拷贝（全通信域恰好一个 kernel 为 1）
    uint32_t ifHandleSelfRank;
    // 仅供 2×8/8+4 双轴 Kernel 使用；普通直写 Kernel 忽略这两个字段
    uint32_t axisId;
    uint32_t pairedRankId;
};

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
    // 每个 die 一份极简小数据拉取 Kernel，与 ccuKernels、kernelArgs 同序
    std::vector<CcuKernelHandle> smallReadCcuKernels;
    // 每个 die 分组一个 CCU Kernel 参数（与 ccuKernels 同序），含 channel 句柄表
    std::vector<CcuKernelArgAllGatherMesh1DMem2Mem> kernelArgs;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << smallReadCcuKernels;
        binaryStream << kernelArgs;
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
        binaryStream >> smallReadCcuKernels;
        binaryStream >> kernelArgs;
    }
};

#endif // OPS_HCCL_CUSTOM_H
