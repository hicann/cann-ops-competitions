/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstring>
#include <memory>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc = nullptr;
    // KernelArg实例指针
    void *kernelArg = nullptr;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

enum class KernelTrafficClass : uint8_t {
    INTRA_SERVER = 0,
    INTER_SERVER = 1,
};

struct AlgResourceCtx {
    static constexpr uint32_t WIRE_MAGIC = 0x41474343U;
    static constexpr uint32_t WIRE_VERSION = 8;
    static constexpr uint32_t MAX_KERNEL_COUNT = 4;
    static constexpr uint32_t MAX_THREAD_COUNT = 4;

    ThreadHandle ccuThread; ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer; ///< 本端HCCL通信内存
    ThreadHandle threads[MAX_THREAD_COUNT]{};
    CcuKernelHandle ccuKernels[MAX_KERNEL_COUNT]{};
    uint8_t kernelThreadIndex[MAX_KERNEL_COUNT]{};
    KernelTrafficClass kernelTrafficClass[MAX_KERNEL_COUNT]{};
    uint32_t threadCount = 0;
    uint32_t kernelCount = 0;
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    bool directSmallFastPath = false;
    bool directNhr = false;
    bool hierarchical2x8 = false;
    bool hierarchicalGrid = false;
    bool purePhysicalGrid = false;
    bool physical8Plus4 = false;

private:
    struct WireData {
        uint32_t magic = WIRE_MAGIC;
        uint32_t version = WIRE_VERSION;
        ThreadHandle ccuThread{};
        CommBuffer localBuffer{};
        ThreadHandle threads[MAX_THREAD_COUNT]{};
        CcuKernelHandle ccuKernels[MAX_KERNEL_COUNT]{};
        uint8_t kernelThreadIndex[MAX_KERNEL_COUNT]{};
        KernelTrafficClass kernelTrafficClass[MAX_KERNEL_COUNT]{};
        uint32_t threadCount = 0;
        uint32_t kernelCount = 0;
        uint32_t rankId = 0;
        uint32_t rankSize = 0;
        char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
        uint8_t directSmallFastPath = 0;
        uint8_t directNhr = 0;
        uint8_t hierarchical2x8 = 0;
        uint8_t hierarchicalGrid = 0;
        uint8_t purePhysicalGrid = 0;
        uint8_t physical8Plus4 = 0;
    };

public:
    std::vector<char> Serialize() const
    {
        if (threadCount > MAX_THREAD_COUNT || kernelCount > MAX_KERNEL_COUNT) {
            return {};
        }
        for (uint32_t index = 0; index < kernelCount; ++index) {
            if (kernelThreadIndex[index] >= threadCount ||
                kernelTrafficClass[index] > KernelTrafficClass::INTER_SERVER) {
                return {};
            }
        }
        WireData wire;
        wire.ccuThread = ccuThread;
        wire.localBuffer = localBuffer;
        wire.threadCount = threadCount;
        wire.kernelCount = kernelCount;
        wire.rankId = rankId;
        wire.rankSize = rankSize;
        std::memcpy(wire.commName, commName, sizeof(commName));
        wire.directSmallFastPath = static_cast<uint8_t>(directSmallFastPath);
        wire.directNhr = static_cast<uint8_t>(directNhr);
        wire.hierarchical2x8 = static_cast<uint8_t>(hierarchical2x8);
        wire.hierarchicalGrid = static_cast<uint8_t>(hierarchicalGrid);
        wire.purePhysicalGrid = static_cast<uint8_t>(purePhysicalGrid);
        wire.physical8Plus4 = static_cast<uint8_t>(physical8Plus4);
        for (uint32_t index = 0; index < threadCount; ++index) {
            wire.threads[index] = threads[index];
        }
        for (uint32_t index = 0; index < kernelCount; ++index) {
            wire.ccuKernels[index] = ccuKernels[index];
            wire.kernelThreadIndex[index] = kernelThreadIndex[index];
            wire.kernelTrafficClass[index] = kernelTrafficClass[index];
        }
        std::vector<char> result(sizeof(WireData));
        std::memcpy(result.data(), &wire, sizeof(WireData));
        return result;
    }

    bool DeSerialize(const void *data, uint64_t size)
    {
        if (data == nullptr || size != sizeof(WireData)) {
            return false;
        }
        WireData wire;
        std::memcpy(&wire, data, sizeof(WireData));
        if (wire.magic != WIRE_MAGIC || wire.version != WIRE_VERSION ||
            wire.threadCount > MAX_THREAD_COUNT || wire.kernelCount > MAX_KERNEL_COUNT) {
            return false;
        }
        for (uint32_t index = 0; index < wire.kernelCount; ++index) {
            if (wire.kernelThreadIndex[index] >= wire.threadCount ||
                wire.kernelTrafficClass[index] > KernelTrafficClass::INTER_SERVER) {
                return false;
            }
        }
        ccuThread = wire.ccuThread;
        localBuffer = wire.localBuffer;
        threadCount = wire.threadCount;
        kernelCount = wire.kernelCount;
        for (uint32_t index = 0; index < threadCount; ++index) {
            threads[index] = wire.threads[index];
        }
        for (uint32_t index = 0; index < kernelCount; ++index) {
            ccuKernels[index] = wire.ccuKernels[index];
            kernelThreadIndex[index] = wire.kernelThreadIndex[index];
            kernelTrafficClass[index] = wire.kernelTrafficClass[index];
        }
        rankId = wire.rankId;
        rankSize = wire.rankSize;
        std::memcpy(commName, wire.commName, sizeof(commName));
        commName[sizeof(commName) - 1] = '\0';
        directSmallFastPath = wire.directSmallFastPath != 0;
        directNhr = wire.directNhr != 0;
        hierarchical2x8 = wire.hierarchical2x8 != 0;
        hierarchicalGrid = wire.hierarchicalGrid != 0;
        purePhysicalGrid = wire.purePhysicalGrid != 0;
        physical8Plus4 = wire.physical8Plus4 != 0;
        return true;
    }
};

#endif // OPS_HCCL_CUSTOM_H
