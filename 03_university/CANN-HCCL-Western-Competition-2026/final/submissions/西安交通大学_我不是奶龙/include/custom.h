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

#include <memory>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

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

struct AlgResourceCtx {
    ThreadHandle ccuThread{};
    CommBuffer localBuffer{};
    std::vector<ThreadHandle> threads;
    // F081 probe layout: one generic fallback set plus three exact-size sets.
    // The exact sets share the already-acquired Channels and Threads; only
    // their CCU Kernel handles are independent.
    std::vector<CcuKernelHandle> ccuKernels;
    // Kept only so the unchanged generic fallback helpers can prove that no
    // optional parallel-RS mission was registered in this probe layout.
    std::vector<CcuKernelHandle> parallelRsKernels;
    std::vector<CcuKernelHandle> perf512KiBKernels;
    std::vector<CcuKernelHandle> perf512MiBKernels;
    std::vector<CcuKernelHandle> perf400MiB4BKernels;
    std::vector<uint32_t> kernelNetLayers;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << parallelRsKernels;
        binaryStream << perf512KiBKernels;
        binaryStream << perf512MiBKernels;
        binaryStream << perf400MiB4BKernels;
        binaryStream << kernelNetLayers;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> parallelRsKernels;
        binaryStream >> perf512KiBKernels;
        binaryStream >> perf512MiBKernels;
        binaryStream >> perf400MiB4BKernels;
        binaryStream >> kernelNetLayers;
    }
};

#endif // OPS_HCCL_CUSTOM_H
