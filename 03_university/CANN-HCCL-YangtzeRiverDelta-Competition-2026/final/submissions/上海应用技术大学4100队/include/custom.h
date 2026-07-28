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
#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

struct CcuKernelInfo {
    char kernelFuncName[64]{};
    void *kernelFunc = nullptr;
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

struct AlgResourceCtx {
    ThreadHandle ccuThread{};
    CommBuffer localBuffer{};
    aclrtStream primaryStream{};
    aclrtStream auxiliaryStream{};
    aclrtEvent startEvent{};
    aclrtEvent finishEvent{};
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> directKernels;
    std::vector<CcuKernelHandle> meshKernels;
    std::vector<CcuKernelHandle> scatterKernels;
    std::vector<CcuKernelHandle> gatherKernels;
    bool meshEnabled = false;
    bool stagedEnabled = false;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << primaryStream;
        binaryStream << auxiliaryStream;
        binaryStream << startEvent;
        binaryStream << finishEvent;
        binaryStream << threads;
        binaryStream << directKernels;
        binaryStream << meshKernels;
        binaryStream << scatterKernels;
        binaryStream << gatherKernels;
        binaryStream << meshEnabled;
        binaryStream << stagedEnabled;

        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> primaryStream;
        binaryStream >> auxiliaryStream;
        binaryStream >> startEvent;
        binaryStream >> finishEvent;
        binaryStream >> threads;
        binaryStream >> directKernels;
        binaryStream >> meshKernels;
        binaryStream >> scatterKernels;
        binaryStream >> gatherKernels;
        binaryStream >> meshEnabled;
        binaryStream >> stagedEnabled;
    }
};

#endif // OPS_HCCL_CUSTOM_H