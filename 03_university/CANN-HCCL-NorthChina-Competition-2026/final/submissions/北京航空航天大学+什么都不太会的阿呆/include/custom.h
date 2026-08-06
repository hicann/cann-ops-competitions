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
 #include <hccl/hccl_types.h>
 #include <hccl/hccl_res.h>
 
 #include "binary_stream.h"
 #include "common.h"
 
 constexpr uint32_t ALLGATHER_PARALLEL_LOCAL_RANK_NUM = 8;
 constexpr uint32_t ALLGATHER_PARALLEL_MAX_KERNEL_NUM = 4;
 constexpr uint32_t ALLGATHER_PARALLEL_MAX_INTER_CHANNEL_NUM = 8;
 constexpr uint32_t ALLGATHER_ASYM_8X4_RANK_NUM = 12;
 constexpr uint32_t ALLGATHER_ASYM_LARGE_SERVER_RANK_NUM = 8;
 constexpr uint32_t ALLGATHER_ASYM_SMALL_SERVER_RANK_NUM = 4;
 
 struct CommBuffer {
     void *addr = nullptr;
     uint64_t size = 0;
 };
 
 struct CcuKernelArgBase {
     ChannelHandle channels[MAX_RANK_SIZE] = {};
     uint32_t channelCount = 0;
 };
 
 // ccu kernel register所需信息
 struct CcuKernelInfo {
     // kernel名称
     char kernelFuncName[64] = {};
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
 
 struct AlgResourceCtx {
     ThreadHandle ccuThread{};                                   ///< 首次创建资源时使用的主CCU thread
     CommBuffer localBuffer{};                                   ///< 本端HCCL通信内存
     uint32_t parallel2x8Enabled = 0;                            ///< 是否构建了2x8 OmniPipe资源
     uint32_t asym8x4Enabled = 0;                                ///< 是否构建了8+4非对称OmniPipe资源
     uint32_t parallelCounterpartRank = MAX_RANK_SIZE;           ///< 另一Server中相同位置的Rank
     uint32_t parallelInterChannelCount = 0;                     ///< 同位置Rank间的聚合Channel数，OmniPipe固定为1
     std::vector<uint32_t> parallelLocalRanks;                   ///< 本Server中的8个Rank
     std::vector<ThreadHandle> threads;                          ///< 主Thread及并行Kernel使用的从Thread
     std::vector<CcuKernelHandle> ccuKernels;                    ///< Direct单切片Kernel，每个Die一个
     std::vector<CcuKernelHandle> groupCopyCcuKernels;           ///< 4x1大包Direct+GroupCopy Kernel
     std::vector<CcuKernelHandle> dualSliceCcuKernels;           ///< Direct双切片Kernel，每个Die一个
     std::vector<CcuKernelHandle> parallelPhase1IntraCcuKernels; ///< OmniPipe Stage 0的Server内Kernel
     std::vector<CcuKernelHandle> parallelPhase1InterCcuKernels; ///< OmniPipe Stage 0的Server间Kernel
     std::vector<CcuKernelHandle> parallelPhase2IntraCcuKernels; ///< OmniPipe Stage 1~3的Server内Kernel
     std::vector<CcuKernelHandle> parallelPhase2InterCcuKernels; ///< OmniPipe Stage 1~3的Server间Kernel
 
     // 序列化
     std::vector<char> Serialize()
     {
         BinaryStream binaryStream;
         binaryStream << ccuThread;
         binaryStream << localBuffer;
         binaryStream << parallel2x8Enabled;
         binaryStream << asym8x4Enabled;
         binaryStream << parallelCounterpartRank;
         binaryStream << parallelInterChannelCount;
         binaryStream << parallelLocalRanks;
         binaryStream << threads;
         binaryStream << ccuKernels;
         binaryStream << groupCopyCcuKernels;
         binaryStream << dualSliceCcuKernels;
         binaryStream << parallelPhase1IntraCcuKernels;
         binaryStream << parallelPhase1InterCcuKernels;
         binaryStream << parallelPhase2IntraCcuKernels;
         binaryStream << parallelPhase2InterCcuKernels;
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
         binaryStream >> parallel2x8Enabled;
         binaryStream >> asym8x4Enabled;
         binaryStream >> parallelCounterpartRank;
         binaryStream >> parallelInterChannelCount;
         binaryStream >> parallelLocalRanks;
         binaryStream >> threads;
         binaryStream >> ccuKernels;
         binaryStream >> groupCopyCcuKernels;
         binaryStream >> dualSliceCcuKernels;
         binaryStream >> parallelPhase1IntraCcuKernels;
         binaryStream >> parallelPhase1InterCcuKernels;
         binaryStream >> parallelPhase2IntraCcuKernels;
         binaryStream >> parallelPhase2InterCcuKernels;
     }
 };
 
 #endif // OPS_HCCL_CUSTOM_H
