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

#ifndef OPS_HCCL_COMMON_H
#define OPS_HCCL_COMMON_H

#include <unordered_map>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <ccu/ccu_types.h>
#include <ccu/ccu_variable.hpp>
#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>
#include <acl/acl_rt.h>

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t INVALID_VALUE_RANKID = 0xFFFFFFFF;
constexpr uint32_t MAX_DATA_SIZE = 256 * 1024 * 1024; // 单次通信的最大数据量，256MB
constexpr uint64_t MAX_RANK_SIZE = 16;

struct OpParam {
    char tag[TAG_LENGTH];
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
    uint64_t count = 0;
    uint32_t root = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    ThreadHandle cpuThread;
    void *resCtx = nullptr; ///< 通信引擎上下文中的资源信息，存放 AlgResourceCtx 序列化后的内容
    uint64_t ctxSize = 0;
};

const std::unordered_map<HcclDataType, uint32_t> SIZE_TABLE = {{HCCL_DATA_TYPE_FP32, sizeof(float)}};

#endif // OPS_HCCL_COMMON_H
