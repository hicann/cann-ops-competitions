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

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <vector>

#include <ccu/ccu_types.h>

#include "log.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

// 检查 CCU 接口返回值，非成功则记录日志并返回该错误码
#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call)                                                     \
    do {                                                                      \
        CcuResult ccuRet = (call);                                            \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) {                                \
            HCCL_ERROR("[%s] call trace: ccuRet -> %d", __func__, ccuRet);    \
            return ccuRet;                                                    \
        }                                                                     \
    } while (0)
#endif // CCU_CHK_RET

namespace ops_hccl {

// CCU Kernel 函数：普通拓扑执行 Mesh1D Mem2Mem，2×8 执行直写或双轴两阶段算法
CcuResult CcuKernel(CcuKernelArg arg);

// 极小数据拉取 Kernel：各 rank 主动读取 peer input，避免 MS 广播与完整尾同步
CcuResult CcuKernelSmallRead(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
