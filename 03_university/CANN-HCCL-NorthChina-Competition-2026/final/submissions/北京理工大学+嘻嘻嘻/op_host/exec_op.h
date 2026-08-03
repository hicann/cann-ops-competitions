/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include <hccl/hcomm_primitives.h>

#include "custom.h"

namespace ops_hccl {
/**
 * @brief 根据本次调用参数和已注册资源上下文完成 AllGather CCU 任务编排
 * @param param 本次 AllGather 调用的动态参数
 * @param resCtx 已完成校验和反序列化的静态资源上下文
 * @return 执行成功返回 HCCL_SUCCESS，参数、资源或下发失败返回对应错误码
 */
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx);
}
#endif
